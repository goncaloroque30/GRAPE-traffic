import logging
import sqlite3
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pandas as pd
from pitot.isa import temperature
from traffic import cache_dir
from traffic.data import anp_data, client, metars

from .AirportTraffic import AirportTraffic

_log = logging.getLogger(__name__)

# -- Constants --
scenario_id = "All"
performance_id = "Performance"
noise_id = "Noise"
emissions_id = "Emissions"
emissions_id_lto_cycle = "Emissions LTO Cycle"


class GrapeTraffic:
    """GrapeTraffic uses processed trajectories in an `AirportTraffic` instance to generate a GRAPE study and calculate airport environmental impacts with GRAPE.
    Two separate uses are considered:
        - the `run` method does everything: generate the input tables, create the GRAPE study and perform the calculations.
        - to split the process, you can call `GrapeTraffic.generate_tables`, followed by `GrapeTraffic.generate_grape()` and `GrapeTraffic.run(force_overwrite=False)`.
          This can be useful to perform intermediary steps, e.g. edit the GRAPE tables before generating the GRAPE study.

    Args:
        apt_traffic: the arrivals and/or departures to process and add to the GRAPE study.
        receptors_path: a file path to the list of receptors for which to calculate noise. Defaults to `None`.
            If any of the columns `Receptor`, `Latitude`, `Longitude`, `AltitudeMSL_ft` are not found, an error will be raised.
            None will not add any receptors, meaning no noise will be calculated.
        study_path: the path to the GRAPE study which will be created and used. Defaults to `study.grp` in the working directory.
    """

    eedb_url: ClassVar[str] = "https://www.easa.europa.eu/en/downloads/131424/en"
    """The URL to the EEDB database."""

    lto_path: ClassVar[Path] = Path(__file__).parent / "LTO.csv"
    """File path to the association between aircraft IDs and LTO IDs (EEDB UID or FOI ID). M"""

    eedb_path: ClassVar[Path | None] = None
    """File path to a user specified EEDB database."""

    foi_path: ClassVar[Path | None] = None
    """File path to the user obtained FOI database."""

    grape_exe: ClassVar[Path | None] = None
    """File path to GRAPE executable. The latest version is available under https://goncaloroque30.github.io/GRAPE-Docs/."""

    lto_emissions_cutoff: ClassVar[float] = 3000.0
    """Controls up to which altitude emissions are calculated."""

    cache_dir: ClassVar[Path]

    def __init__(
        self,
        apt_traffic: AirportTraffic,
        receptors_path: str | Path | None,
        study_path: str | Path = Path().cwd() / "study.grp",
    ) -> None:
        from . import config_file

        if self.grape_exe is None:
            raise RuntimeError(f"Please provide the path to the GRAPE executable in '{config_file.as_posix()}'.")

        if apt_traffic.arr is None and apt_traffic.dep is None:
            raise RuntimeError("No arrivals or departures provided.")

        self.receptors: pd.DataFrame | None = None
        if receptors_path:
            self.receptors = pd.read_csv(Path(receptors_path))
            if any(
                (name := c) not in self.receptors.columns
                for c in ("Receptor", "Latitude", "Longitude", "AltitudeMSL_ft")
            ):
                raise RuntimeError(f"The receptor list provided does not contain column {name}.")

        self.apt_traffic: AirportTraffic = apt_traffic
        self.grape_tables: dict[str, pd.DataFrame] = {}
        self.grape_path: Path = Path(study_path)

        self.lto_data = pd.read_csv(self.lto_path)
        if any(
            (name := c) not in self.lto_data
            for c in (
                "ID",
                "LTO ID",
                "Source",
                "Maximum Sea Level Static Thrust (lbf)",
            )
        ):
            raise RuntimeError(f"The receptor list provided does not contain column {name}.")
        self.lto_data = self.lto_data.rename(
            columns={
                "ID": "id",
                "LTO ID": "lto_engine_id",
                "Source": "source",
                "Maximum Sea Level Static Thrust (lbf)": "maximum_sea_level_static_thrust",
            }
        )

    def generate_tables(self) -> None:
        """Generate the GRAPE tables and store them in `self.grape_tables`, overwriting any existing data."""
        self._generate_tables_lto()
        self._generate_tables_fleet()
        self._generate_tables_operations()
        self._generate_tables_scenarios()

    def generate_grape(self, overwrite: bool = False) -> None:
        """Creates an empty GRAPE study, imports the ANP data into it and fills the GRAPE tables with the data in `self.grape_tables`.
           If `self.grape_tables` has no entries, calls the generate_tables() method.

        Args:
            overwrite: if the file in `self.grape_path` exists and `overwrite` is false, an error will be thrown. Defaults to False.
        """
        if self.grape_path.exists() and not overwrite:
            raise RuntimeError(f"The grape study at {self.grape_path} already exists.")

        if not self.grape_tables:
            self.generate_tables()

        self.grape_path.parent.mkdir(parents=True, exist_ok=True)

        anp_file = cache_dir / "anp.zip"
        anp_folder = cache_dir / "anp"
        # Unzip files to cache folder if it does not yet exists
        if not anp_folder.exists():
            with zipfile.ZipFile(anp_file) as zip:
                zip.extractall(anp_folder)

        # -- Create GRAPE File and Import Data --
        # Create
        output = subprocess.Popen(
            f'"{self.grape_exe}" -x -c "{self.grape_path}" -anp "{anp_folder}"',
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout
        if output is not None:
            print(output.read().decode())

        # SQL Connection
        con = sqlite3.connect(self.grape_path)

        [
            tbl.to_sql(name=f"{tbl_name}", con=con, if_exists="append", index=False)
            for tbl_name, tbl in self.grape_tables.items()
        ]
        # Finalize
        con.commit()
        con.close()

    def run(self, force_overwrite: bool = True) -> None:
        """Call GRAPE to perform all the calculation runs.

        Args:
            force_overwrite: if set to `False` and the file in `self.grape_path` already exists, use the existing file. Defaults to True.
        """
        if not self.grape_path.exists() or force_overwrite:
            self.generate_grape(overwrite=True)

        cmd = [
            f"{GrapeTraffic.grape_exe}",
            "-x",
            "-o",
            f"{self.grape_path.resolve()}",
            "-rp",
            f"{scenario_id}-{performance_id}",
            "-rn",
            f"{scenario_id}-{performance_id}-{noise_id}",
            "-re",
            f"{scenario_id}-{performance_id}-{emissions_id}",
            "-re",
            f"{scenario_id}-{performance_id}-{emissions_id_lto_cycle}",
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    def _generate_tables_lto(self) -> None:
        # EEDB
        eedb = self._get_eedb()
        eedb_nvpm = self._get_eedb_nvpm()

        # EEDB: Apply Doc 9889 missing SN values approach
        def _doc9889(idx, scale_factors):
            sn_col = eedb.columns.get_loc("SN T/O")
            for sf in scale_factors:
                idx_mode = idx & eedb.iloc[:, sn_col].isnull()  # type: ignore
                eedb.iloc[idx_mode, sn_col] = eedb.loc[idx_mode, "SN Max"] * sf  # type: ignore
                sn_col += 1  # type: ignore

        _doc9889(eedb["Manufacturer"] == "Aviadvigatel", [1.0, 1.0, 0.8, 0.3])
        _doc9889(
            (eedb["Manufacturer"] == "General Electric Company")
            & eedb["Engine Identification"].str.startswith("CF34", na=False),
            [1.0, 0.4, 0.3, 0.3],
        )
        _doc9889(eedb["Manufacturer"] == "Textron Lycoming", [1.0, 1.0, 0.6, 0.3])
        _doc9889(
            (eedb["Manufacturer"] == "CFM International") & eedb["Combustor Description"].str.contains("DAC", na=False),
            [0.3, 0.3, 0.3, 1.0],
        )
        _doc9889(pd.Series(True, index=eedb.index), [1.0, 0.9, 0.3, 0.3])

        # EEDB: Correct
        eedb["mixed_nozzle"] = eedb["Eng Type"] == "MTF"
        eedb = eedb.drop(
            columns=[
                "Eng Type",
                "Manufacturer",
                "Engine Identification",
                "Combustor Description",
                "SN Max",
            ]
        )

        # LTO Association
        eedb_data = (
            (
                self.lto_data.loc[self.lto_data["source"] == "EEDB", ["id", "lto_engine_id"]]
                .merge(
                    eedb,
                    how="left",
                    left_on="lto_engine_id",
                    right_on="UID No",
                )
                .drop(columns="UID No")
            )
            .merge(
                eedb_nvpm,
                how="left",
                left_on="lto_engine_id",
                right_on="UID No",
            )
            .drop(columns="UID No")
        )
        eedb_data["maximum_sea_level_static_thrust"] = eedb_data["Rated Thrust (kN)"] * 1000.0  # kN 2 N
        eedb_data = eedb_data.drop(columns=["Rated Thrust (kN)"])

        # FOI
        if self.foi_path is not None:
            foi = pd.read_excel(self.foi_path)
            foi["ENGINE_ID"] = foi["ENGINE_ID"].astype("str")

            foi_data = (
                self.lto_data.loc[self.lto_data["source"] == "FOI"]
                .merge(
                    foi,
                    how="left",
                    left_on="lto_engine_id",
                    right_on="ENGINE_ID",
                )
                .drop(columns=["ENGINE_ID", "source"])
            )
            foi_data["maximum_sea_level_static_thrust"] = (
                foi_data["maximum_sea_level_static_thrust"] * 9.80665 / 2.204623  # lbf to N
            )

            lto_grape = pd.concat(
                [
                    eedb_data.drop(columns=["id"])
                    .rename(columns={"lto_engine_id": "id"})
                    .drop_duplicates(subset=["id"]),
                    foi_data.drop(columns=["id"])
                    .rename(columns={"lto_engine_id": "id"})
                    .drop_duplicates(subset=["id"]),
                ]
            )
        else:
            lto_grape = (
                eedb_data.drop(columns=["id"]).rename(columns={"lto_engine_id": "id"}).drop_duplicates(subset=["id"])
            )

        for col in lto_grape.columns:
            if "(g/kg)" in col:
                lto_grape[col] /= 1e3  # g/kg 2 kg/kg
            elif "(mg/kg)" in col:
                lto_grape[col] /= 1e6  # mg/kg 2 kg/kg
        for col in ["B/P Ratio", "SN T/O", "SN C/O", "SN App", "SN Idle"]:
            lto_grape[col] = lto_grape[col].infer_objects().fillna(0.0)
        lto_grape["mixed_nozzle"] = lto_grape["mixed_nozzle"].infer_objects().fillna(False)

        lto_grape = lto_grape.rename(
            columns={
                "B/P Ratio": "bypass_ratio",
                "HC EI T/O (g/kg)": "emission_index_hc_takeoff",
                "HC EI C/O (g/kg)": "emission_index_hc_climb_out",
                "HC EI App (g/kg)": "emission_index_hc_approach",
                "HC EI Idle (g/kg)": "emission_index_hc_idle",
                "CO EI T/O (g/kg)": "emission_index_co_takeoff",
                "CO EI C/O (g/kg)": "emission_index_co_climb_out",
                "CO EI App (g/kg)": "emission_index_co_approach",
                "CO EI Idle (g/kg)": "emission_index_co_idle",
                "NOx EI T/O (g/kg)": "emission_index_nox_takeoff",
                "NOx EI C/O (g/kg)": "emission_index_nox_climb_out",
                "NOx EI App (g/kg)": "emission_index_nox_approach",
                "NOx EI Idle (g/kg)": "emission_index_nox_idle",
                "SN T/O": "smoke_number_takeoff",
                "SN C/O": "smoke_number_climb_out",
                "SN App": "smoke_number_approach",
                "SN Idle": "smoke_number_idle",
                "Fuel Flow T/O (kg/sec)": "fuel_flow_takeoff",
                "Fuel Flow C/O (kg/sec)": "fuel_flow_climb_out",
                "Fuel Flow App (kg/sec)": "fuel_flow_approach",
                "Fuel Flow Idle (kg/sec)": "fuel_flow_idle",
                "nvPM EImass_SL T/O (mg/kg)": "emission_index_nvpm_takeoff",
                "nvPM EImass_SL C/O (mg/kg)": "emission_index_nvpm_climb_out",
                "nvPM EImass_SL App (mg/kg)": "emission_index_nvpm_approach",
                "nvPM EImass_SL Idle (mg/kg)": "emission_index_nvpm_idle",
                "nvPM EInum_SL T/O (#/kg)": "emission_index_nvpm_number_takeoff",
                "nvPM EInum_SL C/O (#/kg)": "emission_index_nvpm_number_climb_out",
                "nvPM EInum_SL App (#/kg)": "emission_index_nvpm_number_approach",
                "nvPM EInum_SL Idle (#/kg)": "emission_index_nvpm_number_idle",
            }
        ).reset_index(drop=True)

        self.grape_tables["lto_fuel_emissions"] = lto_grape

    def _generate_tables_fleet(self) -> None:
        acft = self.apt_traffic.list_aircraft()
        assert isinstance(acft, pd.DataFrame)  # guaranteed

        assert hasattr(anp_data, "aircraft")  # anp aircraft table should exist
        anp_subs = anp_data.substitution
        anp_subs["aircraft_id"] = anp_subs.apply(
            lambda r: f"{r['ICAO_CODE']} {r['AIRCRAFT_VARIANT']}".strip(),
            axis="columns",
        )

        fleet = (
            acft[["aircraft_id", "anp"]]
            .drop_duplicates(subset=["aircraft_id"])
            .merge(
                anp_data.aircraft[["ACFT_ID", "Number Of Engines", "NPD_ID", "Power Parameter"]],  # type: ignore (anp aircraft table guaranteed to exist)
                how="left",
                left_on="anp",
                right_on="ACFT_ID",
            )
            .merge(
                anp_subs[
                    [
                        "aircraft_id",
                        "DELTA_APP_dB",
                        "DELTA_DEP_dB",
                    ]
                ],
                how="left",
                on="aircraft_id",
            )
            .rename(
                columns={
                    "aircraft_id": "id",
                    "Number Of Engines": "engine_count",
                    "ACFT_ID": "doc29_performance_id",
                    "NPD_ID": "doc29_noise_id",
                    "DELTA_APP_dB": "doc29_noise_arrival_delta_db",
                    "DELTA_DEP_dB": "doc29_noise_departure_delta_db",
                }
            )
            .assign(sfi_id=None)
            .merge(
                self.lto_data[["id", "lto_engine_id"]],
                on="id",
                how="left",
            )
        )

        fleet["engine_count"] = pd.to_numeric(fleet["engine_count"])

        def _update_npd_id(row):
            if any(val in row["Power Parameter"].lower() for val in ["percent", "%"]):
                return f"{row['doc29_noise_id']} {row['doc29_performance_id']}"
            return row["doc29_noise_id"]

        new_doc29_noise_id = fleet.apply(_update_npd_id, axis="columns")
        fleet["doc29_noise_id"] = new_doc29_noise_id
        fleet = fleet.drop(columns=["anp", "Power Parameter"])

        # Warn if aircraft were not successfully associated with LTO data
        fleet_lto_na = fleet["lto_engine_id"].isna()
        if fleet_lto_na.sum():
            _log.warning(f"The aircraft {','.join(fleet.loc[fleet_lto_na, 'id'])} have no LTO engine associated.")

        self.grape_tables["fleet"] = fleet

    def _generate_tables_operations(self) -> None:
        flights = self.apt_traffic.list_flights()
        assert isinstance(flights, pd.DataFrame)  # guaranteed

        operations_tracks_4d = (
            flights.rename(
                columns={
                    "flight_id": "id",
                    "aircraft_id": "fleet_id",
                }
            )
            .drop(
                columns=[
                    "runway",
                    "icao24",
                    "origin",
                    "destination",
                    "cumdist_max",
                    "go_around",
                    "changed_runway",
                ]
            )
            .assign(count=1)
        )

        operations_tracks_4d["time"] = operations_tracks_4d["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        self.grape_tables["operations_tracks_4d"] = operations_tracks_4d

        def _convert_traffic(df: pd.DataFrame) -> pd.DataFrame:
            ret = (
                df[
                    [
                        "flight_id",
                        "timestamp",
                        "cumdist",
                        "longitude",
                        "latitude",
                        "altitude",
                        "TAS",
                        "groundspeed",
                        "thrust",
                    ]
                ]
                .rename(
                    columns={
                        "flight_id": "operation_id",
                        "timestamp": "time",
                        "cumdist": "cumulative_ground_distance",
                        "altitude": "altitude_msl",
                        "TAS": "true_airspeed",
                        "thrust": "corrected_net_thrust_per_engine",
                    }
                )
                .assign(
                    point_number=lambda x: x.groupby(["operation_id"]).cumcount() + 1,
                    bank_angle=0.0,
                    fuel_flow_per_engine=0.0,
                )
            )
            ret["time"] = pd.to_datetime(ret["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            ret["cumulative_ground_distance"] *= 1852  # NM 2 m
            ret["altitude_msl"] *= 0.3048  # ft 2 m
            ret["true_airspeed"] *= 1852.0 / 3600.0  # kts 2 m/s
            ret["groundspeed"] *= 1852.0 / 3600.0  # kts 2 m/s
            ret["corrected_net_thrust_per_engine"] *= 9.80665 / 2.204623  # lbf to N
            return ret

        cumul = []
        if self.apt_traffic.arr is not None:
            cumul.append(
                _convert_traffic(self.apt_traffic.arr.data).assign(
                    operation="Arrival",
                    flight_phase="Approach",
                )
            )

        if self.apt_traffic.dep is not None:
            cumul.append(
                _convert_traffic(self.apt_traffic.dep.data).assign(
                    operation="Departure",
                    flight_phase=self.apt_traffic.dep.data["flight_phase"],
                )
            )
        operations_tracks_4d_points = pd.concat(cumul, ignore_index=True)
        self.grape_tables["operations_tracks_4d_points"] = operations_tracks_4d_points

    def _generate_tables_scenarios(self):
        flights = self.apt_traffic.list_flights()
        assert isinstance(flights, pd.DataFrame)  # guaranteed

        # scenarios
        self.grape_tables["scenarios"] = pd.DataFrame({"id": [scenario_id]})

        # scenarios_tracks_4d
        operations_tracks_4d = self.grape_tables["operations_tracks_4d"]
        self.grape_tables["scenarios_tracks_4d"] = (
            operations_tracks_4d[["id", "operation"]]
            .assign(scenario_id=scenario_id)
            .rename(columns={"id": "operation_id"})
        )

        # performance_run
        self.grape_tables["performance_run"] = pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "id": performance_id,
                    "coordinate_system_type": "Geodesic WGS84",
                    "coordinate_system_longitude_0": None,
                    "coordinate_system_latitude_0": None,
                    "filter_minimum_altitude": None,
                    "filter_maximum_altitude": None,
                    "filter_minimum_cumulative_ground_distance": None,
                    "filter_maximum_cumulative_ground_distance": None,
                    "filter_ground_distance_threshold": None,
                    "segmentation_speed_delta_threshold": None,
                    "flights_performance_model": "Doc29",
                    "flights_doc29_low_altitude_segmentation": True,
                    "tracks_4d_calculate_performance": True,
                    "tracks_4d_minimum_points": None,
                    "tracks_4d_recalculate_time": False,
                    "tracks_4d_recalculate_cumulative_ground_distance": False,
                    "tracks_4d_recalculate_groundspeed": False,
                    "tracks_4d_recalculate_fuel_flow": True,
                    "fuel_flow_model": "LTO Doc9889",
                    "fuel_flow_lto_altitude_correction": True,
                }
            ]
        )

        # performance_run_atmospheres
        metar_data = []
        if self.apt_traffic.arr is not None:
            spans = (
                self.apt_traffic.arr.data[["destination", "timestamp"]]
                .groupby("destination")
                .agg({"timestamp": ["min", "max"]})
            )
            spans.columns = spans.columns.map(" ".join)  # multilevel columns into single level
            spans = spans.reset_index()
            [
                metar_data.append(
                    metars.fetch(
                        row["destination"],
                        row["timestamp min"],
                        row["timestamp max"],
                    )
                )
                for _, row in spans.iterrows()
            ]
        if self.apt_traffic.dep is not None:
            spans = (
                self.apt_traffic.dep.data[["origin", "timestamp"]].groupby("origin").agg({"timestamp": ["min", "max"]})
            )
            spans.columns = spans.columns.map(" ".join)  # multilevel columns into single level
            spans = spans.reset_index()
            [
                metar_data.append(
                    metars.fetch(
                        row["origin"],
                        row["timestamp min"],
                        row["timestamp max"],
                    )
                )
                for _, row in spans.iterrows()
            ]
        metar_data = [m.data for m in metar_data if m is not None]

        performance_run_atmospheres = pd.concat(metar_data)[
            [
                "valid",
                "wind_speed",
                "wind_direction",
                "relative_humidity",
                "elevation",
                "temperature",
                "sea_level_pressure",
            ]
        ].drop_duplicates(subset=["valid"])
        # GRAPE does not support multiple METAR stations yet
        performance_run_atmospheres.loc[:, "time"] = performance_run_atmospheres.loc[:, "valid"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        performance_run_atmospheres.loc[:, "wind_speed"] *= 1852.0 / 3600.0  # kts 2 m/s
        performance_run_atmospheres.loc[:, "relative_humidity"] /= 100.0  # 0-100 2 0-1
        performance_run_atmospheres.loc[:, "temperature"] += 273.15  # °C 2 K
        performance_run_atmospheres.loc[:, "sea_level_pressure"] *= 100.0  # hPa 2 Pa
        performance_run_atmospheres = performance_run_atmospheres.assign(
            scenario_id=scenario_id,
            performance_run_id=performance_id,
            temperature_delta=(
                performance_run_atmospheres["temperature"] - temperature(performance_run_atmospheres["elevation"])
            ),
            pressure_delta=performance_run_atmospheres["sea_level_pressure"] - 101325,
        ).drop(columns=["valid", "elevation", "temperature", "sea_level_pressure"])

        self.grape_tables["performance_run_atmospheres"] = performance_run_atmospheres

        # noise_run
        self.grape_tables["noise_run"] = pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "performance_run_id": performance_id,
                    "id": noise_id,
                    "noise_model": "Doc29",
                    "atmospheric_absorption": "SAE ARP 5534",
                    "receptor_set_type": "Points",
                    "save_single_event_metrics": True,
                }
            ]
        )

        # noise_run_receptor_points
        if self.receptors is not None:
            noise_run_receptor_points = self.receptors.rename(
                columns={
                    "Receptor": "id",
                    "Longitude": "longitude",
                    "Latitude": "latitude",
                    "AltitudeMSL_ft": "altitude_msl",
                }
            )[["id", "longitude", "latitude", "altitude_msl"]]
            noise_run_receptor_points["altitude_msl"] *= 0.3048  # ft 2 m
            noise_run_receptor_points["scenario_id"] = scenario_id
            noise_run_receptor_points["performance_run_id"] = performance_id
            noise_run_receptor_points["noise_run_id"] = noise_id
            self.grape_tables["noise_run_receptor_points"] = noise_run_receptor_points

        # noise_run_cumulative_metrics
        self.grape_tables["noise_run_cumulative_metrics"] = pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "performance_run_id": performance_id,
                    "noise_run_id": noise_id,
                    "id": "Total Exposure",
                    "start_time": (flights["time"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": (flights["time"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                }
            ]
        )

        # noise_run_cumulative_metrics_weights
        self.grape_tables["noise_run_cumulative_metrics_weights"] = pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "performance_run_id": performance_id,
                    "noise_run_id": noise_id,
                    "noise_run_cumulative_metric_id": "Total Exposure",
                    "time_of_day": "00:00:00",
                    "weight": 1.0,
                }
            ]
        )

        # noise_run_cumulative_metrics_number_above_thresholds
        dict_noise_run_thr = {
            "scenario_id": scenario_id,
            "performance_run_id": performance_id,
            "noise_run_id": noise_id,
            "noise_run_cumulative_metric_id": "Total Exposure",
            "threshold": [65, 70, 75, 80, 85],
        }
        self.grape_tables["noise_run_cumulative_metrics_number_above_thresholds"] = (
            pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_noise_run_thr.items()])).infer_objects().ffill()
        )

        # Emissions_run
        dict_emi_run = {
            "scenario_id": scenario_id,
            "performance_run_id": performance_id,
            "id": emissions_id,
            "calculate_gas_emissions": True,
            "calculate_particle_emissions": True,
            "emissions_model": "Segments",
            "bffm2_gas_emission_indexes": True,
            "emissions_model_particles_smoke_number": "FOA 4",
            "save_segment_results": True,
            "lto_cycle_idle": 0.0,
            "lto_cycle_approach": 0.0,
            "lto_cycle_climb_out": 0.0,
            "lto_cycle_takeoff": 0.0,
            "emissions_maximum_altitude": self.lto_emissions_cutoff * 0.3048,  # ft 2 m,
        }
        dict_emi_run_lto_cycle = dict_emi_run.copy()
        dict_emi_run_lto_cycle["id"] = emissions_id_lto_cycle
        dict_emi_run_lto_cycle["emissions_model"] = "LTO Cycle"
        dict_emi_run_lto_cycle["lto_cycle_idle"] = 0
        dict_emi_run_lto_cycle["lto_cycle_approach"] = 240
        dict_emi_run_lto_cycle["lto_cycle_climb_out"] = 132
        dict_emi_run_lto_cycle["lto_cycle_takeoff"] = 42
        dict_emi_run_lto_cycle["emissions_maximum_altitude"] = None

        self.grape_tables["emissions_run"] = pd.DataFrame([dict_emi_run, dict_emi_run_lto_cycle])

    def _get_eedb(self) -> pd.DataFrame:
        return pd.read_excel(
            self._get_eedb_path(),
            sheet_name="Gaseous Emissions and Smoke",
            usecols=[
                "UID No",
                "Manufacturer",
                "Engine Identification",
                "Combustor Description",
                "Rated Thrust (kN)",
                "Eng Type",
                "B/P Ratio",
                "Fuel Flow T/O (kg/sec)",
                "Fuel Flow C/O (kg/sec)",
                "Fuel Flow App (kg/sec)",
                "Fuel Flow Idle (kg/sec)",
                "HC EI T/O (g/kg)",
                "HC EI C/O (g/kg)",
                "HC EI App (g/kg)",
                "HC EI Idle (g/kg)",
                "CO EI T/O (g/kg)",
                "CO EI C/O (g/kg)",
                "CO EI App (g/kg)",
                "CO EI Idle (g/kg)",
                "NOx EI T/O (g/kg)",
                "NOx EI C/O (g/kg)",
                "NOx EI App (g/kg)",
                "NOx EI Idle (g/kg)",
                "SN T/O",
                "SN C/O",
                "SN App",
                "SN Idle",
                "SN Max",
            ],
        )

    def _get_eedb_nvpm(self) -> pd.DataFrame:
        return pd.read_excel(
            self._get_eedb_path(),
            sheet_name="nvPM Emissions",
            usecols=[
                "UID No",
                "nvPM EImass_SL T/O (mg/kg)",
                "nvPM EImass_SL C/O (mg/kg)",
                "nvPM EImass_SL App (mg/kg)",
                "nvPM EImass_SL Idle (mg/kg)",
                "nvPM EInum_SL T/O (#/kg)",
                "nvPM EInum_SL C/O (#/kg)",
                "nvPM EInum_SL App (#/kg)",
                "nvPM EInum_SL Idle (#/kg)",
            ],
        )

    def _get_eedb_path(self) -> Path:
        if self.eedb_path is not None:
            return self.eedb_path
        else:
            p = self.cache_dir / "eedb.xlsx"

            if not p.exists():
                f = client.get(self.eedb_url)
                buffer = BytesIO()
                [buffer.write(chunk) for chunk in f.iter_bytes(chunk_size=1024)]
                with open(p, "wb") as f:
                    f.write(buffer.getbuffer())

            return p
