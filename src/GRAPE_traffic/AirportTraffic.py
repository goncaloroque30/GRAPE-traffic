from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from pitot.geodesy import distance
from scipy.stats import linregress
from traffic.core.flight import Flight, Position
from traffic.core.traffic import Traffic
from traffic.data import anp_data, runways
from traffic.data.basic.runways import RunwayAirport


class AirportTraffic:
    """AirportTraffic is responsible for filtering, smoothing and adding features to airport trajectories (arrivals and departures).
    The objective is to process all trajectories so they can be used in environmental impact estimation.

    Args:
        arrivals: the arrivals trajectories to process.
        departures: the departure trajectories to process.
    """

    altitude_cutoff: ClassVar[float] = 10000.0  # ft
    """Trajectory points above this altitude will be discarded."""

    cumdist_min: ClassVar[float] = 10.0  # NM
    """Flights which traverse less than this cumulative distance will be discarded."""

    def __init__(self, arrivals: Traffic | None = None, departures: Traffic | None = None):
        self.arr = arrivals
        self.dep = departures

        self._cache = {}
        self._cache["features arr"] = {}
        self._cache["features dep"] = {}
        self._cache["features aircraft"] = {}

    def clean(self) -> "AirportTraffic":
        """Cleans the data based on a set on minimum standards. Adds cumdist to the data.

        1. Removes duplicate "timestamp", "icao24".
        2. Calls traffic clean_invalid() and Flight.handle_last_position() for all flights.
        3. NA not allowed in: longitude, latitude, altitude, groundspeed, track, vertical_rate.
        4. Median filter pass which substitutes outliers with backwards and forwards filling (default settings for Traffic).
        5. altitude altitude <= AirportTraffic.altitude_limit.

        Returns:
            The same AirportTraffic instance without invalid data.
        """

        def _clean(traffic: Traffic) -> Traffic | None:
            t = traffic

            # Decide which altitude to use
            altitude = "geoaltitude" if "geoaltitude" in t.data.columns else "altitude"

            # Clean invalid and handle last position
            t = (
                t.drop_duplicates(subset=["timestamp", "icao24"])
                .clean_invalid()
                .pipe(Flight.handle_last_position)
                .eval()
            )
            if t is None:
                return None

            # Drop invalid
            subset = [
                c
                for c in [
                    "longitude",
                    "latitude",
                    altitude,
                    "groundspeed",
                    "track",
                    "vertical_rate",
                ]
                if c in t.data.columns
            ]
            t.data = t.data.dropna(subset=subset).drop(
                columns=[
                    "onground",
                    "alert",
                    "spi",
                    "squawk",
                    "hour",
                    "firstseen",
                    "lastseen",
                    "day",
                ],
                errors="ignore",
            )

            # Median filter pass
            t = t.filter().eval()
            if t is None:
                return None

            # Altitude cutoff
            t = t.query(f"0 <= {altitude} <= {self.altitude_cutoff}")
            if t is None:
                return None

            # Replace altitude with geoaltitude
            if altitude == "geoaltitude":
                t = t.drop(columns="altitude", errors="ignore").rename(columns={"geoaltitude": "altitude"})

            return t

        if self.arr is not None:
            self.arr = _clean(self.arr)

        if self.dep is not None:
            self.dep = _clean(self.dep)

        return self

    def assign_ids(self, id_postfix: str = "") -> "AirportTraffic":
        """Assigns a flight_id to all flights. Arrivals: arr{id_postfix}_0001, departures: dep{id_postfix}_0001.

        Args:
            id_postfix: the postfix to append to "arr" or "dep". Defaults to "".

        Returns:
            The same AirportTraffic instance with new flight_ids.
        """

        def _assign_ids(traffic: Traffic, id_prefix: str) -> Traffic:
            n_digits = len(str(sum(1 for _ in traffic.iterate(by="10 minutes"))))
            return (
                traffic.iterate_lazy(iterate_kw={"by": "10 minutes"})
                .assign_id(name=f"{id_prefix}{{idx:>0{n_digits}}}")
                .eval()
            )

        if self.arr is not None:
            self.arr = _assign_ids(self.arr, f"arr{id_postfix}_")

        if self.dep is not None:
            self.dep = _assign_ids(self.dep, f"dep{id_postfix}_")

        return self

    def cumulative_distance(self) -> "AirportTraffic":
        """Adds the cumdist feature to the data, dropping any overlapping points."""
        if self.arr is not None:
            self.arr = self.arr.cumulative_distance(compute_gs=False, compute_track=False).eval()
            assert isinstance(self.arr, Traffic)  # cumulative_distance() shouldn't return None
            self.arr = self.arr.drop_duplicates(subset=["flight_id", "cumdist"])

        if self.dep is not None:
            self.dep = self.dep.cumulative_distance(compute_gs=False, compute_track=False).eval()
            assert isinstance(self.dep, Traffic)  # cumulative_distance() shouldn't return None
            self.dep = self.dep.drop_duplicates(subset=["flight_id", "cumdist"])

        return self

    def clean_arrivals(self) -> "AirportTraffic":
        """Cleans the arrival data based on a set on minimum standards.

        0. Skip arrival if:
            - destination feature is empty
            - flight does not align with a runway at destination airport
            - maximum cumulative distance is less than `AirportTraffic.cumdist_min`
            - a go around or a runway change on final is detected
            - no ANP aircraft can be associated with the flight
            - the ANP power parameter of the ANP aircraft is not net thrust (CNT) or percentage of maximum thrust.
        1. No points after landing threshold.
        2. altitude, groundspeed and vertical_rate are smoothed with a rolling average over 5s.
        3. altitude is forced to never increase.
        4. Landing threshold is appended as the last point (time, cumdist, altitude, groundspeed and vertical_rate are estimated)
        5. Data is resampled to every second.
        6. Drop duplicate cumdist (after resample)
        7. cumdist is set to be negative (0 at the landing threshold).
        Returns:
            The same AirportTraffic instance with cleaned data.
        """

        def _clean_arrival(arr: Flight) -> Flight | None:
            # TODO: save features (number of points after THR, distance and altitude to THR)

            if self._skip_arrival(arr):
                return None

            f = arr

            # Get last point aligned on runway as Flight and Position
            lp = f.aligned_on_ils(f.destination).final()
            if lp is None:  # may be None despite _skip_arrival due to clean methods
                return None

            lp_pos = lp.at()
            assert isinstance(lp_pos, Position)  # at() never returns None when called without arguments

            # Remove points after threshold
            f = f.before(lp.stop)
            if f is None:
                return None

            # Smooth
            smooth_vars = ["altitude", "groundspeed", "vertical_rate"]
            f.data = f.data.set_index("timestamp")
            f.data[smooth_vars] = f.data[smooth_vars].rolling("5s").mean()
            f = f.reset_index()

            # Force altitude to never increase
            f.data["altitude"] = f.data["altitude"].cummin()

            # -- Add runway threshold at the end
            # Fetch runway information
            rwy_name = lp_pos.ILS
            rwy = runways[arr.destination]  # type: ignore (arr.destination guaranteed by _skip_departure)
            assert isinstance(rwy, RunwayAirport)
            rwy = rwy.data.loc[rwy.data["name"] == rwy_name].squeeze()

            # Speed and vertical rate are the mean of last minute established on ILS
            ils_last_min = f.last("1 min")
            thr_gs = ils_last_min.data["groundspeed"].mean()
            thr_vr = ils_last_min.data["vertical_rate"].mean()

            # Distance and time between last ILS point and runway threshold
            dist_delta = (
                distance(
                    lp_pos.latitude,
                    lp_pos.longitude,
                    rwy.latitude,
                    rwy.longitude,
                )
                / 1852
            )  # m 2 NM
            time_delta = pd.Timedelta(hours=dist_delta / thr_gs)
            thr_cumdist = lp_pos.cumdist + dist_delta

            # Altitude is estimated with linear regression of last minute on ILS
            m_alt, b_alt, _, _, _ = linregress(x=ils_last_min.data["cumdist"], y=ils_last_min.data["altitude"])
            alt_delta = x if (x := (m_alt * thr_cumdist + b_alt) - lp_pos.altitude) < 0 else 0
            alt_cutoff = rwy.elevation if rwy.elevation == rwy.elevation else 0

            # Create threshold point and append to flight
            new_point = pd.Series(
                {
                    "timestamp": (lp_pos.timestamp + time_delta).round("1s"),
                    "cumdist": thr_cumdist,
                    "longitude": rwy["longitude"],
                    "latitude": rwy["latitude"],
                    "altitude": (x if (x := lp_pos.altitude + alt_delta) > alt_cutoff else alt_cutoff),
                    "groundspeed": thr_gs,
                    "vertical_rate": thr_vr,
                }
            )
            f.data = pd.concat([f.data, new_point.to_frame().T], ignore_index=True).infer_objects().ffill()

            # TODO: Add landing roll (from Doc29? Check present in data?)

            # Normalize cumulative distance to the runway threshold
            f.data.cumdist -= f.cumdist_max

            # Resample
            f = f.resample(
                how={
                    "interpolate": [
                        "latitude",
                        "longitude",
                        "groundspeed",
                        "vertical_rate",
                        "altitude",
                        "cumdist",
                    ],
                    "ffill": [
                        "flight_id",
                        "icao24",
                        "callsign",
                        "track",
                        "origin",
                        "destination",
                    ],
                }
            ).drop(columns="track_unwrapped")
            f.data.loc[:, "track"] = (
                f.data.loc[:, "track"].infer_objects().ffill()
            )  # fix for nan tracks after resampling

            # Drop duplicate cumdist (after resampling)
            f = f.drop_duplicates(subset="cumdist")

            return f

        if self.arr is not None:
            self.arr = Traffic.from_flights([_clean_arrival(arr) for arr in self.arr])

        return self

    def clean_departures(self) -> "AirportTraffic":
        """Cleans the departure data based on a set on minimum standards for every flight.

         0. Skip departure if:
            - origin feature is empty
            - flight does not align with a runway at origin airport
            - maximum cumulative distance is less than `AirportTraffic.cumdist_min`
            - no ANP aircraft can be associated with the flight
            - the ANP power parameter of the ANP aircraft is not net thrust (CNT) or percentage of maximum thrust.
        1. Drop duplicate cumdist.
        2. No points before the departure threshold.
        3. altitude, groundspeed and vertical_rate are smoothed with a rolling average over 5s.
        4. altitude is forced to never decrease.
        5. Departure threshold is appended as the first point (time, cumdist, altitude, groundspeed and vertical_rate are estimated)
        6. cumdist is set to be increasing and 0 at the departure threshold.
        7. Data is resampled to every second.
        8. Drop duplicate cumdist (after resample)
        Returns:
            The same AirportTraffic instance with cleaned data.
        """

        def _clean_departure(dep: Flight) -> Flight | None:
            # TODO: save features (number of points before THR, distance and altitude to THR)

            if self._skip_departure(dep):
                return None

            f = dep

            # Drop duplicate cumdist
            f = f.drop_duplicates(subset="cumdist")

            # Get first point aligned on runway as Flight and Position
            fp = dep.takeoff_from_runway(f.origin).next()
            if fp is None:  # may be None despite _skip_departure due to clean methods
                return None

            fp_pos = fp.first("1s").at()
            assert isinstance(fp_pos, Position)  # at() never returns None when called without arguments

            # Remove points before aligned on runway
            f = dep.after(fp_pos.timestamp, strict=False)
            if f is None:
                return None

            # Smooth
            smooth_vars = ["altitude", "groundspeed", "vertical_rate"]
            f.data = f.data.set_index("timestamp")
            f.data[smooth_vars] = f.data[smooth_vars].rolling("5s").mean()
            f = f.reset_index()

            # Force altitude to never decrease
            f.data["altitude"] = f.data["altitude"].cummax()

            # -- Add runway threshold at the start
            # Fetch runway information
            rwy_name = fp_pos.runway
            rwy = runways[f.origin]  # type: ignore (f.origin guaranteed by _skip_departure)
            assert isinstance(rwy, RunwayAirport)
            rwy = rwy.data.loc[rwy.data["name"] == rwy_name].squeeze()

            # Distance from runway threshold
            dist_delta = (
                distance(
                    fp_pos.latitude,
                    fp_pos.longitude,
                    rwy.latitude,
                    rwy.longitude,
                )
                / 1852
            )  # m 2 nm

            # Time from runway threshold assumes constant ground speed
            time_delta = pd.Timedelta(dist_delta / (fp_pos.groundspeed / 2), unit="hours")

            # Create threshold point and append to flight
            new_point = pd.Series(
                {
                    "timestamp": (fp_pos.timestamp - time_delta).round("1s"),  # type: ignore
                    "cumdist": -dist_delta,
                    "longitude": rwy["longitude"],
                    "latitude": rwy["latitude"],
                    "altitude": rwy["elevation"],
                    "groundspeed": 0.0,
                    "vertical_rate": 0.0,
                }
            )
            f.data = pd.concat([new_point.to_frame().T, f.data], ignore_index=True).infer_objects().bfill()

            # TODO: Add takeoff roll (from Doc29? Check present in data?)

            # Normalize cumulative distance to the runway threshold
            f.data["cumdist"] += dist_delta

            # Resample
            f = f.resample(
                how={
                    "interpolate": [
                        "latitude",
                        "longitude",
                        "groundspeed",
                        "vertical_rate",
                        "altitude",
                        "cumdist",
                    ],
                    "ffill": [
                        "flight_id",
                        "icao24",
                        "callsign",
                        "track",
                        "origin",
                        "destination",
                    ],
                }
            ).drop(columns="track_unwrapped")
            f.data.loc[:, "track"] = (
                f.data.loc[:, "track"].infer_objects().ffill()
            )  # fix for nan tracks after resampling

            # Drop duplicate cumdist (after resampling)
            f = f.drop_duplicates(subset="cumdist")

            return f

        if self.dep is not None:
            self.dep = Traffic.from_flights([_clean_departure(dep) for dep in self.dep])

        return self

    def list_all(self) -> pd.DataFrame | None:
        """Get a pd.DataFrame containing information on all flights, including aircraft data.

        Columns: flight_id, operation, time, runway, aircraft_id, icao24, origin, destination, go_around, changed_runway, anp, anp_power_parameter.
        """
        flights = self.list_flights()

        if flights is None:
            return None

        acft = self.list_aircraft()
        assert isinstance(acft, pd.DataFrame)  # if arrivals or departures exist, aircraft must exist

        return flights.merge(
            acft.drop(columns=["icao24", "aircraft_id"]),  # dropping avoids duplicate cols
            on=["flight_id"],
            how="left",
        )

    def list_flights(self) -> pd.DataFrame | None:
        """Get a pd.DataFrame containing information on all flights.

        Columns: flight_id, operation, time, runway, aircraft_id, icao24, origin, destination, cumdist_max, go_around, changed_runway.
        """
        dfs = [df for df in [self.list_arrivals(), self.list_departures()] if df is not None]

        if not dfs:
            return None

        return pd.concat(dfs)

    def list_arrivals(self) -> pd.DataFrame | None:
        """Get a pd.DataFrame containing information on all arrivals.

        Columns: flight_id, operation, time, runway, aircraft_id, icao24, origin, destination, cumdist_max, go_around, changed_runway.
        """

        if self.arr is None:
            return None

        return pd.DataFrame([self._features_arr(f) for f in self.arr])

    def list_departures(self) -> pd.DataFrame | None:
        """Get a pd.DataFrame containing information on all departures.

        Columns: flight_id, operation, time, runway, aircraft_id, icao24, origin, destination, cumdist_max.
        """
        if self.dep is None:
            return None

        return pd.DataFrame([self._features_dep(f) for f in self.dep])

    def list_aircraft(self) -> pd.DataFrame | None:
        """Get a pd.DataFrame containing information on all aircraft.

        Columns: flight_id, icao24, aircraft_id, anp, anp_power_param.
        """
        cumul = []

        if self.arr is not None:
            for arr in self.arr:
                cumul.append(self._features_aircraft(arr))

        if self.dep is not None:
            for dep in self.dep:
                cumul.append(self._features_aircraft(dep))

        return pd.DataFrame(cumul) if len(cumul) != 0 else None

    def _features_arr(self, arr: Flight) -> dict[str, Any]:
        """Get a dictionary listing the features for an arrival.

        Keys: flight_id, operation, time, runway, aircraft_id, icao24, origin, destination, cumdist_max, go_around, changed_runway.
        """
        if arr.flight_id in self._cache["features arr"]:
            return self._cache["features arr"][arr.flight_id]

        acft_id = self._aircraft_id(arr)
        lp = arr.aligned_on_ils(arr.destination).final()
        runway = lp.at().ILS if lp is not None else None  # type: ignore (at() never returns None when called without arguments)
        if "cumdist" not in arr.data.columns:
            arr = arr.cumulative_distance(compute_gs=False, compute_track=False)
        d = {
            "flight_id": arr.flight_id,
            "operation": "Arrival",
            "time": arr.stop,
            "runway": runway,
            "aircraft_id": acft_id,
            "icao24": arr.icao24,
            "origin": arr.origin,
            "destination": arr.destination,
            "cumdist_max": arr.data["cumdist"].abs().max(),
            "go_around": arr.go_around(arr.destination).has(),
            "changed_runway": arr.runway_change(arr.destination).has(),
        }
        self._cache["features arr"][arr.flight_id] = d
        return d

    def _features_dep(self, dep: Flight) -> dict[str, Any]:
        """Get a dictionary listing the features for a departure.

        Keys: flight_id, operation, time, runway, aircraft_id, icao24, origin, destination, cumdist_max.
        """
        if dep.flight_id in self._cache["features dep"]:
            return self._cache["features dep"][dep.flight_id]

        acft_id = self._aircraft_id(dep)
        lp = dep.takeoff_from_runway(dep.origin).next()
        runway = lp.at().runway if lp is not None else None  # type: ignore (at() never returns None when called without arguments)
        if "cumdist" not in dep.data.columns:
            dep = dep.cumulative_distance(compute_gs=False, compute_track=False)
        d = {
            "flight_id": dep.flight_id,
            "operation": "Departure",
            "time": dep.start,
            "runway": runway,
            "aircraft_id": acft_id,
            "icao24": dep.icao24,
            "origin": dep.origin,
            "destination": dep.destination,
            "cumdist_max": dep.data["cumdist"].abs().max(),
        }
        self._cache["features dep"][dep.flight_id] = d
        return d

    def _features_aircraft(self, flight: Flight) -> dict[str, Any]:
        """Get a dictionary listing the aircraft features for a flight.

        Keys: flight_id, icao24, aircraft_id, anp, anp_power_param.
        """
        if flight.flight_id in self._cache["features aircraft"]:
            return self._cache["features aircraft"][flight.flight_id]
        anp = flight.anp
        assert hasattr(anp_data, "aircraft")  # anp aircraft table should exist
        anp_power_param = (
            anp_data.aircraft["Power Parameter"].loc[anp_data.aircraft["ACFT_ID"] == anp].iloc[0]  # type: ignore (anp aircraft table guaranteed to exist)
            if anp is not None
            else None
        )
        d = {
            "flight_id": flight.flight_id,
            "icao24": flight.icao24,
            "aircraft_id": self._aircraft_id(flight),
            "anp": anp,
            "anp_power_param": anp_power_param,
        }
        self._cache["features aircraft"][flight.flight_id] = d
        return d

    def _aircraft_id(self, flight: Flight) -> str | None:
        """Get the aircraft_id for a flight as f"{ICAO Code} {Description}" or None if the aircraft is not in the anp substitution table."""
        if anp_sub := flight.anp_substitution:
            aircraft_icao, aircraft_description = anp_sub
            return f"{aircraft_icao} {aircraft_description}".strip()

        return None

    def compute_thrust(self) -> "AirportTraffic":
        """Add the thrust feature to both arrival and departure flights as per the Doc 29 methodology.
        Additional required features, e.g. TAS, CAS or weather parameters, will also be added.
        """
        self._add_features()
        self._compute_thrust_arrivals()
        self._compute_thrust_departures()

        return self

    def _compute_thrust_arrivals(self) -> None:
        if self.arr is None:
            return

        cumul = []
        for arr in self.arr:
            anp_id = arr.anp
            assert isinstance(anp_id, str)

            arr.data["thrust"] = anp_data.thrust_force_balance(
                anp_id,
                w=None,
                cas=arr.data["CAS"].to_numpy(),
                accel=arr.data["acceleration"].to_numpy(),
                ang=arr.data["climb_angle"].to_numpy(),
                p=arr.data["pressure"].to_numpy(),
                w_percentage=0.9,
            )
            arr.data["flight_phase"] = "Approach"  # Only points before landing THR

            cumul.append(arr)
        self.arr = Traffic.from_flights(cumul)

    def _compute_thrust_departures(self) -> None:
        if self.dep is None:
            return

        cumul = []
        for dep in self.dep:
            anp_id = dep.anp
            assert isinstance(anp_id, str)

            dep.data["thrust"], idx = anp_data.thrust_rating(
                anp_id,
                dep.data["altitude"].to_numpy(),
                dep.data["CAS"].to_numpy(),
                dep.data["TAS"].to_numpy(),
                dep.data["temperature"].to_numpy(),
                dep.data["pressure"].to_numpy(),
                vert_rate=dep.data["vertical_rate"].to_numpy(),
            )
            dep.data["flight_phase"] = None
            dep.data.loc[:idx, "flight_phase"] = "Initial Climb"
            dep.data.loc[idx:, "flight_phase"] = "Climb"

            cumul.append(dep)
        self.dep = Traffic.from_flights(cumul)

    def _add_features(self) -> None:
        self._add_features_arrivals()
        self._add_features_departures()

    def _add_features_arrivals(self) -> None:
        if self.arr is None:
            return

        cumul = []
        for f in self.arr:
            d = self._features_arr(f)
            cumul.append(
                f.compute_weather(
                    src="METAR",
                    metar_station=d["destination"],
                    include_wind=True,
                )
            )

        self.arr = Traffic.from_flights(cumul)
        assert isinstance(self.arr, Traffic)  # compute_weather shouldn't delete flights

        self.arr = (
            self.arr.pipe(Flight.compute_acceleration)
            .pipe(Flight.compute_climb_angle)
            .pipe(Flight.compute_TAS)
            .pipe(Flight.compute_CAS)
            .eval()
        )

    def _add_features_departures(self) -> None:
        if self.dep is None:
            return

        cumul = []
        for f in self.dep:
            d = self._features_dep(f)
            cumul.append(
                f.compute_weather(
                    src="METAR",
                    metar_station=d["origin"],
                    include_wind=True,
                )
            )

        self.dep = Traffic.from_flights(cumul)
        assert isinstance(self.dep, Traffic)  # compute_weather shouldn't delete flights

        self.dep = self.dep.pipe(Flight.compute_TAS).pipe(Flight.compute_CAS).eval()

    def save(self, output_folder: Path) -> None:
        """Save arrivals and departures to folder.

        Args:
            output_folder: path to output folder.
        """
        output_folder.mkdir(exist_ok=True, parents=True)

        if self.arr is not None:
            self.arr.to_csv(output_folder / "arrivals.csv", index=False)

        if self.dep is not None:
            self.dep.to_csv(output_folder / "departures.csv", index=False)

    @classmethod
    def from_folder(cls, folder: Path) -> "AirportTraffic":
        """Load arrivals and departures to folder.

        Args:
            output_folder: path to folder containing arrivals.csv and departures.csv.
        """
        apt_traffic = AirportTraffic()

        apt_traffic.arr = Traffic.from_file(folder / "arrivals.csv")
        if apt_traffic.arr is not None:
            apt_traffic.arr.data["timestamp"] = pd.to_datetime(apt_traffic.arr.data["timestamp"], utc=True)
        apt_traffic.dep = Traffic.from_file(folder / "departures.csv")
        if apt_traffic.dep is not None:
            apt_traffic.dep.data["timestamp"] = pd.to_datetime(apt_traffic.dep.data["timestamp"], utc=True)

        return apt_traffic

    def _skip_arrival(self, arr: Flight):
        f = self._features_arr(arr)
        f_acft = self._features_aircraft(arr)
        return (
            f["runway"] is None
            or f["destination"] is None
            or f["cumdist_max"] < self.cumdist_min
            or f["go_around"]
            or f["changed_runway"]
            or f_acft["anp"] is None
            or f_acft["anp_power_param"] is None
            or not any(s in f_acft["anp_power_param"].lower() for s in ["cnt", "pound"])
        )

    def _skip_departure(self, dep: Flight):
        f = self._features_dep(dep)
        f_acft = self._features_aircraft(dep)
        return (
            f["runway"] is None
            or f["origin"] is None
            or f["cumdist_max"] < self.cumdist_min
            or f_acft["anp"] is None
            or f_acft["anp_power_param"] is None
            or not any(s in f_acft["anp_power_param"].lower() for s in ["cnt", "pound"])
        )
