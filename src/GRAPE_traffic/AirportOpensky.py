from __future__ import annotations

from collections import namedtuple
from dataclasses import astuple, dataclass
from typing import ClassVar

from pitot.geodesy import destination
from traffic.core.flight import Flight
from traffic.core.structure import Airport
from traffic.core.time import timelike, to_datetime
from traffic.core.traffic import Traffic
from traffic.data import airports, opensky


@dataclass
class BoundingBoxParameters:
    Width: float  # NM
    Height: float  # NM
    Rotation: float = 0.0  # Deg

    def __iter__(self):
        yield from astuple(self)


class AirportOpensky:
    """AirportOpensky uses the traffic library to access Opensky Network data for a given airport and time frame, splitting it by arrivals and departures.
    After instantiating the class call:
        - AirportOpensky.fetch_all to get all flights
        - AirportOpensky.fetch_arrivals to get arrivals only
        - AirportOpensky.fetch_departures to get departures only

    Args:
        airport: the airport for which to fetch data.
        start_time: the start time
        stop_time (timelike): _description_
    """

    bounding_box: ClassVar[BoundingBoxParameters] = BoundingBoxParameters(40.0, 40.0)
    """The width, height and rotation of the bounding box centered on the airport coordinates for which data will be fetched."""

    def __init__(
        self,
        airport: "str | Airport",
        start_time: timelike,
        stop_time: timelike,
    ):
        if isinstance(airport, str):
            apt = airports[airport]
            if apt is None:
                raise RuntimeError(f"Airport '{airport}' not found.")
            self.airport = apt
        else:
            self.airport = airport

        # Bounding box, defines self.bb_[ll, ul, ur, lr] as GeoPoint instances
        self.__set_bounding_box()

        # Save start and stop times
        self.start_time = to_datetime(start_time)
        self.stop_time = to_datetime(stop_time)

    def fetch_all(self) -> Traffic | None:
        return self.__fetch(True, True)

    def fetch_arrivals(self) -> Traffic | None:
        return self.__fetch(True, False)

    def fetch_departures(self) -> Traffic | None:
        return self.__fetch(False, True)

    def __fetch(self, arrivals: bool, departures: bool) -> Traffic | None:
        west, south = self.bb_ll
        east, north = self.bb_ur
        bounds = (west, south, east, north)
        r = None

        if arrivals and departures:
            r = opensky.history(
                airport=self.airport.icao,
                start=self.start_time,
                stop=self.stop_time,
                bounds=bounds,
            )
        elif arrivals:
            r = opensky.history(
                arrival_airport=self.airport.icao,
                start=self.start_time,
                stop=self.stop_time,
                bounds=bounds,
            )
        elif departures:
            r = opensky.history(
                departure_airport=self.airport.icao,
                start=self.start_time,
                stop=self.stop_time,
                bounds=bounds,
            )

        if isinstance(r, Flight):
            r = Traffic.from_flights([r])

        return r.reset_index(drop=True) if r is not None else None

    def __set_bounding_box(self):
        # Sets the bounding box
        # ul---------ur
        # |           |
        # |    apt    |
        # |           |
        # ll----h----lr
        GeoPoint = namedtuple("GeoPoint", "lon lat")

        width, height, rotation = self.bounding_box
        width *= 1852.0  # nautical miles to meters
        height *= 1852.0  # nautical miles to meters

        lat, lon, _ = destination(
            self.airport.latitude,
            self.airport.longitude,
            rotation + 180.0,
            height / 2.0,
        )
        bb_h = GeoPoint(lon, lat)

        lat, lon, _ = destination(bb_h.lat, bb_h.lon, rotation + 270.0, width / 2.0)
        self.bb_ll = GeoPoint(lon, lat)

        lat, lon, _ = destination(self.bb_ll.lat, self.bb_ll.lon, rotation, height)
        self.bb_ul = GeoPoint(lon, lat)

        lat, lon, _ = destination(self.bb_ul.lat, self.bb_ul.lon, rotation + 90.0, width)
        self.bb_ur = GeoPoint(lon, lat)

        lat, lon, _ = destination(self.bb_ur.lat, self.bb_ur.lon, rotation + 180.0, height)
        self.bb_lr = GeoPoint(lon, lat)
