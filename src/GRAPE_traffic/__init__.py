from configparser import ConfigParser
from pathlib import Path

from appdirs import user_cache_dir, user_config_dir

from .AirportOpensky import AirportOpensky, BoundingBoxParameters
from .AirportTraffic import AirportTraffic
from .GrapeTraffic import GrapeTraffic

# -- Configuration management --

config_dir = Path(user_config_dir("GRAPE-traffic"))
config_file = config_dir / "GRAPE-traffic.conf"

if not config_dir.exists():
    config_template = (Path(__file__).parent / "grape_traffic.conf").read_text()
    config_dir.mkdir(parents=True)
    config_file.write_text(config_template)

config = ConfigParser()
config.read(config_file.as_posix())

cache_dir = Path(user_cache_dir("GRAPE-traffic"))
cache_dir.mkdir(parents=True, exist_ok=True)

# -- State Initialization --
AirportOpensky.bounding_box = BoundingBoxParameters(
    float(config.get("bounding box", "width", fallback=40.0)),
    float(config.get("bounding box", "height", fallback=40.0)),
    float(config.get("bounding box", "rotation", fallback=0.0)),
)

AirportTraffic.altitude_cutoff = float(config.get("trajectory point filters", "altitude", fallback=10000.0))
AirportTraffic.cumdist_min = float(config.get("flight filters", "cumulative_distance_min", fallback=10.0))

GrapeTraffic.cache_dir = cache_dir
GrapeTraffic.foi_path = Path(config.get("grape traffic", "foi_path"))
GrapeTraffic.grape_exe = Path(config.get("grape traffic", "grape_exe"))
if config.has_option("grape traffic", "lto_path"):
    GrapeTraffic.lto_path = Path(config.get("grape traffic", "lto_path"))
if config.has_option("grape traffic", "eedb_path"):
    GrapeTraffic.eedb_path = Path(config.get("grape traffic", "eedb_path"))
GrapeTraffic.lto_emissions_cutoff = float(config.get("grape traffic", "lto_emissions_cutoff", fallback=3000.0))
