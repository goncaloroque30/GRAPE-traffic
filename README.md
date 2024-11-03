# GRAPE-traffic ![License](https://img.shields.io/github/license/goncaloroque30/grape-traffic) ![os](https://img.shields.io/badge/os-Windows-blue)

The *GRAPE-traffic* library connects the functionality provided by [traffic](https://github.com/goncaloroque30/traffic/tree/env_impacts) (a fork from the [original traffic](https://github.com/xoolive/traffic) library) with the environmental impact calculation engine [GRAPE](https://goncaloroque30.github.io/GRAPE-Docs/) to estimate noise and local air quality emissions for trajectory data, e.g. as provided by the [Opensky Network](https://opensky-network.org/).

## Installation

*GRAPE-traffic* uses [poetry](https://python-poetry.org/) for dependency management. Start by cloning the repository to your computer and then run `poetry install` at the root of the repository. To run the code in the [notebooks](#notebooks), use the [development installation](#development).

You will need GRAPE to estimate environmental impacts. Head over to the [docs](https://goncaloroque30.github.io/GRAPE-Docs/) or the [Github repo](https://github.com/goncaloroque30/GRAPE) and download the latest version (follow the installation instructions).

![Static Badge](https://img.shields.io/badge/Warning-FFFF00)

GRAPE currently only supports Windows.

## Getting Started

To get acquainted with the functionality provided by this library, head over to the `example` notebook in the *notebooks* directory. The code is built on top of a [fork of the traffic](https://github.com/goncaloroque30/traffic/tree/env_impacts) library. The implemented functionality is therefore split between that fork and this library. The features implemented in the fork and not mentioned in the `example` notebook are:

- **[ANP database](https://www.easa.europa.eu/en/domains/environment/policy-support-and-research/aircraft-noise-and-performance-anp-data) and Doc 29 thrust calculation support**: *traffic* provides functionality to access multiple types of data. The fork adds the ANP database to the list, and implements the Doc 29 thrust calculation methodology on top of it.
- **[IEM weather data](https://mesonet.agron.iastate.edu/request/download.phtml)**: although the original *traffic* library already provides functionality to work with **METAR** data, the fork adds a number of features such as caching. It also adds the `compute_weather()` method to the `Traffic` and `Flight` classes, which computes weather at altitude with the [ISA atmospheric model](https://en.wikipedia.org/wiki/International_Standard_Atmosphere), and can use either the ISA standard values or the ones obtained from METAR as the base values.
- **runway threshold elevation**: as runway threshold elevation is required for ensuring commonality between all trajectories, it is added to the list of retrievable features in the classes `RunwayAirport` and `Runways` in the *traffic* library.
- **additional features for `Flight`**: *traffic* already provides functionality to enhance trajectory data with computed features, such as TAS. The fork complements this with the `compute_acceleration`, `compute_climb_angle` and `compute_CAS` methods.

## Configuration

This library follows the same convention of the *traffic* library for configuration (i.e. configuration files stored in your computer). To check where each configuration file is located, run:

```python
import traffic

import GRAPE_traffic

print(f"Traffic .conf: {traffic.config_file}")
print(f"GRAPE Traffic .conf: {GRAPE_traffic.config_file}")
```

The most relevant configurations for each library used by *GRAPE-traffic* are listed below. For other configurations, see the comments in each configuration file.

### traffic

- `opensky`: input here your access credentials to the Opensky Network. Use the `test opensky` notebook (see [notebooks](#notebooks) section) to check that everything is working properly.

- `anp`: override access to the online available [ANP databases](https://www.easa.europa.eu/en/domains/environment/policy-support-and-research/aircraft-noise-and-performance-anp-data) with one locally stored.

### GRAPE-traffic

- `grape_exe`: the path to the GRAPE executable on your computer (see the [installation](#installation) section).

- `foi_path`: path to the [FOI database](https://www.foi.se/en/foi/research/aeronautics-and-space-issues/environmental-impact-of-aircraft.html) for turboprop engine emissions.

- `lto_path`: path to an aircraft ID to LTO (EEDB/FOI) engine association overriding the default one (`src/GRAPE_traffic/LTO.csv`).

- `eedb_path`: override access to the online available [EEDB database](https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank) with one locally stored.

## Development

After cloning the repository to your computer, open a terminal at the root of the repository and run:

```sh
poetry install --with dev
poetry run pre-commit install
```

## Notebooks

- `example`: read through this notebook to get acquainted with the functionality of this library and the different steps involved in estimating environmental impacts from trajectory data.
- `test opensky`: a simple notebook to test if your connection to the [Opensky Network](https://opensky-network.org/) is working properly.
- `eddk *`: notebooks used in an use case of this library for the EDDK airport. This use case uses airport data not provided in this repository.
