## depot: a Python library of map-making utilities for Subway Builder

## Requirements
This library requires a shell.  Windows users must use WSL.  No support is provided 
for a non-WSL approach, but savvy users may be able to figure one out.

This library is confirmed to work with the following Python package versions:
| Package            | Version       |
| ------------------ |:-------------:|
| python             | 3.13.9        |
| duckdb             | 1.5.1         |
| geopandas          | 1.1.1         |
| mapbox_vector_tile | 2.2.0         |
| numpy              | 2.3.5         |
| pandas             | 2.3.3         |
| shapely            | 2.1.2         |
| duckdb             | 1.5.1         |
| tqdm               | 4.68.2        |
| xarray             | 2026.4.0      |
| httpx              | 0.28.1        |
| mercantile         | 1.2.1         |
| netcdf4            | 1.7.4         |
| osmnx              | 2.1.0         |

Users can prepare a [conda](https://docs.conda.io/projects/conda/en/stable/index.html) 
environment with these package versions using the supplied environment file:

    conda env create -f environment.yml

For non-conda environment managers, ensure you have the appropriate versions listed 
above along with any dependencies.  Users that do not already have a preferred 
Python environment manager are recommended to use conda due to the provided 
environment.yml file.  To install conda, at the link above, download the Miniforge 
installer for your OS. Run it and follow the instructions.  Then use the above command 
to replicate the environment that is confirmed to work for this library.

In addition to the Python environment, all of the following CLI programs must be 
available within the path to create the non-demand files needed for custom maps:
* node
* [mapshaper](https://github.com/mbloch/mapshaper)
* [osmium](https://osmcode.org/osmium-tool/)
* java
* [tippecanoe and tile-join](https://github.com/felt/tippecanoe)
* sqlite3
* jq
* [pmtiles](https://github.com/protomaps/go-pmtiles/releases)
* [planetiler.jar](https://github.com/onthegomap/planetiler/releases)

Except for planetiler.jar, these must be executable so that `depot` can 
determine that they are available (e.g., if `pmtiles` is not executable, 
it will be marked as missing).  The code will not run if any of these 
requirements are missing.

## Installation
In the repo directory, run

    pip install .

The library is now installed and can be imported.  See `examples/` for scripts 
that use the library to build a map.

## Usage

At present, `depot` includes the ability to create the non-demand files for 
custom maps.  This is handled through the `MapGen` class.

### `MapGen` inputs
| Parameter          | Description       |
| ------------------ |:-------------:|
| `city`             | str. 2-4 character city code.        |
| `bbox`             | list of floats. Bounding box for the map. [min_lon, min_lat, max_lon, max_lat]         |
| `osmpbf`           | str or list of str. Path(s) to local .osm.pbf file(s) to use as a source. Obtain them from <https://download.geofabrik.de/>        |
| `outputdir`        | str. Path to output directory. Within the specified directory, a new directory named `city` will be created to hold all outputs and intermediate files. Defaults to the current directory. Default: current working directory         |
| `building_index_filter_size`     | int. Filters buildings below this size (in m^2) for collisions. Default: 40         |
| `building_tile_filter_size`      | int. Filters buildings below this size (in m^2) for pmtiles. Must be <= `building_index_filter_size`.  If None, it is set to `building_index_filter_size`. Default: None         |
| `building_index_simplification`  | int or float. Minimum distance in meters between building nodes.  Higher values reduce buildings_index.json file size at the cost of reduced accuracy.  Be careful to not use too large of a value. Default: 1 |
| `building_tile_simplification`   | int or float. Like `building_index_simplification`, but for the buildings in the pmtiles file. |
| `max_building_tile_size`         | int. Maximum size per tile in KB when considering only buildings. The absolute maximum per tile is 500, which includes buildings, rivers, roads, and more. Default: 450  |
| `cities`           | list of str. OSM 'place' values to show at the lowest zooms. If None, labels will not be created for that zoom. Default: None         |
| `suburbs`          | list of str. Like cities, but for medium zooms. Default: None         |
| `neighborhoods`    | list of str. Like cities, but for the highest zooms. Default: None         |
| `cities_additional`        | str. Path/to/geojson file that contains label features to be shown at the lowest zoom levels. If None, it is not used. Default: None |
| `suburbs_additional`       | str. Like `cities_additional`, but for medium zooms. |
| `neighborhoods_additional` | str. `Like cities_additional`, but for the highest zooms. |
| `places_suffix`    | str. Suffix to add after the `place` tag when pulling labels from OSM. For example, if using Chinese labels, set this to "CN" to pull from `place:CN`. Default: "" |
| `label_name_language`   | str or None. Controls which name field is used for label text. Use `prefer:<lang>` to try `name:<lang>` first and fall back to `name`, or `force:<lang>` to use only `name:<lang>`. Default: None |
| `road_name_preferred_language`             | str or None. Preferred OSM language code suffix for road names in `roads.geojson`. For example, `en` prefers `name:en` and falls back to `name`. Default: None |
| `buildings_geojson`     | str or None. If a string, path to a buildings.geojson file to use as input. If None, fetches Overture buildings. Default: None |
| `redownload_buildings`  | bool. Determines whether to re-fetch buildings (True) or load previously-saved buildings if available (False). Default: False        |
| `create_building_foundations`  | bool. Determines whether to calculate buildings foundations layer.  Increases PMTiles size, but enables the map layer ingame.  If False, a default foundation of 10 m is used and the ingame map layer is disabled. Default: True        |
| `create_ocean_foundations`     | bool. Determines whether to calculate ocean foundations layer.  When enabled, tracks cannot be built within water, only above the water or below the sea/lake/river bed.  If False, players can build anywhere in the water without restriction.  When enabled, only slight increase to PMTiles size. Default: True        |
| `reprocess_bathymetry_data`    | bool. Determines whether to read previously calculated bathymetric data if available or recalculate it fresh. Default: True        |
| `color_military_like_aerodrome`  | bool. If True, military bases are colored on the map the same as airports. If False, it looks like any other ordinary tile. Default: True  |
| `maxzoom`          | int. Maximum zoom level for maps. Default: 15         |
| `ncores`           | int. Number of cores to use when processing tiles in parallel. Setting this to None will use all available cores. Default: 1         |
| `cleanup_files`    | bool. If True, deletes some intermediate files that are created and used within the same function. Default: True         |
| `RAM`              | int or float. Sets the amount of RAM in GB to use when calling mapshaper.  If you get heap allocation errors, increase this value.  Keep in mind your OS and other programs still need to run, so don't try to allocate your system's full RAM amount. Default: 4         |
| `verb`             | bool. Determines whether to print additional info or not. Default: True        |

### `MapGen` methods that you care about
| Class method                 | Description                                                    |
| ---------------------------- |:--------------------------------------------------------------:|
| `extract_base_data`          | osmium extract for base layers                                 |
| `process_buildings`          | Fetch Overture buildings and create buildings_index.json       |
| `process_roads_and_aeroways` | Creates roads.geojson and runways_taxiways.geojson             |
| `generate_pmtiles`           | Creates the PMTiles file with no labels                        |
| `add_labels`                 | Adds labels to the PMTiles file created by `generate_pmtiles`  |
| `run_all`                    | Runs the above 5 methods consecutively                         |
| `check_labels`               | Reports the types and number of places (available map labels)  |

These methods take no inputs; they use the user-provided inputs when initializing the object. 
`extract_base_data` must be executed first; 
`process_buildings` must be executed before `generate_pmtiles`; 
and `generate_pmtiles` must be executed before `add_labels`.

Users may want to re-run `process_buildings` and `generate_pmtiles` multiple times to tweak the filtering and 
simplification parameters.  Run `check_labels` to see what types of labels are available for your map and their 
frequency to inform what you categorize for `add_labels`.  Users may want to re-run `add_labels` multiple times 
to decide which place tags should be in which categories.

### Labels
Users may want to look at OSM's list of available 'place' keys: <https://wiki.openstreetmap.org/wiki/Key:place>
The `check_labels` method reports what types of places are used in the map, and how often.

For reference, the setups slurry uses for the maps they have made are provided below:

    US maps:
        cities = ['city', 'borough', 'town']
        suburbs = ['suburb', 'village']
        neighborhoods = ['neighbourhood', 'hamlet', 'quarter', 'locality']
    PR maps:
        cities = ['city', 'borough', 'town']
        suburbs = ['suburb']
        neighborhoods = ['village', 'quarter']
    MX maps:
        cities = ['city', 'borough']
        suburbs = ['city', 'borough', 'town', 'suburb']
        neighborhoods = ['city', 'borough', 'town', 'suburb', 'village', 'hamlet']

Experiment and see what provides the right amount of labeling.

## Future plans
- A module to create and manipulate demand data

## Contributions
This tool is designed to serve the needs of the map-making community. 
Feel free to submit Pull Requests with new features, optimizations, etc.

## Questions?  Issues?  Requests?

Open an issue here on GitHub or provide feedback in Discord.

## License

This library is made available under the GPL v3 license.  See LICENSE for details.
