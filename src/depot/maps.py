import sys, os
import subprocess
import shutil
import requests
import httpx
import json
import struct
import gzip
import math
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import sqlite3
import zlib
import mapbox_vector_tile
import duckdb
import geopandas as gpd
import pandas as pd
import shapely
from shapely import wkb, set_precision
from shapely.ops import unary_union, orient
from shapely.geometry import shape, mapping, box, Polygon, MultiPolygon, Point, LineString
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import mercantile
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import depot.utils as U


class MapGen:
    """
    Class to make the building_index.json, roads.geojson, 
    runways_and_taxiways.geojson, and CITY.pmtiles files needed to make maps 
    for Subway Builder.
    
    Methods
    -------
    run_all : Runs all steps to make map files.
    _run_command : Helper to run shell commands safely.
    _merge_osmpbf_files : Merges multiple input .osm.pbf files into one.
    extract_base_data : osmium extract for base layers.
    _convert_to_game_format : Converts GeoJSON buildings into a spatial grid-
                              indexed JSON for the game engine.
    create_buildings_index_binary : Converts GeoJSON buildings into packed 
                                    binary index streams (.bin and .bin.gz).
    _fetch_overture_buildings : Queries Overture Maps S3 bucket using DuckDB 
                                and saves to GeoJSON.
    _get_latest_overture_release : Queries the latest version of Overture maps.
    get_utm_epsg : Calculates the UTM EPSG code using the instance's bbox 
                   attribute.  Automatically called when bbox is set.
    process_buildings : Overture fetch -> Mapshaper cleanup -> Game conversion.
    process_roads_and_aeroways : Extracts roads and aeroways, applies JQ 
                                 filters and buffering.
    generate_pmtiles : Full Planetiler -> Tile-join -> Tippecanoe -> PMTiles 
                       flow.
    _apply_jq : Internal helper for JQ operations.
    _buffer_linestrings : Internal helper to convert LineStrings to Polygons 
                          (buffer fix).
    _calculate_buffer : Internal helper to calculate a buffer size for 
                        LineStrings based on the zoom level.
    load_bathymetry_data : Connects to GEBCO's bathymetry data via OPeNDAP 
                           and processes it into SB's ocean_depths_index format.
    _process_columns_worker : Worker function for ocean depth index creation.
    _generate_ocean_depth_tiles : Creates ocean_foundations mbtiles from the 
                                  ocean_depth_index.json.
    _get_kind_and_rank : Helper to map OSM/Planetiler tags to game-engine 
                         specific kinds and ranks.
    _process_tile_worker : Worker function to handle vector tile re-mapping.
    fix_mbtiles : Translates 'clean' mbtiles to 'fixed' mbtiles with proper 
                  schema and hierarchy.
    _generate_building_tiles : Processes building GeoJSON into zoom-specific 
                               MBTiles using mapshaper and tippecanoe.
    _set_default_building_height : Sets default building height for buildings 
                                   geojson file.
    _create_building_foundation_files : Calculates the building foundation depths 
                                        and stores as mbtiles.
    _calculate_building_foundation : Calculates the foundation depth for a single 
                                     building.
    _update_mbtiles_metadata : Sqlite3 metadata update.
    _validate_env : Checks if all required CLI tools are installed and 
                    accessible.
    rename_geojson_property : Renames a GeoJSON property key using jq.
    check_labels : Checks city.osm.pbf and reports the types and counts of places.
    add_labels : Extraction and tiling for labels. 
    _combine_geojson_labels : Merge user labels into OSM labels.
    _validate_places : Ensures cities/suburbs/neighborhoods are valid entries.
    _validate_additional_places : Ensures additional cities/suburbs/
                                  neighborhoods are valid files.
    _rewrite_label_geojson_names : Normalizes feature properties.name based on 
                                   label_name_language.
    _select_label_name : Returns the label text to store in properties.name.
    _get_road_name_jq_expression : Returns the jq expression used to populate 
                                   roads.geojson name.
    """
    def __init__(self, city, bbox, osmpbf=None, outputdir='.', 
                       building_index_filter_size=40, 
                       building_tile_filter_size=None, 
                       building_index_simplification=1,
                       building_tile_simplification=1,
                       max_building_tile_size=None,
                       cities=None, suburbs=None, neighborhoods=None,
                       cities_additional=None, suburbs_additional=None, 
                       neighborhoods_additional=None, 
                       places_suffix="", label_name_language=None,
                       road_name_preferred_language=None,
                       buildings_geojson=None, redownload_buildings=False, 
                       create_building_foundations=True, create_ocean_foundations=True,
                       reprocess_bathymetry_data=False, 
                       color_military_like_aerodrome=True,
                       maxzoom=15, 
                       ncores=1, RAM=4, cleanup_files=True, verb=True, debug=False):
        """
        Inputs
        ------
        city: str. 2-4 character city code.
        bbox: list of floats. Bounding box for the map.
                            [min_lon, min_lat, max_lon, max_lat]
        osmpbf: str, or list of str. Path to local .osm.pbf file to use as a 
                     source. If the map's area spans multiple .osm.pbf files, 
                     pass them as a list of paths to each file.
                     If None, will fetch the data online (NOT YET IMPLEMENTED,
                     YOU MUST PROVIDE AT LEAST ONE LOCAL .OSM.PBF FILE).
                     Default: None
        outputdir: str. Path to output directory. Within the 
                        specified directory, a new directory named 
                        `city` will be created to hold all outputs 
                        and intermediate files.
                        Defaults to the current directory.
                        Default: current working directory
        building_index_filter_size: int. Filters buildings below this size (in m^2) 
                                   for collisions and for pmtiles.
                                   Default: 40
        building_tile_filter_size: int.  Filters buildings below this size (in m^2) for pmtiles
                         at the highest zooms.  Must be >= building_index_filter_size.
                         If None, uses `building_index_filter_size`.
                         Default: None
        building_index_simplification: int or float. Minimum distance in 
                                meters between building nodes.  Higher values 
                                reduce buildings_index.json file size at the 
                                cost of reduced accuracy.  Be careful to not 
                                use too large of a value.
                                Default: 1
        building_tile_simplification: int or float. Like 
                                `building_index_simplification`, but for the 
                                buildings in the pmtiles file.
        max_building_tile_size: int or None. Maximum size per tile in KB when 
                                considering only buildings. 
                                Normally pmtiles are capped at 500 kb per tile 
                                for performance reasons; users may wish to 
                                adopt the same for their maps.
                                If None, no limit is enforced. 
                                Default: None
        cities: list of str. OSM 'place' values to show at the lowest zooms.
                             If None, labels will not be created for that zoom.
        suburbs: list of str. Like cities, but for medium zooms.
        neighborhoods: list of str. Like cities, but for the highest zooms.
        cities_additional: str. Path/to/geojson file that contains label features 
                             to be shown at the lowest zoom levels.
                             If None, it is not used.
                             Default: None
        suburbs_additional: str. Like cities_additional, but for medium zooms.
        neighborhoods_additional: str. Like cities_additional, but for the highest zooms.
        places_suffix: str. Suffix to add after the `place` tag when pulling 
                            labels from OSM. Must be a two-letter ISO code.  
                            For example, if using Chinese labels, set this to 
                            "CN" to pull from `place:CN`.
        label_name_language: str or None. Controls which OSM name field is 
                            used for label text. Use "prefer:<lang>" to try
                            `name:<lang>` first and fall back to `name`, or
                            "force:<lang>" to use only `name:<lang>`.
                            If None, uses `name`.
        road_name_preferred_language: str or None. Preferred OSM language
                            code to use for road names in roads.geojson.
                            If set to `en`, roads will use `name:en` when
                            available and fall back to `name`. If None, uses
                            `name`.
        buildings_geojson: str. Path to buildings.geojson file to use.
                                If provided, Overture buildings will not be 
                                downloaded.
                                If None, Overture buildings will be downloaded.
                                Default: None
        redownload_buildings: bool. Determines whether to re-fetch 
                                    buildings (True) or load previously-saved
                                    buildings if available (False).
                                    Default: False
        create_building_foundations: bool. Determines whether to calculate 
                                     buildings foundations layer.  Increases 
                                     PMTiles size, but enables the map layer 
                                     ingame.  If False, a default foundation 
                                     of 10 m is used and the ingame map layer 
                                     is disabled.
                                     Default: True
        create_ocean_foundations: bool. Determines whether to calculate ocean 
                                  foundations layer.  When enabled, tracks 
                                  cannot be built within water, only above the 
                                  water or below the sea/lake/river bed.  If 
                                  False, players can build anywhere in the 
                                  water without restriction.  When enabled, 
                                  only slight increase to PMTiles size.
                                  Default: True
        reprocess_bathymetry_data: bool. Determines whether to read previously 
                                   calculated bathymetric data if available 
                                   or recalculate it fresh.
                                   Default: True
        color_military_like_aerodrome: bool. If True, military bases are 
                                       colored on the map the same as airports.
                                       If False, it looks like any other 
                                       ordinary tile.
                                       Default: True
        maxzoom: int. Maximum zoom level for maps. Default: 15
        ncores: int. Number of cores to use when processing tiles in parallel.
                     Setting this to None will use all available cores.
                     Default: 1
        RAM: int or float. Sets the amount of RAM in GB to use when calling 
                           mapshaper.  If you get heap allocation errors, 
                           increase this value.  Keep in mind your OS and other
                           programs still need to run, so don't try to allocate
                           your system's full RAM amount.
                           Default: 4
        cleanup_files: bool. If True, deletes some intermediate files that are
                             created and used within the same function.
                             Default: True
        verb: bool. Determines whether to print additional info or not.
                    Default: True
        debug: bool. Determines whether to output some additional details to 
                     the pmtiles file that are not needed but can be helpful.
                     Default: False
        """
        self.verb = bool(verb)
        self.debug = bool(debug)
        # Ensure the environment is set up correctly
        self._validate_env()
        
        # Load user params
        self.city = city
        self.bbox = bbox
        self.outputdir = outputdir
        if osmpbf is None:
            raise ValueError("Received osmpbf=None. In the future, this will "
                        "fetch from Overpass, but it is not yet implemented. "
                        "Specify a local .osm.pbf file.")
        self.osmpbf_sources = osmpbf
        self.osmpbf = osmpbf
        if isinstance(osmpbf, list):
            if len(osmpbf) > 1:
                # Multiple .osm.pbf files that need to be merged
                self._merge_osmpbf_files()
            else:
                self.osmpbf = osmpbf[0]
        self.buildings_geojson = buildings_geojson
        self.REFETCH_BUILDINGS = bool(redownload_buildings)
        self.create_building_foundations = bool(create_building_foundations)
        self.create_ocean_foundations = bool(create_ocean_foundations)
        self.reprocess_bathymetry_data = bool(reprocess_bathymetry_data)
        self.color_military_like_aerodrome = bool(color_military_like_aerodrome)
        self.maxzoom = int(maxzoom)
        self.ncores = ncores
        self.RAM = RAM # Multiplied by 1000 in the setter to convert GB -> MB
        self.cleanup_files = bool(cleanup_files)
        
        # Set building area limits
        self.building_index_filter_size = building_index_filter_size
        self.building_tile_filter_size = building_tile_filter_size \
                                    if building_tile_filter_size is not None \
                                    else self.building_index_filter_size
        if self.building_tile_filter_size > self.building_index_filter_size:
            raise ValueError(f"building_tile_filter_size "
                             f"({self.building_tile_filter_size}) cannot be "
                             f"larger than building_index_filter_size "
                             f"({self.building_index_filter_size})")
        
        # Building simplifications
        self.building_index_simplification = building_index_simplification
        self.building_tile_simplification  = building_tile_simplification

        # Maximum size per tile for buildings
        self.max_building_tile_size = max_building_tile_size
        if self.max_building_tile_size is not None:
            if self.max_building_tile_size < 100:
                raise ValueError("`max_building_tile_size` should be >=100"
                                f"\nReceived: {self.max_building_tile_size}")
            self.max_building_tile_size = int(max_building_tile_size) * 1000
        
        # Labels
        self.cities = cities
        self.suburbs = suburbs
        self.neighborhoods = neighborhoods
        
        self.cities_additional = cities_additional
        self.suburbs_additional = suburbs_additional
        self.neighborhoods_additional = neighborhoods_additional

        self.label_name_language = label_name_language
        self.road_name_preferred_language = road_name_preferred_language

        if len(places_suffix)==3 and places_suffix[0]==':':
            self.places_suffix = places_suffix
        elif len(places_suffix)==2 and ':' not in places_suffix:
            self.places_suffix = ':' + places_suffix
        else:
            self.places_suffix = ""
        # Initialize bathymetry variable
        self.bathy_data = None
        
        if self.verb:
            print("***** MapGen initialized *****")
            print("------------------------------")
            print(f"city                : {self.city}")
            print(f"bbox                : {self.bbox}")
            print(f"redownload_buildings: {self.REFETCH_BUILDINGS}")
            print(f"osmpbf source files : {self.osmpbf_sources}")
            print(f"create_building_foundations  : {self.create_building_foundations}")
            print(f"create_ocean_foundations     : {self.create_ocean_foundations}")
            print(f"reprocess_bathymetry_data    : {self.reprocess_bathymetry_data}")
            print(f"color_military_like_aerodrome: {self.color_military_like_aerodrome}")
            print(f"building_index_filter_size   : {self.building_index_filter_size} m2")
            print(f"building_tile_filter_size    : {self.building_tile_filter_size} m2")
            print(f"max_building_tile_size       : {self.max_building_tile_size}")
            print(f"maxzoom      : {self.maxzoom}")
            print(f"ncores       : {self.ncores}")
            print(f"RAM          : {self.RAM} MB")
            print(f"cleanup_files: {self.cleanup_files}")
            print(f"Files will be saved in {self.city_dir}")
        
    def run_all(self):
        """
        Runs all steps to make map files.
        """
        self.extract_base_data()
        self.process_buildings()
        self.process_roads_and_aeroways()
        self.generate_pmtiles()
        self.add_labels()
    
    def _run_command(self, cmd, cwd=None):
        """
        Helper to run shell commands safely.
        """
        try:
            result = subprocess.run(cmd, check=True, shell=isinstance(cmd, str), 
                           cwd=cwd)#, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {cmd}\nError: {e.stderr}")

    def _merge_osmpbf_files(self):
        """Merges multiple input .osm.pbf files into one."""
        merged_osmpbf = os.path.join(self.city_dir, f"{self.city.lower()}-merged-source.osm.pbf")
        osmium_cmd = ["osmium", "merge"]
        osmium_cmd.extend(self.osmpbf)
        osmium_cmd.extend(["-o", merged_osmpbf, "--overwrite"])
        self._run_command(osmium_cmd)
        self.osmpbf = merged_osmpbf
    
    def extract_base_data(self):
        """
        osmium extract for base layers.
        """
        if self.verb:
            print(f"***** Extracting base data for {self.city} *****")
        out_pbf = os.path.join(self.city_dir, f"{self.city.lower()}.osm.pbf")
        bbox_str = ",".join(map(str, self.bbox))
        
        cmd = [
            "osmium", "extract", "--strategy", "smart",
            "-S", 
            "tags=natural=water,landuse=water,landuse=reservoir,"
            "waterway=riverbank,waterway=dock,highway=residential",
            "--bbox", bbox_str, self.osmpbf, "-o", 
            out_pbf, "--overwrite"
        ]
        self._run_command(cmd)
        
        # Filter out buildings
        nobuilding_pbf = os.path.join(self.city_dir, f"{self.city.lower()}-nobuildings.osm.pbf")
        self._run_command([
            "osmium", "tags-filter", out_pbf, 
            "n/building=yes", "w/building=yes", "-i",
            "-o", nobuilding_pbf, "--overwrite"
        ])
        
        self.nobuildings_geojson = nobuilding_pbf.replace('.osm.pbf', '.geojson')
        self._run_command([
            "ogr2ogr", "-f", "GeoJSONSeq", 
            self.nobuildings_geojson, nobuilding_pbf
        ])
    
    def _convert_to_game_format(self, input_path, default_height=4.0):
        """
        Converts GeoJSON buildings into a spatial grid-indexed JSON for the 
        game engine.
        """
        output_path = input_path.replace('cleaned', 'index')
        CS = 0.0009  # Cell size constant

        def calculate_polygon_centroid(coords):
            area = 0.0
            cx, cy = 0.0, 0.0
            if not coords or not coords[0]: return [0, 0]
            ring = coords[0]
            n = len(ring) - 1
            for i in range(n):
                x0, y0 = ring[i]
                x1, y1 = ring[i+1]
                cross_product = (x0 * y1 - x1 * y0)
                area += cross_product
                cx += (x0 + x1) * cross_product
                cy += (y0 + y1) * cross_product
            area *= 0.5
            if area == 0: return ring[0]
            cx /= (6 * area)
            cy /= (6 * area)
            return [cx, cy]
        
        if self.verb:
            print(f"***** Converting {self.city} buildings to game format *****")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load building data: {e}")

        items = data.get('features', data.get('geometries', []))
        if not items:
            print("WARNING: No buildings found to index.")
            return

        buildings = []
        min_lon, min_lat = float('inf'), float('inf')
        max_lon, max_lat = float('-inf'), float('-inf')
        max_found_depth = 1

        for item in items:
            geom = item.get('geometry', item)
            props = item.get('properties', {})
            if not geom or geom.get('type') not in ['Polygon', 'MultiPolygon']:
                continue

            polys_coords = [geom['coordinates']] if geom['type'] == 'Polygon' \
                            else geom['coordinates']
            height = props.get('height')
            if height is None:
                height = default_height
            
            for poly_coord in polys_coords:
                cleaned_p = []
                b_minx, b_miny = float('inf'), float('inf')
                b_maxx, b_maxy = float('-inf'), float('-inf')

                # Determine foundation depth
                if self.create_building_foundations:
                    if isinstance(poly_coord, list) and isinstance(poly_coord[0], list):
                        coord = poly_coord[0]
                    else:
                        coord = poly_coord
                    foundation = self._calculate_building_foundation(coord, height)
                else:
                    foundation = 10
                
                for ring in poly_coord:
                    if len(ring) < 3: continue
                    if ring[0] != ring[-1]: ring.append(ring[0])
                    
                    cleaned_ring = []
                    for p in ring:
                        px, py = p[0], p[1]
                        cleaned_ring.append([px, py])
                        if px < b_minx: b_minx = px
                        if py < b_miny: b_miny = py
                        if px > b_maxx: b_maxx = px
                        if py > b_maxy: b_maxy = py
                    cleaned_p.append(cleaned_ring)

                if not cleaned_p: continue

                # Update global bbox
                min_lon, min_lat = min(min_lon, b_minx), min(min_lat, b_miny)
                max_lon, max_lat = max(max_lon, b_maxx), max(max_lat, b_maxy)
                max_found_depth = max(max_found_depth, foundation)

                buildings.append({
                    "b": [b_minx, b_miny, b_maxx, b_maxy],
                    "f": foundation,
                    "p": cleaned_p,
                    "center": calculate_polygon_centroid(cleaned_p)
                })

        if not buildings:
            print("STOP: No valid buildings found after processing!")
            return

        # Grid Calculation
        lat_mid = (min_lat + max_lat) / 2
        distortion_factor = 1 / math.cos(math.radians(lat_mid))
        cs_x = CS * distortion_factor
        grid_width_cols = math.ceil((max_lon - min_lon) / cs_x)
        grid_height_rows = math.ceil((max_lat - min_lat) / CS)

        cells = {}
        for idx, b in enumerate(buildings):
            cx, cy = b['center']
            gx = max(0, min(int((cx - min_lon) / cs_x), grid_width_cols - 1))
            gy = max(0, min(int((cy - min_lat) / CS), grid_height_rows - 1))
            
            key = (gx, gy)
            if key not in cells: cells[key] = []
            cells[key].append(idx)
            del b['center'] # Clean up temporary data

        final_json = {
            "cs": CS,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "grid": [grid_width_cols + 1, grid_height_rows + 1],
            "cells": [[x, y] + idxs for (x, y), idxs in cells.items()],
            "buildings": buildings,
            "stats": {"count": len(buildings), "maxDepth": max_found_depth}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, separators=(',', ':'))
        if self.verb:
            print(f"Successfully saved building index to {output_path}")

    def create_buildings_index_binary(self, input_path, default_height=4.0):
        """
        Converts GeoJSON buildings into packed binary index streams (.bin and .bin.gz).
        """
        # Pre-compiled Struct layouts for buildings index binary
        _HEADER_STRUCT = struct.Struct("<IBBHI IIIIIII d ddddd")
        _UINT32 = struct.Struct("<I")
        _FLOAT32 = struct.Struct("<f")
        _COORD2D = struct.Struct("<2d")
        _BINARY_MAGIC = 0x49424253
        _BINARY_VERSION = 1
        _HEADER_SIZE = 88
        # Build output filename
        output_bin = input_path.replace("cleaned", "index").replace(".json", ".bin")
        if output_bin == input_path:
            output_base, _ = os.path.splitext(input_path)
            output_bin = f"{output_base}_index.bin"
        output_bin_gz = f"{output_bin}.gz"

        CS = 0.0009  # Spatial grid cell size constant (~100m)

        def round_num(num: float) -> float:
            return math.floor(num * 100000.0 + 0.5) / 100000.0

        def round_coords(coords: list) -> list:
            if not coords:
                return coords
            if isinstance(coords[0], (int, float)):
                return [round_num(c) for c in coords]
            return [round_coords(item) for item in coords]

        if self.verb:
            print(f"***** Generating building spatial index for {self.city} *****")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load building data: {e}")

        items = data.get("features", data.get("geometries", []))
        if not items:
            print("WARNING: No buildings found to index.")
            return

        buildings = []
        min_lon, min_lat = float("inf"), float("inf")
        max_lon, max_lat = float("-inf"), float("-inf")

        for index, item in enumerate(items):
            geom = item.get("geometry", item)
            props = item.get("properties", {})
            if not geom or "coordinates" not in geom:
                continue
            
            height = props.get('height')
            if height is None:
                height = default_height
            
            try:
                # Parse geometry via Shapely
                geom_shape = shape(geom)
                
                if geom_shape.is_empty:
                    continue

                # Handle MultiPolygons
                if isinstance(geom_shape, MultiPolygon):
                    # Fallback Strategy: Extract the largest constituent polygon component by area
                    geom_shape = max(geom_shape.geoms, key=lambda p: p.area)

                if not isinstance(geom_shape, Polygon):
                    continue

                # Reconstruct coordinates from the chosen/resolved single Polygon
                # Exterior ring is first, followed by internal holes (interiors)
                polygon_coords = [list(geom_shape.exterior.coords)] + \
                                 [list(hole.coords) for hole in geom_shape.interiors]

                cleaned_polygon = []
                b_minx, b_miny = float("inf"), float("inf")
                b_maxx, b_maxy = float("-inf"), float("-inf")

                for ring in polygon_coords:
                    if len(ring) < 3:
                        continue
                    # Ensure closure
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])

                    cleaned_ring = []
                    for pt in ring:
                        px, py = pt[0], pt[1]
                        cleaned_ring.append([px, py])
                        if px < b_minx:
                            b_minx = px
                        if py < b_miny:
                            b_miny = py
                        if px > b_maxx:
                            b_maxx = px
                        if py > b_maxy:
                            b_maxy = py
                    cleaned_polygon.append(cleaned_ring)

                if not cleaned_polygon:
                    continue

                b_minx, b_miny = round_num(b_minx), round_num(b_miny)
                b_maxx, b_maxy = round_num(b_maxx), round_num(b_maxy)

                # Determine foundation depth
                if self.create_building_foundations:
                    if isinstance(cleaned_polygon, list) and isinstance(cleaned_polygon[0], list):
                        coord = cleaned_polygon[0]
                    else:
                        coord = cleaned_polygon
                    foundation = self._calculate_building_foundation(coord, height)
                else:
                    foundation = 10

                buildings.append(
                    {
                        "bounds": (b_minx, b_miny, b_maxx, b_maxy),
                        "foundationDepth": foundation,
                        "polygon": round_coords(cleaned_polygon),
                    }
                )

                if b_minx < min_lon:
                    min_lon = b_minx
                if b_maxx > max_lon:
                    max_lon = b_maxx
                if b_miny < min_lat:
                    min_lat = b_miny
                if b_maxy > max_lat:
                    max_lat = b_maxy

            except Exception as error:
                print(f"Warning: Error processing building at index {index}: {error}")

        if not buildings:
            print("STOP: No valid buildings found after processing!")
            return

        min_lon, min_lat = round_num(min_lon), round_num(min_lat)
        max_lon, max_lat = round_num(max_lon), round_num(max_lat)

        cellSizeLon = CS / math.cos((((min_lat + max_lat) / 2) * math.pi) / 180)
        cols = math.ceil((max_lon - min_lon) / cellSizeLon)
        rows = math.ceil((max_lat - min_lat) / CS)

        cells_map = {}
        for building_id, b in enumerate(buildings):
            b_min_x, b_min_y, b_max_x, b_max_y = b["bounds"]
            min_col = math.floor((b_min_x - min_lon) / cellSizeLon)
            max_col = math.floor((b_max_x - min_lon) / cellSizeLon)
            min_row = math.floor((b_min_y - min_lat) / CS)
            max_row = math.floor((b_max_y - min_lat) / CS)

            for col in range(max(0, min_col), min(cols - 1, max_col) + 1):
                for row in range(max(0, min_row), min(rows - 1, max_row) + 1):
                    cells_map.setdefault((col, row), []).append(building_id)

        sorted_cells = [
            {"col": k[0], "row": k[1], "buildingIds": v}
            for k, v in sorted(cells_map.items(), key=lambda c: (c[0][1], c[0][0]))
            if v
        ]

        total_rings = sum(len(b["polygon"]) for b in buildings)
        total_coords = sum(len(ring) for b in buildings for ring in b["polygon"])
        total_cell_refs = sum(len(c["buildingIds"]) for c in sorted_cells)
        max_found_depth = max(b["foundationDepth"] for b in buildings)

        # Calculate buffer allocations
        o = _HEADER_SIZE
        bounds_offset = o
        o += len(buildings) * 32
        foundation_depths_offset = o
        o += len(buildings) * 4
        o = (o + 7) & ~7
        building_ring_offsets_offset = o
        o += (len(buildings) + 1) * 4
        ring_coord_offsets_offset = o
        o += (total_rings + 1) * 4
        o = (o + 7) & ~7
        coords_offset = o
        o += total_coords * 16
        cell_row_starts_offset = o
        o += (rows + 1) * 4
        cell_cols_offset = o
        o += len(sorted_cells) * 4
        cell_building_offsets_offset = o
        o += (len(sorted_cells) + 1) * 4
        cell_building_ids_offset = o
        o += total_cell_refs * 4

        buffer = bytearray(o)

        # Write header metadata
        _HEADER_STRUCT.pack_into(
            buffer,
            0,
            _BINARY_MAGIC,
            _BINARY_VERSION,
            0,
            0,
            len(buildings),
            cols,
            rows,
            total_rings,
            total_coords,
            len(sorted_cells),
            total_cell_refs,
            0,
            CS,
            float(max_found_depth),
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        )

        # 1. Bounds entries
        off = bounds_offset
        for b in buildings:
            struct.pack_into("<4d", buffer, off, *b["bounds"])
            off += 32

        # 2. Foundation depths
        off = foundation_depths_offset
        for b in buildings:
            _FLOAT32.pack_into(buffer, off, b["foundationDepth"])
            off += 4

        # 3. Geometry lookups
        ring_cursor = coord_cursor = 0
        ring_off = building_ring_offsets_offset
        coord_off = ring_coord_offsets_offset
        c_off = coords_offset

        for b in buildings:
            _UINT32.pack_into(buffer, ring_off, ring_cursor)
            ring_off += 4

            for ring in b["polygon"]:
                _UINT32.pack_into(buffer, coord_off, coord_cursor)
                coord_off += 4

                for pt in ring:
                    _COORD2D.pack_into(buffer, c_off, pt[0], pt[1])
                    c_off += 16
                    coord_cursor += 1
                ring_cursor += 1

        _UINT32.pack_into(buffer, ring_off, ring_cursor)
        _UINT32.pack_into(buffer, coord_off, coord_cursor)

        # 4. Spatial index sequences
        ref_cursor = cell_row = 0
        _UINT32.pack_into(buffer, cell_row_starts_offset, 0)

        for i, cell in enumerate(sorted_cells):
            while cell_row < cell["row"]:
                cell_row += 1
                _UINT32.pack_into(
                    buffer, cell_row_starts_offset + (cell_row * 4), i
                )

            _UINT32.pack_into(buffer, cell_cols_offset, cell["col"])
            cell_cols_offset += 4

            _UINT32.pack_into(
                buffer, cell_building_offsets_offset, ref_cursor
            )
            cell_building_offsets_offset += 4

            for b_id in cell["buildingIds"]:
                _UINT32.pack_into(buffer, cell_building_ids_offset, b_id)
                cell_building_ids_offset += 4
                ref_cursor += 1

        _UINT32.pack_into(
            buffer, cell_building_offsets_offset, ref_cursor
        )

        while cell_row < rows:
            cell_row += 1
            _UINT32.pack_into(
                buffer,
                cell_row_starts_offset + (cell_row * 4),
                len(sorted_cells),
            )

        # Stream outputs
        bin_data = bytes(buffer)
        with open(output_bin, "wb") as f:
            f.write(bin_data)

        bin_compressed = gzip.compress(bin_data, compresslevel=9)
        with open(output_bin_gz, "wb") as f:
            f.write(bin_compressed)
    
    def _fetch_overture_buildings(self):
        """
        Queries Overture Maps S3 bucket using DuckDB and saves to GeoJSON.
        """
        buildings_pkl = os.path.join(self.city_dir, "buildings.pkl")
        self.buildings_geojson = os.path.join(self.city_dir, 
                                              "buildings.geojson")
        
        # Check if we already have the data to avoid expensive re-downloads
        if not os.path.exists(buildings_pkl) or self.REFETCH_BUILDINGS:
            if self.verb:
                print("Fetching the latest Overture version tag...")
            OVERTURE_RELEASE = self._get_latest_overture_release()
            if self.verb:
                print(f"***** Querying Overture buildings for {self.city} *****")
            
            # Initialize DuckDB with spatial and cloud extensions
            con = duckdb.connect()
            con.execute("INSTALL spatial; LOAD spatial;")
            # Use 'httpfs' if using AWS credentials
            con.execute("INSTALL azure; LOAD azure;")
            
            # Overture S3 pathing
            s3_path = f"s3://overturemaps-us-west-2/release/" \
                      f"{OVERTURE_RELEASE}/theme=buildings/type=building/*"

            query = f"""
            SELECT 
                id,
                geometry,
                names.primary as name,
                height
            FROM read_parquet('{s3_path}', hive_partitioning=1)
            WHERE bbox.xmin >= {self.bbox[0]} AND bbox.xmax <= {self.bbox[2]}
              AND bbox.ymin >= {self.bbox[1]} AND bbox.ymax <= {self.bbox[3]}
            """

            try:
                # Fetch to DataFrame
                df = con.query(query).to_df()
                
                if df.empty:
                    print(f"WARNING: No buildings found in Overture for bbox "\
                          f"{self.bbox}")
                    return
                
                if self.verb:
                    print("Converting WKB to Geometry...", flush=True)
                # Convert binary geometry to Shapely objects
                df["geometry"] = df["geometry"].apply(
                    lambda x: wkb.loads(bytes(x)) 
                              if isinstance(x, (bytes, bytearray)) else x
                )
                
                # Pickle for faster loading in the future
                df.to_pickle(buildings_pkl)
            except Exception as e:
                raise RuntimeError(f"Overture data fetch failed: {e}")
            finally:
                con.close()
            
        else:
            if self.verb:
                print("***** Loading previously downloaded buildings file: *****")
                print("    "+buildings_pkl)
            df = pd.read_pickle(buildings_pkl)
        
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        if self.verb:
            print(f"Saving to {self.buildings_geojson}...", flush=True)
        gdf.to_file(self.buildings_geojson, driver='GeoJSON')
    
    @staticmethod
    def _get_latest_overture_release():
        headers = { "User-Agent" : "Subway-Builder-Modded/Depot" }
        with httpx.Client(http2=True, timeout=10) as client:
            response = client.get("https://stac.overturemaps.org/catalog.json", headers=headers)
        #response = requests.get("https://stac.overturemaps.org/catalog.json", headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("latest")
    
    def get_utm_epsg(self):
        """
        Calculates the UTM EPSG code using the instance's bbox attribute.
        Automatically called when bbox is set.
        """
        w, s, e, n = self.bbox
        
        # Logic remains the same
        center_lon = (w + e) / 2
        center_lat = (s + n) / 2
        
        zone = int(math.floor((center_lon + 180) / 6) + 1)
        
        # Determine N/S hemisphere prefix
        epsg_prefix = 32600 if center_lat >= 0 else 32700
        self.epsg = f"epsg:{epsg_prefix + zone}"
    
    def process_buildings(self):
        """
        Overture fetch -> Mapshaper cleanup -> Game conversion.
        """
        if self.verb:
            print("***** Processing Buildings *****")
        
        if self.buildings_geojson is None:
            # 1. Fetch buildings from Overture
            self._fetch_overture_buildings()

        # 2. Mapshaper Cleanup
        cleaned_json = os.path.join(self.city_dir, "buildings_cleaned.json")
        mapshaper_cmd = (
            f"node --max-old-space-size={self.RAM} $(which mapshaper) "
            f"{self.buildings_geojson} -proj {self.epsg} -snap 0.5 -clean "
            f"-filter 'this.area > {self.building_index_filter_size}' "
            f"-simplify dp interval={self.building_index_simplification} "
            f"-proj wgs84 -o precision=0.00001 {cleaned_json}"
        )
        self._run_command(mapshaper_cmd)

        # 3. GeoJSON to Game Format
        # Path adjusted based on your bash script relative paths
        self._convert_to_game_format(cleaned_json)
        # New binary format for buildings index
        self.create_buildings_index_binary(cleaned_json)
        
    def process_roads_and_aeroways(self, roads_list=['motorway', 'motorway_link', 
                                                     'trunk', 'trunk_link', 
                                                     'primary', 'primary_link', 
                                                     'secondary', 'secondary_link', 
                                                     'tertiary', 'tertiary_link', 
                                                     'unclassified', 'residential']):
        """
        Extracts roads and aeroways, applies JQ filters and buffering.

        Inputs
        ------
        roads_list: list, strings. Road types to include in the extraction.
                    Default: ['motorway', 'motorway_link', 'trunk', 'trunk_link', 
                              'primary', 'primary_link', 'secondary', 
                              'secondary_link', 'tertiary', 'tertiary_link', 
                              'unclassified', 'residential']
        """
        if self.verb:
            print("***** Processing Roads and Aeroways *****")
        city_pbf = os.path.join(self.city_dir, f"{self.city.lower()}.osm.pbf")
        roads_pbf = os.path.join(self.city_dir, "roads.pbf")
        roads_geojson = os.path.join(self.city_dir, "roads.geojson")
        
        # 1. Roads
        roads_str = ",".join(roads_list)
        self._run_command(["osmium", "tags-filter", city_pbf, 
                           f"w/highway={roads_str}", "-o", roads_pbf, 
                           "--overwrite"])
        this_dir = os.path.dirname(os.path.abspath(__file__))
        self._run_command(["osmium", "export", roads_pbf, 
                           "-c", os.path.join(this_dir, "roads_config.json"),
                           "-o", roads_geojson, "--geometry-types=linestring", 
                           "--overwrite"])
        if self.cleanup_files:
            os.remove(roads_pbf)
        
        jq_roads = (
            '.features |= map({type: "Feature", properties: { '
            'roadClass: (if .properties.highway == "motorway" or '
                           '.properties.highway == "trunk" then "highway" '
            'elif .properties.highway == "primary" or '
                 '.properties.highway == "secondary" then "major" '
                 'else "minor" end), '
            'structure: (if .properties.bridge then "bridge" '
                      'elif .properties.tunnel then "tunnel" '
                      'else "normal" end), '
            f'name: ({self._get_road_name_jq_expression()})}}, geometry: .geometry}})'
        )
        self._apply_jq(roads_geojson, jq_roads)
        
        # 2. Aeroways
        aero_pbf = os.path.join(self.city_dir, "runways_taxiways.pbf")
        aero_geojson = os.path.join(self.city_dir, "runways_taxiways.geojson")
        
        self._run_command(["osmium", "tags-filter", city_pbf, 
                           "wr/aeroway=runway,taxiway", "-o", aero_pbf, 
                           "--overwrite"])
        self._run_command(["osmium", "export", aero_pbf, "-o", aero_geojson, 
                           "--geometry-types=linestring,polygon", 
                           "--add-unique-id=type_id", "--overwrite"])
        if self.cleanup_files:
            os.remove(aero_pbf)
        
        jq_aero = (
            '.features |= map({type: "Feature", properties: { '
            'roadType: (.properties.aeroway // '
                       '.properties.roadType // "runway"), '
            'z_order: 0, osm_way_id: (.id // .properties["@id"] | '
                                            'sub("^[awrn]"; "") | tostring), '
            'area: 0}, geometry: (if .geometry.type == "MultiPolygon" then '
                    '{type: "Polygon", coordinates: .geometry.coordinates[0]} '
            'else .geometry end)})'
        )
        self._apply_jq(aero_geojson, jq_aero)
        
        # 3. Buffer Aeroways
        self._buffer_linestrings(aero_geojson)
    
    def generate_pmtiles(self):
        """
        Full Planetiler -> Tile-join -> Tippecanoe -> PMTiles flow.
        """
        if self.verb:
            print("***** Generating PMTiles *****")
        base_name = self.city.lower()
        path_prefix = os.path.join(self.city_dir, base_name)
        city_pbf = f"{path_prefix}.osm.pbf"
        self.nobuildings_geojson = os.path.join(self.city_dir, f"{self.city.lower()}-nobuildings.geojson")
        self.raw_mbtiles = f"{path_prefix}.mbtiles"
        clean_mbtiles = f"{path_prefix}-clean.mbtiles"
        fixed_mbtiles = f"{path_prefix}-fixed.mbtiles"
        merged_mbtiles = f"{path_prefix}-merged.mbtiles"
        foundations_mbtiles = f"{path_prefix}-foundations.mbtiles"
        self.buildings_mbtiles = os.path.join(self.city_dir, "buildings.mbtiles")
        final_pmtiles = os.path.join(self.city_dir, self.city+"-nolabels.pmtiles")
        foundations_pmtiles = os.path.join(self.city_dir, self.city+"_foundations.pmtiles")
        
        # 1. Planetiler
        bounds_str = ",".join(map(str, self.bbox))
        self._run_command([
            "java", "-Xmx16g", "-jar", self.planetiler_path,
            f"--osm-path={city_pbf}", 
            f"--output={self.raw_mbtiles}",
            f"--bounds={bounds_str}",
            "--download", 
            "--minzoom=0", 
            f"--maxzoom={self.maxzoom}", 
            "--only-layers=aerodrome_label,aeroway,boundary,landcover,landuse,park,water,water_name,waterway,transportation,roads",
            "--force"
        ])
        
        # 2. Initial Tile-join Clean
        self._run_command([
            "tile-join", "--force", "--rename=landcover:landuse", 
            "--rename=park:landuse", "--exclude=housenumber", 
            "--exclude=aerodrome_label", "--exclude=mountain_peak", 
            "--exclude=transportation_name", 
            "--exclude=building", "--exclude=buildings", "-pk", 
            "-o", clean_mbtiles, self.raw_mbtiles
        ])
        
        # 3. Fix the tiles as SB expects
        self.fix_mbtiles() # Turns clean_mbtiles into fixed_mbtiles
        
        # Ensure ocean depth data is available
        if self.create_ocean_foundations:
            # Ocean depth index
            if self.bathy_data is None:
                self.load_bathymetry_data()
            # Ocean depth map layer
            self._generate_ocean_depth_tiles()
        
        if self.cleanup_files:
            os.remove(self.raw_mbtiles)
            os.remove(clean_mbtiles)
        
        # 4. Building overlays, including foundations
        self._generate_building_tiles()
        
        # Clean metadata
        self._update_mbtiles_metadata(fixed_mbtiles)
        self._update_mbtiles_metadata(self.buildings_mbtiles)
        
        # 5. Merge buildings and foundations
        merge_cmd = [
            "tile-join", "--force", 
            "-o", merged_mbtiles,
            fixed_mbtiles, self.buildings_mbtiles
        ]
        if self.create_ocean_foundations:
            merge_cmd.append(self.ocean_foundations_mbtiles)
        merge_cmd.append("--no-tile-size-limit")
        self._run_command(merge_cmd)
        
        if self.cleanup_files:
            os.remove(fixed_mbtiles)
            os.remove(self.buildings_mbtiles)
            if self.create_ocean_foundations:
                os.remove(self.ocean_foundations_mbtiles)
        
        # 6. Metadata and PMTiles Convert
        self._update_mbtiles_metadata(merged_mbtiles)
        self._run_command(["pmtiles", "convert", merged_mbtiles, 
                           final_pmtiles])
        if self.create_building_foundations:
            self._update_mbtiles_metadata(self.buildings_foundations_mbtiles)
            self._run_command(["pmtiles", "convert", 
                               self.buildings_foundations_mbtiles, 
                               foundations_pmtiles])
        
        if self.cleanup_files:
            os.remove(merged_mbtiles)
            #os.remove(foundations_mbtiles)
            if self.create_building_foundations:
                os.remove(self.buildings_foundations_mbtiles)
        
    def _apply_jq(self, filepath, filter_str):
        """
        Internal helper for JQ operations.
        """
        tmp_file = filepath + ".tmp"
        with open(tmp_file, 'w') as out_f:
            subprocess.run(["jq", "-c", filter_str, filepath], stdout=out_f, 
                            check=True)
        os.replace(tmp_file, filepath)
    
    def _buffer_linestrings(self, data_input, buffer_width=0.00015):
        """
        Flexible helper to convert LineStrings/MultiLineStrings to strict Polygons.
        
        Accepts:
          - str: A filepath to a GeoJSON file (loads, modifies, and overwrites it).
          - dict: A GeoJSON-like feature dictionary or FeatureCollection.
          - Shapely geometry: A LineString/MultiLineString (returns a list of Polygons).
        """
        # Case 1: input is shapely geometry
        if hasattr(data_input, 'geom_type'):
            if data_input.geom_type in ["LineString", "MultiLineString"]:
                buffered = data_input.buffer(buffer_width, cap_style=2)
                if buffered.geom_type == "MultiPolygon":
                    return list(buffered.geoms)
                return [buffered]
            return [data_input]

        # Case 2: input is filepath (str)
        is_filepath = isinstance(data_input, str)
        if is_filepath:
            with open(data_input, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = data_input # Assume it's already a GeoJSON dict

        # Process geojson structure
        unpacked_features = []
        features = data.get('features', [data]) if isinstance(data, dict) and 'type' in data else []
        
        for feat in features:
            if feat.get('geometry', {}).get('type') in ['LineString', 'MultiLineString']:
                geom = shape(feat['geometry'])
                buffered = geom.buffer(buffer_width, cap_style=2)
                
                if buffered.geom_type == "MultiPolygon":
                    for poly in buffered.geoms:
                        new_feat = {
                            'type': 'Feature',
                            'properties': feat.get('properties', {}).copy(),
                            'geometry': mapping(poly)
                        }
                        new_feat['geometry']['type'] = 'Polygon'
                        unpacked_features.append(new_feat)
                else:
                    feat['geometry'] = mapping(buffered)
                    feat['geometry']['type'] = 'Polygon'
                    unpacked_features.append(feat)
            else:
                unpacked_features.append(feat)

        # Return or save based on input
        if is_filepath:
            data['features'] = unpacked_features
            with open(data_input, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return None
        else:
            if 'features' in data:
                data['features'] = unpacked_features
                return data
            return unpacked_features[0] if unpacked_features else data
    
    def _calculate_buffer(self, zoom):
        target_meters = 10
        extent = 4096
        meters_per_tile = 40075016.686 / (2**zoom)
        units_per_meter = extent / meters_per_tile
        target_buffer = (target_meters / 2) * units_per_meter
        safe_buffer = max(target_buffer, 4.0)
        return safe_buffer
    
    def load_bathymetry_data(self, opendap_url="https://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2026/sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.nc", 
                             buffer=0.05, CELL_SIZE=0.0027,
                             resolution_multiplier=4):
        """
        Connects to GEBCO's bathymetry data via OPeNDAP and processes 
        it into SB's ocean_depths_index format.
        
        Inputs
        ------
        opendap_url: str. URL to OPeNDAP endpoint for GEBCO bathymetry data.
        buffer: float. Buffer size in degrees when extracting the bathymetry 
                data. Setting this too small, zero, or negative will 
                mess up the ocean depth data at the edges of the map. Don't 
                change this number unless you have a very good reason.
                Default: 0.05
        CELL_SIZE: float. Nominal cell size for ocean collision calculations. 
                   Smaller values increase file size. Larger values reduce 
                   performance. Don't change this number unless you have a 
                   very good reason.
                   Default: 0.0027
        resolution_multiplier: int or float. Multiplier for the resolution of 
                               the bathymetry data. Smooths the contours at 
                               the cost of increased file sizes. Use 1 for no 
                               resolution change.
                               Default: 4
        """
        self.fdepths = os.path.join(self.city_dir, "ocean_depth_index.json.gz")
        
        if not self.reprocess_bathymetry_data:
            # Open and load the gzipped JSON data, if available
            if os.path.exists(self.fdepths):
                with gzip.open(self.fdepths, "rt", encoding="utf-8") as f:
                    self.bathy_data = json.load(f)
                    if self.verb:
                        print("Loaded previously processed bathymetry data")
                return
        
        if self.verb:
            print("Loading and processing bathymetry data")
        
        self.bathy = xr.open_dataset(opendap_url)
        min_lon, min_lat, max_lon, max_lat = self.bbox
        # Load the raw bathymetry subset
        self.bathy = self.bathy.elevation.sel(
            lon=slice(min_lon - buffer, max_lon + buffer), 
            lat=slice(min_lat - buffer, max_lat + buffer)
        ).load()
        
        # Set the target grid, and interpolate to it
        step_x = CELL_SIZE / float(np.cos(np.radians((min_lat + max_lat) / 2.)))
        step_y = CELL_SIZE
        grid_x = np.arange(min_lon, max_lon+step_x, step_x)
        grid_y = np.arange(min_lat, max_lat+step_y, step_y)
        
        self.bathy = self.bathy.interp(
            lon=grid_x, 
            lat=grid_y, 
            method="linear"
        )
        
        # Extract coordinate arrays
        lons = self.bathy.lon.values
        lats = self.bathy.lat.values
        depths = self.bathy.values
        
        if depths.min() < 0:
            # Set up depth contour levels (strictly below sea level)
            depth_min = float(depths.min())
            # Standard grid of depths - reverse it so it goes deepest -> shallowest
            DEPTH_LEVELS = -1 * np.concatenate((np.arange(   0,    40,    5, dtype=float), 
                                                np.arange(  40,   100,   10, dtype=float), 
                                                np.arange( 100,   500,   50, dtype=float),
                                                np.arange( 500,  1000,  100, dtype=float),
                                                np.arange(1000, 11000, 1000, dtype=float)))[::-1]
            DEPTH_LEVELS = DEPTH_LEVELS[DEPTH_LEVELS > depth_min]
            DEPTH_LEVELS = np.insert(DEPTH_LEVELS, 0, depth_min)
            if self.verb:
                print("  Depth levels:", DEPTH_LEVELS)
            
            # Increase the resolution multiplier for smoother curves
            dense_lons = np.linspace(lons[0], lons[-1], len(lons) * resolution_multiplier)
            dense_lats = np.linspace(lats[0], lats[-1], len(lats) * resolution_multiplier)
            dense_mesh_lon, dense_mesh_lat = np.meshgrid(dense_lons, dense_lats)
            interp_func = RegularGridInterpolator(
                (lats, lons), 
                depths, 
                method='cubic',  
                bounds_error=False, 
                fill_value=None
            )
            interp_points = np.stack([dense_mesh_lat.ravel(), dense_mesh_lon.ravel()], axis=-1)
            dense_depths = interp_func(interp_points).reshape(dense_mesh_lon.shape)
            
            # Generate contours
            fig, ax = plt.subplots()
            contour_set = ax.contourf(dense_lons, dense_lats, dense_depths, levels=DEPTH_LEVELS)
            plt.close(fig)

            # First Pass: Group geometries by depth level
            raw_stacked_contours = []
            for i in range(len(contour_set.allsegs)):
                segments = contour_set.allsegs[i]
                level = contour_set.levels[i] 
                if level >= 0 or not segments:
                    continue
                
                level_polys = []
                for seg in segments:
                    if len(seg) < 3:
                        continue
                    coords = seg.tolist()
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                        
                    try:
                        poly_geo = Polygon(coords)
                        if not poly_geo.is_valid:
                            poly_geo = poly_geo.buffer(0)
                        if not poly_geo.is_empty and poly_geo.area > 0:
                            if poly_geo.geom_type == "Polygon":
                                level_polys.append(poly_geo)
                            elif poly_geo.geom_type == "MultiPolygon":
                                level_polys.extend(list(poly_geo.geoms))
                    except Exception:
                        continue
                        
                if level_polys:
                    raw_stacked_contours.append((level, unary_union(level_polys)))

            # Sort from deepest to shallowest (ascending order: e.g., -16, -12, -8, -4)
            # This ensures we process the deep layers first so shallow layers can slice them up
            raw_stacked_contours.sort(key=lambda x: x[0], reverse=False)

            # Corrected Pass: Deeper polygons get clipped by shallower ones
            contours_by_level = []
            
            for i in range(len(raw_stacked_contours)):
                current_level, current_geom = raw_stacked_contours[i]
                
                # Look ahead in our deepest-to-shallowest array to find all shallower layers
                # Subtract every shallower polygon that overlaps this deep one
                shallower_geoms = [
                    sh_geom for sh_lvl, sh_geom in raw_stacked_contours[i+1:] 
                    if sh_lvl > current_level
                ]
                
                if shallower_geoms:
                    # Combine all shallow caps into one unified cutter mask
                    shallow_mask = unary_union(shallower_geoms)
                    if current_geom.intersects(shallow_mask):
                        # Punch holes in the deep trench where the shallow shelf caps it
                        current_geom = current_geom.difference(shallow_mask)
                
                if not current_geom.is_valid:
                    current_geom = current_geom.buffer(0)
                    
                if not current_geom.is_empty and current_geom.area > 1e-9:
                    contours_by_level.append((current_level, current_geom))
        else:
            if self.verb:
                print("  No ocean depths detected")
            contours_by_level = []
        
        if self.raw_mbtiles is None or not os.path.exists(self.raw_mbtiles):
            raise FileNotFoundError(f"Valid MBTiles source path required for water parsing. Got: {self.raw_mbtiles}")
        
        if self.verb:
            print(f"  Decoding water & ocean polygons directly from {self.raw_mbtiles} at zoom {self.maxzoom}")
            
        water_polygons = []
        # Target high resolution Zoom level `self.maxzoom`
        tiles = list(mercantile.tiles(self.bbox[0], self.bbox[1], 
                                      self.bbox[2], self.bbox[3], 
                                      self.maxzoom))
        
        conn = sqlite3.connect(self.raw_mbtiles)
        cursor = conn.cursor()
        
        for t in tiles:
            tms_y = (1 << self.maxzoom) - 1 - t.y # Invert Y for standard TMS lookup scheme
            cursor.execute(
                f"SELECT tile_data FROM tiles WHERE zoom_level={self.maxzoom} AND tile_column=? AND tile_row=?",
                (t.x, tms_y)
            )
            row = cursor.fetchone()
            if not row:
                continue
                
            try:
                tile_bytes = gzip.decompress(row[0])
            except Exception:
                tile_bytes = row[0]
                
            tile_bounds = mercantile.bounds(t)
            decoded = mapbox_vector_tile.decode(tile_bytes)
            
            for layer_name in ['water', 'waterway']:
              if layer_name in decoded:
                tile_bounds = mercantile.bounds(t)
                # Calculate the scaling factors from local tile space to WGS84
                extent = decoded[layer_name].get('extent', 4096)
                
                # Precompute bounds dimensions
                lon_width = tile_bounds.east - tile_bounds.west
                lat_height = tile_bounds.north - tile_bounds.south

                for feature in decoded[layer_name]['features']:
                    props = feature.get('properties', {})
                    # Check both 'kind' and 'class' as different schema flavors use one or the other
                    feature_kind = props.get('kind', props.get('class', ''))
                    if feature_kind == 'ditch':
                        continue
                    
                    raw_geom = feature['geometry']
                    try:
                        # Convert the raw tile feature into a Shapely object (in pixel space 0-4096)
                        geom_shape = shape(raw_geom)
                        
                        # If it's a line, calculate buffer
                        if geom_shape.geom_type in ["LineString", "MultiLineString"]:
                            safe_buffer = self._calculate_buffer(self.maxzoom)
                            processed_shapes = self._buffer_linestrings(geom_shape, buffer_width=safe_buffer)
                        else:
                            processed_shapes = [geom_shape]
                        
                        # Project the resulting pixel-space polygons into geographic space
                        for pixel_poly in processed_shapes:
                            
                            # Inline projection function that converts a shape's dictionary back to degrees
                            def transform_coords(geom_dict):
                                if geom_dict['type'] == 'Point':
                                    px, py = geom_dict['coordinates']
                                    return [
                                        tile_bounds.west + (px / extent) * lon_width,
                                        tile_bounds.south + (py / extent) * lat_height
                                    ]
                                elif geom_dict['type'] in ['LineString', 'MultiPoint']:
                                    return [transform_coords({'type': 'Point', 'coordinates': c}) for c in geom_dict['coordinates']]
                                elif geom_dict['type'] in ['Polygon', 'MultiLineString']:
                                    return [[transform_coords({'type': 'Point', 'coordinates': c}) for c in ring] for ring in geom_dict['coordinates']]
                                elif geom_dict['type'] == 'MultiPolygon':
                                    return [[[transform_coords({'type': 'Point', 'coordinates': c}) for c in ring] for ring in poly] for poly in geom_dict['coordinates']]
                                return geom_dict['coordinates']

                            # Export the pixel polygon back to a dictionary format so we can re-project it
                            pixel_geojson = mapping(pixel_poly)
                            
                            geo_geojson = {
                                'type': pixel_geojson['type'],
                                'coordinates': transform_coords(pixel_geojson)
                            }
                            
                            geo_poly = shape(geo_geojson)
                            
                            if not geo_poly.is_valid:
                                geo_poly = geo_poly.buffer(0)
                                
                            if not geo_poly.is_empty:
                                if geo_poly.geom_type == "Polygon":
                                    water_polygons.append(geo_poly)
                                elif geo_poly.geom_type == "MultiPolygon":
                                    water_polygons.extend(list(geo_poly.geoms))
                    except Exception:
                        continue
                        
        conn.close()
        
        # Unify tiles and constrain strictly to the clipping bounding box
        bbox_poly = box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])
        osm_water_layer = unary_union(water_polygons).intersection(bbox_poly)
        
        if not osm_water_layer.is_valid:
            osm_water_layer = osm_water_layer.buffer(0)
        
        # Combine EVERY single parsed contour layer to map where GEBCO says ANY water is
        all_gebco_water_geom = unary_union([geom for lvl, geom in contours_by_level])
        
        # Find everywhere OSM says there is water, but GEBCO left a gap (is empty / land)
        water_gaps = osm_water_layer.difference(all_gebco_water_geom)
        
        # Clean precision anomalies
        if not water_gaps.is_empty and water_gaps.area > 1e-7:
            if not water_gaps.is_valid:
                water_gaps = water_gaps.buffer(0)
                
            if self.verb:
                print("  Patching water gaps at -4m depth")
            
            # Find if a -5m layer index already exists in contours_by_level
            minus_5_idx = next((i for i, (lvl, _) in enumerate(contours_by_level) if lvl == -5), None)
            
            if minus_5_idx is not None:
                existing_5m_geom = contours_by_level[minus_5_idx][1]
                updated_5m_geom = unary_union([existing_5m_geom, water_gaps])
                contours_by_level[minus_5_idx] = (-5, updated_5m_geom)
            else:
                contours_by_level.append((-5, water_gaps))
        
        # Process cells
        depth_entries = []
        cell_refs = {}

        def _iter_polygon_parts(geom):
            if geom.geom_type == "Polygon":
                return [geom]
            elif geom.geom_type == "MultiPolygon":
                return list(geom.geoms)
            return []

        # Process every cell in latitude-corrected grid matrix
        if self.verb:
            print("  Processing cells")
        # Pre-clip contours to water mask
        water_clipped_contours = []
        for level, level_geom in contours_by_level:
            if level_geom.is_empty:
                continue
            # Pre-clip this bathymetry contour down to the water boundaries
            clipped_level = level_geom.intersection(osm_water_layer)
            if not clipped_level.is_empty:
                water_clipped_contours.append((level, clipped_level))

        if not water_clipped_contours:
            if self.verb:
                print("  No contours overlap the water mask. Skipping cell loop.")
            return

        # Build R-tree spatial index
        # Keep a simple list of the geometries for the tree
        index_geoms = [geom for _, geom in water_clipped_contours]
        spatial_tree = STRtree(index_geoms)

        # Cache float transformations ahead of time
        final_level_cache = {
            level: (int(level) if float(level).is_integer() else round(float(level), 2))
            for level, _ in water_clipped_contours
        }

        # Evaluate cells
        if self.verb:
            print("  Evaluating cells")
        
        # Sanitize contours
        valid_water_contours = []
        for level, geom in water_clipped_contours:
            if not shapely.is_valid(geom):
                geom = shapely.make_valid(geom)
            valid_water_contours.append((level, geom))

        # Precompute Y bounds
        y_bounds = [(min_lat + (cy * step_y), min_lat + ((cy + 1) * step_y)) for cy in range(len(grid_y))]
        
        # Select optimal number of parallel workers
        # Never use more than 1/2 the available cores
        ncores = max(1, min(self.ncores, os.cpu_count() // 2))
        if self.verb:
            core_str = "core" if ncores == 1 else "cores"
            print(f"  Processing ocean depth indices using {ncores} {core_str}")
        
        all_cx = list(range(len(grid_x)))
        chunk_size = int(math.ceil(len(all_cx) / ncores / 50))
        cx_chunks = [all_cx[i:i + chunk_size] for i in range(0, len(all_cx), chunk_size)]
        
        # Calculate depth indices in parallel
        parallel_results = []
        with ProcessPoolExecutor(max_workers=ncores) as executor:
            # Submit all chunks to start running in parallel immediately
            futures = {
                executor.submit(
                    self._process_columns_worker,
                    chunk, min_lon, step_x, step_y, len(grid_y), y_bounds,
                    spatial_tree, valid_water_contours, final_level_cache
                ): chunk 
                for chunk in cx_chunks
            }
            
            with tqdm(total=len(futures), desc="Processing Grid Chunks", unit="chunk") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        parallel_results.append(result)
                    except Exception as e:
                        print(f"\n[ERROR] Chunk failed with exception: {e}")
                    
                    pbar.update(1)

        # Merge isolated process outputs sequentially back into the instance payload
        depth_entries = []
        cell_refs = {}
        global_entry_offset = 0

        for local_depth_entries, local_cell_refs in parallel_results:
            depth_entries.extend(local_depth_entries)
            for (cx, cy), indices in local_cell_refs.items():
                global_indices = [idx + global_entry_offset for idx in indices]
                cell_refs.setdefault((cx, cy), []).extend(global_indices)
            global_entry_offset += len(local_depth_entries)
        
        # Format sparse lookup list
        cells = [
            [int(cx), int(cy), *[int(i) for i in refs]]
            for (cx, cy), refs in sorted(cell_refs.items())
        ] or [[0, 0, 0]]
        
        # Format for game and save
        all_depths = [d['d'] for d in depth_entries]
        min_d = min(all_depths) if all_depths else 0.0

        self.bathy_data = {
            "cs": float(CELL_SIZE),
            "bbox": self.bbox,
            "grid": [len(grid_x), 
                     len(grid_y)],
            "cells": cells,
            "depths": depth_entries,
            "stats": {
                "count": int(len(depth_entries)),
                "minDepth": int(min_d) if float(min_d).is_integer() else float(min_d),
                "maxDepth": 0
            }
        }

        with gzip.open(self.fdepths, "wt", encoding="utf-8") as f:
            json.dump(self.bathy_data, f, separators=(',', ':'))
        
        if self.verb:
            print(f"Successfully generated ocean depth index. Processed {len(cells)} active grid cells.")
    
    @staticmethod
    def _process_columns_worker(cx_chunk, min_lon, step_x, step_y, grid_y_len, y_bounds, 
                                spatial_tree, valid_water_contours, final_level_cache):
        """
        Processes water depth index in an isolated process. 
        No access to `self` to avoid Pickling Errors.
        """
        local_depth_entries = []
        local_cell_refs = {}
        
        def _iter_polygon_parts(geom):
            if geom.geom_type == "Polygon":
                return [geom]
            elif geom.geom_type == "MultiPolygon":
                return list(geom.geoms)
            return []

        for cx in cx_chunk:
            cell_min_x = min_lon + (cx * step_x)
            cell_max_x = min_lon + ((cx + 1) * step_x)
            
            for cy in range(grid_y_len):
                cell_min_y, cell_max_y = y_bounds[cy]
                cell_poly = box(cell_min_x, cell_min_y, cell_max_x, cell_max_y)
                
                intersecting_indices = spatial_tree.query(cell_poly)
                
                for idx in intersecting_indices:
                    level, level_geom = valid_water_contours[idx]
                    
                    if cell_poly.within(level_geom):
                        clipped_parts = [cell_poly]
                    else:
                        clipped = shapely.clip_by_rect(
                            level_geom, 
                            cell_min_x, cell_min_y, 
                            cell_max_x, cell_max_y
                        )
                        if clipped.is_empty or clipped.area <= 1e-9:
                            continue
                        clipped_parts = _iter_polygon_parts(clipped)
                        
                    for part in clipped_parts:
                        if part.is_empty:
                            continue
                        
                        exterior = [[round(c[0], 6), round(c[1], 6)] for c in part.exterior.coords]
                        holes = [[[round(c[0], 6), round(c[1], 6)] for c in hole.coords] for hole in part.interiors]
                        pb = [round(x, 6) for x in part.bounds]
                        
                        final_level = final_level_cache[level]
                        entry_index = len(local_depth_entries)
                        
                        local_depth_entries.append({
                            "b": pb,
                            "d": final_level,
                            "p": [exterior] + holes,
                        })
                        local_cell_refs.setdefault((cx, cy), []).append(entry_index)
                        
        return local_depth_entries, local_cell_refs
    
    def _generate_ocean_depth_tiles(self):
        """
        Creates ocean_foundations mbtiles from the ocean_depth_index.json
        """
        self.ocean_foundations_geojson = os.path.join(self.city_dir, "ocean_foundations.geojson")
        self.ocean_foundations_mbtiles = os.path.join(self.city_dir, "ocean_foundations.mbtiles")
        if self.bathy_data is None:
            self.load_bathymetry_data()
        
        # Process bathymetry data into geojson format
        bathy_geojson_data = {'type' : 'FeatureCollection', 'features' : []}
        for f in self.bathy_data['depths']:
            feat = {
                'type' : 'Feature',
                'geometry' : {'type' : 'Polygon', 'coordinates' : f['p']},
                'properties' : {'kind' : 'ocean_foundation', 'depth_min' : f['d']}
            }
            bathy_geojson_data['features'].append(feat)
        
        with open(self.ocean_foundations_geojson, 'w', encoding='utf-8') as f:
            json.dump(bathy_geojson_data, f, indent=2)
        
        tippe_cmd = [
            "tippecanoe", "-o", self.ocean_foundations_mbtiles,
            "--layer=ocean_foundations", "--include=depth_min", "--include=kind", 
            "--no-tile-size-limit", 
            "-Z8", f"-z{self.maxzoom}", self.ocean_foundations_geojson, "--force"
        ]
        self._run_command(tippe_cmd)
        
        if self.verb:
            print("Ocean depth tiles created")
    
    def _get_kind_and_rank(self, val):
        """
        Helper to map OSM/Planetiler tags to game-engine specific kinds and 
        ranks.
        """
        priority = {
            'aeroway': 400, 'river': 200, 'park': 189, 'aerodrome': 189
        }
        if not isinstance(val, str): return 'other', None, 0
        v = val.lower()
        if 'runway' in v:
            return 'aeroway', 'runway', priority['aeroway']
        if 'taxiway' in v:
            return 'aeroway', 'taxiway', priority['aeroway']
        if 'river' in v:
            return 'river', None, priority['river']
        if any(x in v for x in ['park', 'nature_reserve', 'cemetery', 'pitch', 
                                'zoo', 'grass', 'wood', 'forest', 'scrub', 
                                'wetland', 'wilderness_area', 
                                'wildlife_sanctuary', 'state_forest', 
                                'national_wildlife_refuge', 'management_area', 
                                'wildlife_management_area']):
            return 'park', None, priority['park']
        if 'aerodrome' in v or \
           ('military' in v and self.color_military_like_aerodrome):
            return 'aerodrome', None, priority['aerodrome']
        return v, None, 0

    def _process_tile_worker(self, tile_tuple):
        """
        Worker function to handle vector tile re-mapping.
        """
        z, x, y, data = tile_tuple
        tile_pbf = zlib.decompress(data, 16 + zlib.MAX_WBITS)
        decoded = mapbox_vector_tile.decode(tile_pbf)
        new_layers_data = {}
        # Temporary storage for water geometries to be dissolved
        water_geoms_to_dissolve = []
        water_id_map = [] # List of tuples: (id, geometry)
        tile_bounds = box(0, 0, 4096, 4096)
        
        safe_buffer = self._calculate_buffer(z)
        
        water_kinds = {
            'ocean', 'river', 'canal', 'drain', 'swimming_pool', 'lake', 
            'cenote', 'lagoon', 'oxbow', 'rapids', 'stream', 'stream_pool', 
            'pond', 'reflecting_pool', 'reservoir'
        }

        for layer_name, layer_content in decoded.items():
            is_bldg_layer = 'building' in layer_name.lower()
            for feature in layer_content['features']:
                old_props = feature.get('properties', {})
                kind, detail, rank = self._get_kind_and_rank(
                    old_props.get('aeroway') or old_props.get('class') or ""
                )

                if kind in water_kinds:
                    dest = "water"
                    final_kind = kind
                    final_rank = rank
                    
                    geom = shape(feature['geometry'])
                    
                    if 'LineString' in feature['geometry']['type']:
                        processed_shapes = self._buffer_linestrings(geom, buffer_width=safe_buffer)
                        
                        if processed_shapes:
                            # Re-pack into a valid MultiPolygon if there are multiple parts, 
                            # or keep as a single Polygon to prevent malformed GeoJSON arrays.
                            if len(processed_shapes) > 1:
                                final_geom = MultiPolygon(processed_shapes)
                            else:
                                final_geom = processed_shapes[0]
                                
                            # Let mapping natively write structural keys ("type", "coordinates", holes)
                            feature['geometry'] = mapping(final_geom)
                            
                            # Update our tracking variable for the downstream dissolve step
                            geom = final_geom
                                
                            feature['geometry'] = mapping(final_geom)

                    if not geom.is_empty:
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                        water_geoms_to_dissolve.append(geom)
                        water_id_map.append((feature.get('id'), geom))
                    continue # These features will be added after the loop
                elif (kind == 'aeroway' or \
                      'runway' in str(old_props).lower() or \
                      'taxiway' in str(old_props).lower()):
                    if self.debug:
                        dest, final_kind, final_rank = "roads", "aeroway", 400
                        if not detail:
                            detail = (
                                'runway' if 'runway' in str(old_props).lower() 
                                else 'taxiway'
                            )
                    else:
                        # Map layer not needed - runways_taxiways.geojson handles this
                        continue
                elif kind == 'aerodrome':
                    dest, final_kind, final_rank = "landuse", "aerodrome", 189
                elif is_bldg_layer or kind == 'building':
                    dest, final_kind, final_rank = "buildings", "building", 400
                elif layer_name in ["transportation", "roads", "navigation"]:
                    if self.debug:
                        dest, final_kind, final_rank = "roads", kind, rank
                    else:
                        # Map layer not needed - roads.geojson handles this
                        continue
                elif kind in ['college', 'commercial', 'hospital', 'industrial', 
                              'residential', 'retail', 'school', 'university']:
                    dest, final_kind, final_rank = kind, kind, rank
                elif kind == 'park':
                    dest, final_kind, final_rank = "landuse", kind, rank
                else:
                    if self.debug:
                        # Store it in a debug layer for easier consideration
                        dest, final_kind, final_rank = "debug", kind, rank
                    else:
                        # Not keeping this feature - drop it
                        continue

                props = {'kind': final_kind, 'sort_rank': final_rank}
                if detail: props['kind_detail'] = detail
                if 'ref' in old_props: props['ref'] = old_props['ref']

                if dest not in new_layers_data: new_layers_data[dest] = []
                new_layers_data[dest].append({
                    "geometry": feature['geometry'], 
                    "properties": props,
                    "id": feature.get('id'), 
                    "type": feature.get('type')
                })
        
        # Handle overlapping park features
        if "landuse" in new_layers_data:
            park_geoms = []
            kept_landuse_feats = []
            
            # We will save the first park's properties to act as a template for the dissolved geometries.
            # (Note: Unique IDs or specific 'ref' tags will be lost during this merge).
            base_park_properties = {"kind": "park", "sort_rank": 189} 

            for feat in new_layers_data["landuse"]:
                if feat["properties"].get("kind") == "park":
                    geom = shape(feat["geometry"])
                    if not geom.is_empty:
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                        park_geoms.append(geom)
                    
                    # Grab a copy of the properties to re-apply later
                    base_park_properties = feat["properties"]
                else:
                    kept_landuse_feats.append(feat)

            if park_geoms:
                # Union all park geometries to dissolve overlapping interior boundaries
                # Slight buffer to remove tiny slivers in the middle of polygons
                swelled_parks = [g.buffer(4) for g in park_geoms]
                merged_parks = unary_union(swelled_parks)
                merged_parks = merged_parks.buffer(-4)
                #merged_parks = unary_union(park_geoms)
                
                # Fix potential self-intersections after the union
                merged_parks = shapely.make_valid(merged_parks)
                
                # Explode the result into individual Polygon features
                final_park_parts = []
                if isinstance(merged_parks, Polygon):
                    final_park_parts.append(merged_parks)
                elif isinstance(merged_parks, MultiPolygon):
                    final_park_parts.extend(list(merged_parks.geoms))
                elif hasattr(merged_parks, 'geoms'):
                    # Fallback for GeometryCollections
                    for g in merged_parks.geoms:
                        if isinstance(g, Polygon):
                            final_park_parts.append(g)
                        elif isinstance(g, MultiPolygon):
                            final_park_parts.extend(list(g.geoms))

                # Reconstruct the features
                for part in final_park_parts:
                    if part.is_empty or part.area < 0.01:
                        continue
                    
                    # Ensure exterior ring is CCW
                    part = orient(part, sign=1.0)
                    
                    kept_landuse_feats.append({
                        "geometry": mapping(part),
                        "properties": base_park_properties,
                        "type": "Polygon"
                        # Drop "id"
                    })
            
            # Replace the layer data with the clean subset
            new_layers_data["landuse"] = kept_landuse_feats
        
        # Handle water features
        if water_geoms_to_dissolve:
            snapped_geoms = [set_precision(g, grid_size=0.1) \
                             for g in water_geoms_to_dissolve]
            
            # Union with a tiny "fusion" buffer
            merged_result = unary_union([g.buffer(0.5) for g in snapped_geoms])
            merged_result = merged_result.buffer(-0.5) # Shrink back
            
            # Snap to integer grid
            merged_result = set_precision(merged_result, grid_size=1.0)
            
            # Fix self-intersections caused by grid snapping
            if not merged_result.is_valid:
                merged_result = merged_result.buffer(0)
            merged_result = merged_result.intersection(tile_bounds)
            
            # Explode MultiPolygons into individual Polygon features
            final_parts = []
            if isinstance(merged_result, Polygon):
                final_parts.append(merged_result)
            elif isinstance(merged_result, MultiPolygon):
                final_parts.extend(list(merged_result.geoms))
            elif hasattr(merged_result, 'geoms'):
                for g in merged_result.geoms:
                    if isinstance(g, Polygon):
                        final_parts.append(g)
                    elif isinstance(g, MultiPolygon):
                        final_parts.extend(list(g.geoms))
            
            if "water" not in new_layers_data:
                new_layers_data["water"] = []
            
            for part in final_parts:
                if part.is_empty or part.area < 0.01:
                    continue
                # Ensure exterior is CCW/CW as per spec
                part = orient(part, sign=1.0)
                # Find which original IDs belong to this new dissolved 'part'
                # We use a small negative buffer (ebbing) to ensure the 
                # centroid or intersection is truly inside the new part.
                associated_ids = []
                for orig_id, orig_geom in water_id_map:
                    if part.intersects(orig_geom):
                        associated_ids.append(orig_id)
                
                # Determine the primary ID (using the first one found)
                primary_id = associated_ids[0] if associated_ids else None
                
                water_feat = {
                    "geometry": mapping(part),
                    "properties": {"kind": "water", "sort_rank": 200},
                    "type": "Polygon"
                }
                if primary_id is not None:
                    water_feat["id"] = primary_id
                new_layers_data["water"].append(water_feat)
        else:
            # Set to None if no water exists
            merged_result = None
        
        # Build aerodrome mask
        aerodrome_mask = None
        if "landuse" in new_layers_data:
            aerodrome_geoms = []
            for feat in new_layers_data["landuse"]:
                if feat["properties"].get("kind") == "aerodrome":
                    a_geom = shape(feat["geometry"]).intersection(tile_bounds)
                    if not a_geom.is_empty:
                        if not a_geom.is_valid:
                            a_geom = a_geom.buffer(0)
                        aerodrome_geoms.append(a_geom)
            if aerodrome_geoms:
                aerodrome_mask = unary_union(aerodrome_geoms)
        
        # ── Clip landuse against the dissolved water mask ───────────
        # OSM `landuse=park` (and the other green/recreational classes
        # that collapse to kind='park' in _get_kind_and_rank) frequently
        # extends over rivers, harbours, caldera lakes, etc. Without
        # subtracting water, those polygons bleed over the rendered
        # water layer at intermediate zooms.
        #
        # Per-tile rather than per-zoom-band: the merged water geometry
        # is already at this tile's resolution (via set_precision +
        # grid snap above), so subtracting from same-tile landuse is
        # geometrically consistent by construction.
        if "landuse" in new_layers_data:
            kept = []
            for feat in new_layers_data["landuse"]:
                kind = feat["properties"].get("kind")
                
                if kind not in ["park", "aerodrome"]:
                    kept.append(feat)
                    continue
                
                geom = shape(feat["geometry"])
                geom = geom.intersection(tile_bounds)
                if geom.is_empty:
                    continue
                if not geom.is_valid:
                    geom = geom.buffer(0)
                
                # Subtract water from parks and aerodromes
                if merged_result is not None and not merged_result.is_empty:
                    if geom.intersects(merged_result):
                        geom = geom.difference(merged_result)
                        if not geom.is_valid:
                            geom = geom.buffer(0)

                # Subtract aerodromes from parks
                if kind == "park" and aerodrome_mask is not None and not aerodrome_mask.is_empty:
                    if geom.intersects(aerodrome_mask):
                        geom = geom.difference(aerodrome_mask)
                        if not geom.is_valid:
                            geom = geom.buffer(0)

                # Final geometry verification & cleaning
                if geom.is_empty or geom.area < 1.0:
                    continue  

                if geom.geom_type not in ("Polygon", "MultiPolygon"):
                    parts = [g for g in geom.geoms 
                             if g.geom_type in ("Polygon", "MultiPolygon")]
                    if not parts:
                        continue
                    geom = unary_union(parts) if len(parts) > 1 else parts[0]
                feat["geometry"] = mapping(geom)
                kept.append(feat)
            new_layers_data["landuse"] = kept
        
        layers_to_encode = []
        for name, feats in new_layers_data.items():
            if not feats: continue
            feats.sort(key=lambda f: f['properties'].get('sort_rank', 0))
            layers_to_encode.append({"name": name, 
                                     "features": feats, 
                                     "extent": 4096, 
                                     "version": 2})

        return (
            z, x, y, zlib.compress(mapbox_vector_tile.encode(layers_to_encode))
        )

    def fix_mbtiles(self):
        """
        Translates 'clean' mbtiles to 'fixed' mbtiles with proper schema 
        and hierarchy.
        """
        path_prefix = os.path.join(self.city_dir, self.city.lower())
        input_path = f"{path_prefix}-clean.mbtiles"
        output_path = f"{path_prefix}-fixed.mbtiles"
        
        if os.path.exists(output_path):
            os.remove(output_path)
        
        if self.verb:
            print(f"***** Fixing MBTiles for {self.city} *****")
        conn = sqlite3.connect(input_path)
        cursor = conn.cursor()
        cursor.execute("SELECT zoom_level, tile_column, tile_row, tile_data " \
                       "FROM tiles")
        all_tiles = cursor.fetchall()
        
        if self.verb:
            print(f"Processing {len(all_tiles)} tiles using {self.ncores} " \
                  f"cores...")
        
        with ProcessPoolExecutor(max_workers=self.ncores) as executor:
            results = list(executor.map(self._process_tile_worker, all_tiles))

        # Setup output database
        out_conn = sqlite3.connect(output_path)
        out_conn.execute("CREATE TABLE metadata (name text, value text)")
        out_conn.execute("CREATE TABLE tiles (zoom_level integer, "\
                                             "tile_column integer, "\
                                             "tile_row integer, "\
                                             "tile_data blob)")
        
        # Copy metadata from input
        cursor.execute("SELECT name, value FROM metadata")
        out_conn.executemany("INSERT INTO metadata VALUES (?, ?)", 
                             cursor.fetchall())
        
        # Insert processed tiles
        out_conn.executemany("INSERT INTO tiles VALUES (?, ?, ?, ?)", results)
        
        # Metadata Sync (class -> kind)
        out_conn.execute("UPDATE metadata SET value = REPLACE(value, "\
                                                            "'class', "\
                                                            "'kind') "\
                         "WHERE name = 'json'")
        out_conn.execute("UPDATE metadata SET value = REPLACE(value, "\
                                                            "'subclass', "\
                                                            "'kind') "\
                         "WHERE name = 'json'")
        
        out_conn.commit()
        out_conn.close()
        conn.close()
        if self.verb:
            print(f"Successfully created fixed MBTiles at {output_path}")
    
    def _generate_building_tiles(self):
        """
        Processes building GeoJSON into zoom-specific MBTiles using 
        mapshaper and tippecanoe.
        """
        if self.verb:
            print("***** Generating Building Overlays *****")
        
        # Paths for intermediate files
        self.buildings_mbtiles = os.path.join(self.city_dir, "buildings.mbtiles")
        if self.buildings_geojson is None:
            self.buildings_geojson = os.path.join(self.city_dir, "buildings.geojson")
        self.buildings_zoom_geojson = os.path.join(self.city_dir, "buildings_zoom.geojson")
        
        mapshaper_cmd = (
            f"node --max-old-space-size={self.RAM} $(which mapshaper) "
            f"{self.buildings_geojson} -proj {self.epsg} -snap 0.5 "
            f"-filter 'this.area > {self.building_index_filter_size}' -clean "
            f"-simplify dp interval={self.building_tile_simplification} "
            f"-proj wgs84 -o precision=0.00001 {self.buildings_zoom_geojson}"
        )
        self._run_command(mapshaper_cmd)
        
        # Remove any features with no geometry
        with open(self.buildings_zoom_geojson, 'r') as f:
            geojson_data = json.load(f)
        geojson_data['features'] = [f for f in geojson_data['features'] \
                                    if 'geometry' in f.keys() and f['geometry'] is not None]
        # Save the modified data
        with open(self.buildings_zoom_geojson, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2)
        
        # Add default building height where needed
        self._set_default_building_height()
        
        # Convert to Vector Tiles with Tippecanoe
        if self.max_building_tile_size is not None:
            building_tile_params = ["--drop-smallest-as-needed",
                f"--maximum-tile-bytes={self.max_building_tile_size}"]
        else:
            building_tile_params = ["--no-tile-size-limit"]
        tippe_cmd = [
            "tippecanoe", "-o", self.buildings_mbtiles,
            "--layer=buildings", "--include=height"] + building_tile_params + \
           ["-Z12", f"-z{self.maxzoom}", self.buildings_zoom_geojson, "--force"
        ]
        self._run_command(tippe_cmd)
        
        # Make buildings foundations file
        self._create_building_foundation_files()
    
    def _set_default_building_height(self, default_height=4):
        """
        Sets default building height for buildings geojson file
        """
        # Load the data
        with open(self.buildings_zoom_geojson, 'r') as f:
            data = json.load(f)

        # Add the field if missing
        for feature in data.get('features', []):
            # Ensure properties object exists
            if 'properties' not in feature:
                feature['properties'] = {}
            
            props = feature['properties']
            val = props.get('height')

            # Force the key to exist and be a float
            if val is None or val == "" or float(val) <= 0:
                props['height'] = float(default_height)
            else:
                try:
                    props['height'] = float(val)
                except (ValueError, TypeError):
                    props['height'] = float(default_height)
        
        # Overwrite it it
        with open(self.buildings_zoom_geojson, 'w') as f:
            json.dump(data, f)
    
    def _create_building_foundation_files(self, alpha=0.25, default_height=4.0):
        """
        Calculates the building foundation depths and stores as mbtiles.
        
        Inputs
        ------
        alpha: float. Soil stiffness parameter.  Normal values are ~0.05 - 0.25
        """
        self.buildings_foundations_geojson = os.path.join(self.city_dir, "buildings_foundations.geojson")
        self.buildings_foundations_mbtiles = os.path.join(self.city_dir, "buildings_foundations.mbtiles")
        
        # Load the buildings used for the tiles
        with open(self.buildings_zoom_geojson, 'r') as f:
            geojson_data = json.load(f)
        
        # Iterate through each feature in the GeoJSON
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon' and 'properties' in feature:
                properties = feature['properties']
                
                # Remove the 'name' field if it exists
                properties.pop('name', None)
                
                # Handle height with a default if None
                height = properties.get('height')
                if height is None:
                    height = default_height
                    
                # Reconstruct the polygon using Shapely to find dimensions
                coords = feature['geometry']['coordinates'][0]
                properties['foundationDepth'] = self._calculate_building_foundation(coords, height, alpha)

        # Save the modified data to a new GeoJSON file
        with open(self.buildings_foundations_geojson, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2)
        
        if self.max_building_tile_size is not None:
            building_tile_params = ["--drop-smallest-as-needed",
                f"--maximum-tile-bytes={self.max_building_tile_size}"]
        else:
            building_tile_params = ["--no-tile-size-limit"]
        tippe_cmd = [
            "tippecanoe", "-o", self.buildings_foundations_mbtiles,
            "--layer=foundations", "--include=foundationDepth"] + building_tile_params +\
           ["-Z12", f"-z{self.maxzoom}", self.buildings_foundations_geojson, "--force"
        ]
        self._run_command(tippe_cmd)
        
        if self.verb:
            print("Building foundations files created")
    
    def _calculate_building_foundation(self, coords, height, alpha=0.25):
        poly = Polygon(coords)
                
        # Get the minimum oriented bounding box and its side lengths
        min_rect = poly.minimum_rotated_rectangle
        rect_coords = list(min_rect.exterior.coords)
        
        side1 = U.haversine(rect_coords[0][0], rect_coords[0][1], 
                            rect_coords[1][0], rect_coords[1][1])
        side2 = U.haversine(rect_coords[1][0], rect_coords[1][1], 
                            rect_coords[2][0], rect_coords[2][1])
        min_width = min(side1, side2)
        
        if min_width > 0:
            # Calculate positive depth from the formula
            calculated_depth = alpha * height * (height / min_width) ** 0.25
            
            # Clamp between 10 and 80
            clamped_depth = max(min(calculated_depth, 80), 10)
            
            foundationDepth = int(clamped_depth)
        else:
            # Fallback if geometry is degenerate/point-like
            foundationDepth = 10
        
        return foundationDepth
    
    def _update_mbtiles_metadata(self, mbtiles_path):
        """
        Sqlite3 metadata update.
        """
        conn = sqlite3.connect(mbtiles_path)
        cur = conn.cursor()
        bounds = ",".join(map(str, self.bbox))
        queries = [
            ("REPLACE INTO metadata (name, value) VALUES (?, ?)", 
                ('name', f'{self.city} Basemap')),
            ("REPLACE INTO metadata (name, value) VALUES (?, ?)", 
                ('type', 'baselayer')),
            ("REPLACE INTO metadata (name, value) VALUES (?, ?)", 
                ('bounds', bounds)),
            ("DELETE FROM metadata WHERE name='generator_options'", ())
        ]
        for q, params in queries:
            try:
                cur.execute(q, params)
            except:
                continue
        conn.commit()
        conn.close()

    def _validate_env(self):
        """
        Checks if all required CLI tools are installed and accessible.
        """
        REQUIRED_BINS = ['node', 'mapshaper', 'osmium', 'java', 'tile-join', 
                         'tippecanoe', 'sqlite3', 'jq', 'pmtiles', 
                         'planetiler.jar']
        missing = []
        for tool in REQUIRED_BINS:
            if tool == 'planetiler.jar':
                self.planetiler_path = shutil.which("planetiler.jar", mode=os.F_OK)
                if not self.planetiler_path:
                    # Maybe it's called 'planetiler' in some setups
                    self.planetiler_path = shutil.which("planetiler", mode=os.F_OK)
                    if not self.planetiler_path:
                        missing.append(tool)
            elif shutil.which(tool) is None:
                missing.append(tool)
        
        if missing:
            raise RuntimeError(
                f"Missing required CLI tools: {', '.join(missing)}. "
                "Please install them and ensure they are in your PATH."
            )
    
    def rename_geojson_property(self, filename, old_key, new_key="roadType"):
        """
        Renames a GeoJSON property key using jq.
        """
        input_path = os.path.join(self.city_dir, filename)
        output_path = f"{input_path}.tmp"

        # jq filter: mapping the old key to the new key and deleting the old one
        jq_filter = f'.features[].properties |= (.{new_key} = .{old_key} '\
                    f'| del(.{old_key}))'

        try:
            with open(output_path, 'w') as out_f:
                subprocess.run(["jq", jq_filter, input_path], stdout=out_f, 
                                check=True)
            
            # Replace original with the modified version
            os.replace(output_path, input_path)
        except subprocess.CalledProcessError as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"jq transformation failed: {e}")
    
    def check_labels(self):
        """Checks city.osm.pbf and reports the types and counts of places."""
        places_osmpbf = os.path.join(self.city_dir, "places.osm.pbf")
        places_geojson = os.path.join(self.city_dir, "places.geojson")
        self._run_command(["osmium", "tags-filter", self.osmpbf, 
                           f"n/place{self.places_suffix}", 
                           "-o", places_osmpbf, "--overwrite"])
        
        self._run_command(["osmium", "export", places_osmpbf, "-o", 
                           places_geojson, "--overwrite"])
        
        self._run_command(f"""grep -o '"place":"[^"]*"' {places_geojson} | cut -d'"' -f4 | sort | uniq -c | sort -rn""")
    
    def add_labels(self):
        """
        Extraction and tiling for labels. 
        Uses self.cities, self.suburbs, and self.neighborhoods.  These are 
        lists of strings representing OSM 'place' values, which are shown at 
        different zoom scales.  Below are some settings that slurry uses for 
        various maps, which might be helpful to see what you want for your map.
        
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
            neighborhoods = ['city', 'borough', 'town', 'suburb', 'village', 
                             'hamlet']
        """
        if self.cities  is None and \
           self.suburbs is None and \
           self.neighborhoods is None:
            if self.verb:
                print("***** add_labels: no labels provided for cities, "
                      "suburbs, or neighborhoods *****")
                print("    A labeled pmtiles file will not be created")
            return
        path_prefix = os.path.join(self.city_dir, self.city)
        no_labels_pmtiles = f"{path_prefix}-nolabels.pmtiles"
        labels_only_pmtiles = f"{path_prefix}-onlylabels.pmtiles"
        final_output = f"{path_prefix}.pmtiles"
        
        # Map the input lists to their respective layer names
        layer_configs = {
            "cities": self.cities,
            "suburbs": self.suburbs,
            "neighborhoods": self.neighborhoods
        }
        
        geojson_paths = {}
        
        for name, tags in layer_configs.items():
            osm_pbf = os.path.join(self.city_dir, f"{name}.osm.pbf")
            geojson = os.path.join(self.city_dir, f"{name}.geojson")
            
            # Extract and Export
            filter_cmd = ["osmium", "tags-filter", self.osmpbf]
            # Build the osmium filter string
            # e.g., "n/place=city n/place=borough"
            filter_cmd.extend([f"n/place{self.places_suffix}={t}" for t in tags])
            filter_cmd.extend(["-o", str(osm_pbf), "--overwrite"])
            self._run_command(filter_cmd)
            self._run_command(["osmium", "export", str(osm_pbf), "-o", 
                               str(geojson), "--overwrite"])
            self._rewrite_label_geojson_names(geojson)
            
            # Combine with additional labels, if provided
            if name == "cities" and self.cities_additional:
                self._combine_geojson_labels(geojson, self.cities_additional)
            elif name == "suburbs" and self.suburbs_additional:
                self._combine_geojson_labels(geojson, self.suburbs_additional)
            elif name == "neighborhoods" and self.neighborhoods_additional:
                self._combine_geojson_labels(geojson, self.neighborhoods_additional)
            
            geojson_paths[name] = str(geojson)
            
            if self.cleanup_files:
                os.remove(osm_pbf)

        # Build Tippecanoe command
        bbox_clean = ",".join(map(str, self.bbox))
        tippe_cmd = [
            "tippecanoe", "-Z", "6", "-z", f"{self.maxzoom}", "-r", "1", "-y", "name",
            "-o", labels_only_pmtiles, "--no-tile-size-limit",
            f"--clip-bounding-box={bbox_clean}",
            "--force"
        ]
        if self.cities is not None:
            tippe_cmd.extend(["-L", f"city_labels:{geojson_paths['cities']}"])
        if self.suburbs is not None:
            tippe_cmd.extend([
                "-L", f"suburb_labels:{geojson_paths['suburbs']}"
            ])
        if self.neighborhoods is not None:
            tippe_cmd.extend([
                "-L", f"neighborhood_labels:{geojson_paths['neighborhoods']}"
            ])
        self._run_command(tippe_cmd)

        # Merge and update metadata
        final_mbtiles = final_output.replace('.pmtiles', '.mbtiles')
        self._run_command(["tile-join", "-o", 
                           final_mbtiles, 
                           no_labels_pmtiles, labels_only_pmtiles,
                           "--no-tile-size-limit", "--force"])
        self._update_mbtiles_metadata(final_mbtiles)
        self._run_command(["pmtiles", "convert", final_mbtiles, 
                           final_output])
        
        if self.cleanup_files:
            os.remove(geojson_paths['cities'])
            os.remove(geojson_paths['suburbs'])
            os.remove(geojson_paths['neighborhoods'])
            os.remove(labels_only_pmtiles)
            os.remove(final_mbtiles)

        if self.verb:
            print(f"***** Done. Final pmtiles created at: *****")
            print(f"    {final_output}")
    
    def _combine_geojson_labels(self, main_labels, addtl_labels):
        """Merge user labels into OSM labels"""
        files = [main_labels, addtl_labels]
        combined_features = []

        # Load features from each file
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                # GeoJSON files are typically 'FeatureCollection' types
                combined_features.extend(data['features'])

        combined_geojson = {
            "type": "FeatureCollection",
            "features": combined_features
        }

        # Overwrite the main_labels
        with open(main_labels, 'w') as f:
            json.dump(combined_geojson, f)
    
    def _validate_places(self, name, val):
        """Ensures cities/suburbs/neighborhoods are valid entries."""
        if val is None:
            return None
        if not isinstance(val, list):
            raise TypeError(f"{name} must be a list of strings or None.\n"
                            f"Received: {type(val).__name__}")
        if len(val) == 0:
            raise ValueError(f"{name} cannot be an empty list. "
                             f"Use None to disable this category.")
        # Check for mixed types within the list
        if not all(isinstance(item, str) for item in val):
            raise TypeError(f"All items in the {name} list must be strings.")
        return val
    
    def _validate_additional_places(self, name, val):
        """Ensures additional cities/suburbs/neighborhoods are valid entries."""
        if val is None:
            return None
        if not os.path.exists(val):
            raise FileNotFoundError(f"Specified {name} file not found:\n"+val)
        return val

    def _rewrite_label_geojson_names(self, geojson_path):
        """Normalizes feature properties.name based on label_name_language."""
        if self._label_name_mode == "default":
            return

        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for feature in data.get('features', []):
            properties = feature.get('properties')
            if not isinstance(properties, dict):
                continue
            feature['properties']['name'] = self._select_label_name(properties)

        with open(geojson_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def _select_label_name(self, properties):
        """Returns the label text to store in properties.name."""
        default_name = properties.get('name', '')
        lang_name = properties.get(f"name:{self._label_name_suffix}", '')

        if self._label_name_mode == "prefer":
            return lang_name if lang_name not in [None, ""] else default_name
        if self._label_name_mode == "force":
            return lang_name if lang_name not in [None, ""] else ""
        return default_name

    def _get_road_name_jq_expression(self):
        """Returns the jq expression used to populate roads.geojson name."""
        if self.road_name_preferred_language is None:
            return '.properties.name // ""'
        return (f'.properties["name:{self.road_name_preferred_language}"] '
                '// .properties.name // ""')

    ##### Properties (city, bbox, osmpbf, outputdir) #####
    
    @property
    def city(self):
        return self._city

    @city.setter
    def city(self, value):
        if not isinstance(value, str):
            raise TypeError("City code must be a string.")

        # Check length (2-4 characters)
        if not (2 <= len(value) <= 4):
            raise ValueError(f"City code '{value}' must be 2-4 characters "
                             f"long.")

        # Check that the first two characters are letters
        if not value[:2].isalpha():
            raise ValueError(f"First two characters of '{value}' must be "
                             f"letters.")

        # Check that the remaining characters (if any) are alphanumeric
        if len(value) > 2 and not value[2:].isalnum():
            raise ValueError(f"Characters 3-4 of '{value}' must be letters "
                             f"or numbers.")

        self._city = value.upper()
    
    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError("bbox must be a list, tuple, or numpy array.")

        if len(value) != 4:
            raise ValueError(f"bbox must have exactly 4 values, got "
                             f"{len(value)}.")
        
        # Strict Type Check: Reject strings even if they look like numbers
        if not all(isinstance(x, (int, float)) for x in value):
            # Find the culprit for a better error message
            offenders = [
                type(x).__name__ for x in value 
                if not isinstance(x, (int, float))
            ]
            raise TypeError(f"bbox values must be int or float. "
                            f"Received types: {offenders}")
        
        # Now convert to floats for internal consistency
        clean_bbox = [float(x) for x in value]

        # Logical validation: [min_lon, min_lat, max_lon, max_lat]
        # Value 0 < Value 2 (Longitudes)
        if not clean_bbox[0] < clean_bbox[2]:
            raise ValueError(
                f"Invalid Longitude range: "
                f"minimum longitude ({clean_bbox[0]}) "
                f"must be less than maximum longitude ({clean_bbox[2]})."
            )

        # Value 1 < Value 3 (Latitudes)
        if not clean_bbox[1] < clean_bbox[3]:
            raise ValueError(
                f"Invalid Latitude range: minimum latitude ({clean_bbox[1]}) "
                f"must be less than maximum latitude ({clean_bbox[3]})."
            )

        self._bbox = clean_bbox
        self.get_utm_epsg()
    
    @property
    def osmpbf(self):
        return self._osmpbf

    @osmpbf.setter
    def osmpbf(self, value):
        if value is None:
            self._osmpbf = None
            return

        if isinstance(value, str):
            if not os.path.exists(value):
                raise ValueError(f"The path provided for osmpbf does not exist: "
                                 f"{value}")
            if not value.lower().endswith('.osm.pbf'):
                raise ValueError("The osmpbf file must have a .osm.pbf extension."
                                f"\nReceived: {value}")
        elif isinstance(value, list):
            for v in value:
                if not isinstance(v, str):
                    raise TypeError("When osmpbf is a list, it must be a list of strings."
                                   f"\nReceived element of type {type(v)}")
                if not os.path.exists(v):
                    raise ValueError(f"The path provided for osmpbf does not exist: "
                                     f"{v}")
                if not v.lower().endswith('.osm.pbf'):
                    raise ValueError("The osmpbf file must have a .osm.pbf extension."
                                    f"\nReceived: {v}")
        else:
            # Not a string or list of strings
            raise TypeError("osmpbf must be a string, list of strings, or None."
                           f"\nReceived type {type(value)}")

        self._osmpbf = value
    
    @property
    def building_index_filter_size(self):
        return self._building_index_filter_size
        
    @building_index_filter_size.setter
    def building_index_filter_size(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"building_index_filter_size must be numeric, not {type(value).__name__}")
        elif value < 0:
            raise ValueError(f"building_index_filter_size must be >= 0.\nReceived {value}")
        self._building_index_filter_size = value
    
    @property
    def building_tile_filter_size(self):
        return self._building_tile_filter_size
    
    @building_tile_filter_size.setter
    def building_tile_filter_size(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"building_tile_filter_size must be numeric, not {type(value).__name__}")
        elif value < 0:
            raise ValueError(f"building_tile_filter_size must be >= 0.\nReceived {value}")
        self._building_tile_filter_size = value
    
    @property
    def building_index_simplification(self):
        return self._building_index_simplification
    
    @building_index_simplification.setter
    def building_index_simplification(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"building_index_simplification must be numeric, not {type(value).__name__}")
        elif value < 0:
            raise ValueError(f"building_index_simplification must be >= 0.\nReceived {value}")
        self._building_index_simplification = value
    
    @property
    def building_tile_simplification(self):
        return self._building_tile_simplification
    
    @building_tile_simplification.setter
    def building_tile_simplification(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"building_tile_simplification must be numeric, not {type(value).__name__}")
        elif value < 0:
            raise ValueError(f"building_tile_simplification must be >= 0.\nReceived {value}")
        self._building_tile_simplification = value
    
    @property
    def outputdir(self):
        return self._outputdir

    @outputdir.setter
    def outputdir(self, value):
        if not isinstance(value, str):
            raise TypeError("outputdir must be a string.")
        
        # Default empty string to current directory for cleaner path joining
        target_path = value if value != '' else '.'
        
        if not os.path.isdir(target_path):
            raise ValueError(f"outputdir must be a valid directory: "
                             f"{target_path}")
            
        self._outputdir = target_path
        
        # Set and create city_dir
        self.city_dir = os.path.join(self._outputdir, self.city)
        os.makedirs(self.city_dir, exist_ok=True)
    
    @property
    def ncores(self):
        """Getter for ncores."""
        return self._ncores

    @ncores.setter
    def ncores(self, value):
        """Setter for ncores with validation and capping logic."""
        if value is not None:
            if not isinstance(value, int):
                raise TypeError(f"ncores must be an integer. "
                                f"Received: {type(value).__name__}")
            
            if value <= 0:
                raise ValueError(f"ncores must be a positive integer. "
                                 f"Received: {value}")
            
            cpu_total = os.cpu_count() or 1 # Fallback to 1 if cpu_count=None
            if value > cpu_total:
                if self.verb:
                    print(f"Core count exceeded: reducing ncores to "
                          f"{cpu_total}")
                value = cpu_total
        
        self._ncores = value
    
    @property
    def RAM(self):
        """Getter for RAM (returns value in MB)."""
        return self._RAM

    @RAM.setter
    def RAM(self, value):
        """
        Setter for RAM. 
        Expects GB (int or float) and stores as MB (int).
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"RAM must be an int or float. "
                            f"Received: {type(value).__name__}")
        
        if value < 1:
            raise ValueError(f"RAM limit must be at least 1 GB. "
                             f"Received: {value}")
        
        # GB uses base 10 (1000) - not to be confused with GiB (1024)
        # We store as MB for internal CLI tool flags
        self._RAM = int(value * 1000)

    @property
    def cities(self):
        return self._cities

    @cities.setter
    def cities(self, value):
        self._cities = self._validate_places("cities", value)

    @property
    def suburbs(self):
        return self._suburbs

    @suburbs.setter
    def suburbs(self, value):
        self._suburbs = self._validate_places("suburbs", value)

    @property
    def neighborhoods(self):
        return self._neighborhoods

    @neighborhoods.setter
    def neighborhoods(self, value):
        self._neighborhoods = self._validate_places("neighborhoods", value)
    
    @property
    def cities_additional(self):
        return self._cities_additional

    @cities_additional.setter
    def cities_additional(self, value):
        self._cities_additional = self._validate_additional_places("cities_additional", value)

    @property
    def suburbs_additional(self):
        return self._suburbs_additional

    @suburbs_additional.setter
    def suburbs_additional(self, value):
        self._suburbs_additional = self._validate_additional_places("suburbs_additional", value)

    @property
    def neighborhoods_additional(self):
        return self._neighborhoods_additional

    @neighborhoods_additional.setter
    def neighborhoods_additional(self, value):
        self._neighborhoods_additional = self._validate_additional_places("neighborhoods_additional", value)

    @property
    def label_name_language(self):
        return self._label_name_language

    @label_name_language.setter
    def label_name_language(self, value):
        if value is None:
            self._label_name_language = None
            self._label_name_mode = "default"
            self._label_name_suffix = None
            return

        if not isinstance(value, str):
            raise TypeError("label_name_language must be a string or None.")

        parts = value.split(":", 1)
        if len(parts) != 2 or parts[0] not in ["prefer", "force"] or parts[1] == "":
            raise ValueError("label_name_language must be None or a string in the form 'prefer:<lang>' or 'force:<lang>'. "
                             f"Received: {value}")

        self._label_name_language = value
        self._label_name_mode = parts[0]
        self._label_name_suffix = parts[1]

    @property
    def road_name_preferred_language(self):
        return self._road_name_preferred_language

    @road_name_preferred_language.setter
    def road_name_preferred_language(self, value):
        if value is None:
            self._road_name_preferred_language = None
            return

        if not isinstance(value, str):
            raise TypeError("road_name_preferred_language must be a string or None.")

        value = value.strip()
        if value == "":
            raise ValueError("road_name_preferred_language cannot be empty.")
        if ":" in value:
            raise ValueError("road_name_preferred_language should be only the OSM language code suffix, not a full key like 'name:en'.")

        self._road_name_preferred_language = value
