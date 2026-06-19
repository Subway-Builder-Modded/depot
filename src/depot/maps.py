import sys, os
import subprocess
import shutil
import requests
import json
import struct
import gzip
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sqlite3
import zlib
import mapbox_vector_tile
import duckdb
import geopandas as gpd
import pandas as pd
from shapely import wkb, set_precision
from shapely.ops import unary_union, orient
from shapely.geometry import shape, mapping, box, Polygon, MultiPolygon


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
    _fetch_overture_buildings : Queries Overture Maps S3 bucket using DuckDB 
                                and saves to GeoJSON.
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
    _get_kind_and_rank : Helper to map OSM/Planetiler tags to game-engine 
                         specific kinds and ranks.
    _process_tile_worker : Worker function to handle vector tile re-mapping.
    fix_mbtiles : Translates 'clean' mbtiles to 'fixed' mbtiles with proper 
                  schema and hierarchy.
    _generate_building_tiles : Processes building GeoJSON into zoom-specific 
                               MBTiles using mapshaper and tippecanoe.
    _update_mbtiles_metadata : Sqlite3 metadata update.
    _validate_env : Checks if all required CLI tools are installed and 
                    accessible.
    rename_geojson_property : Renames a GeoJSON property key using jq.
    check_labels : Checks city.osm.pbf and reports the types and counts of places.
    add_labels : Extraction and tiling for labels. 
    _validate_places : Ensures cities/suburbs/neighborhoods are valid entries.
    _validate_additional_places : Ensures additional cities/suburbs/
                                  neighborhoods are valid entries.
    """
    def __init__(self, city, bbox, osmpbf=None, outputdir='.', 
                       building_index_filter_size=40, 
                       building_tile_filter_size=None, 
                       building_index_simplification=1,
                       building_tile_simplification=1,
                       max_building_tile_size=450,
                       cities=None, suburbs=None, neighborhoods=None,
                       cities_additional=None, suburbs_additional=None, 
                       neighborhoods_additional=None, 
                       places_suffix="", label_name_language=None,
                       road_name_preferred_language=None,
                       buildings_geojson=None, redownload_buildings=False, 
                       color_military_like_aerodrome=True,
                       ncores=1, RAM=4, cleanup_files=True, verb=True):
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
                                   for collisions and for pmtiles at zooms 
                                   14-15.
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
        max_building_tile_size: int. Maximum size per tile in KB when 
                                considering only buildings. The absolute 
                                maximum per tile is 500, which includes 
                                buildings, rivers, roads, and more.
                                Default: 450
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
        color_military_like_aerodrome: bool. If True, military bases are 
                                       colored on the map the same as airports.
                                       If False, it looks like any other 
                                       ordinary tile.
                                       Default: True
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
        """
        self.verb = bool(verb)
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
        self.color_military_like_aerodrome = bool(color_military_like_aerodrome)
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
        if max_building_tile_size > 500 or max_building_tile_size < 100:
            raise ValueError("`max_building_tile_size` must be >=100 and "
                            f"<=500.\nReceived: {max_building_tile_size}")
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
        
        if self.verb:
            print("***** MapGen initialized *****")
            print("------------------------------")
            print(f"city                : {self.city}")
            print(f"bbox                : {self.bbox}")
            print(f"osmpbf source files : {self.osmpbf_sources}")
            print(f"redownload_buildings: {self.REFETCH_BUILDINGS}")
            print(f"building_index_filter_size: {self.building_index_filter_size} m2")
            print(f"building_tile_filter_size : {self.building_tile_filter_size} m2")
            print(f"ncores              : {self.ncores}")
            print(f"RAM                 : {self.RAM} MB")
            print(f"cleanup_files       : {self.cleanup_files}")
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
            "tags=natural=water,landuse=reservoir,waterway=riverbank,"
            "highway=residential", 
            "--bbox", bbox_str, self.osmpbf, "-o", 
            out_pbf, "--overwrite"
        ]
        self._run_command(cmd)
        
        # Filter out buildings
        nobuilding_pbf = os.path.join(self.city_dir, f"{self.city.lower()}-nobuildings.osm.pbf")
        self._run_command([
            "osmium", "tags-filter", out_pbf, 
            "n/building=yes", "w/building=yes", 
            "-o", nobuilding_pbf, "--overwrite"
        ])
        
        self.nobuildings_geojson = nobuilding_pbf.replace('.osm.pbf', '.geojson')
        self._run_command([
            "ogr2ogr", "-f", "GeoJSONSeq", 
            self.nobuildings_geojson, nobuilding_pbf
        ])
        
    def _convert_to_game_format(self, input_path):
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
            
            for poly_coord in polys_coords:
                cleaned_p = []
                b_minx, b_miny = float('inf'), float('inf')
                b_maxx, b_maxy = float('-inf'), float('-inf')

                # Determine foundation depth
                foundation = props.get('f', props.get('depth', props.get('building:levels:underground', 1)))
                try:
                    foundation = int(foundation)
                    if foundation > max_found_depth:
                        max_found_depth = foundation
                except:
                    foundation = 1

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

    def create_buildings_index_binary(self, input_path):
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

            try:
                # Parse geometry via Shapely
                geom_shape = shape(geom)

                # Handle MultiPolygons
                if isinstance(geom_shape, MultiPolygon):
                    # Fallback Strategy: Extract the largest constituent polygon component by area
                    geom_shape = max(geom_shape.geoms, key=lambda p: p.area)

                if not isinstance(geom_shape, Polygon):
                    continue

                # Reconstruct coordinates from the chosen/resolved single Polygon
                # Exterior ring is first, followed by internal holes (interiors)
                polygon_coords = [list(geom_shape.exterior.coords)] + [
                    list(hole.coords) for hole in geom_shape.interiors
                ]

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
                foundation = props.get(
                    "f",
                    props.get("depth", props.get("building:levels:underground", 1)),
                )
                try:
                    foundation = max(1, round(float(foundation)))
                except (ValueError, TypeError):
                    foundation = 1

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
                print(
                    f"Warning: Error processing building at index {index}: {error}"
                )

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

        # Calculate Buffer Allocations
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

        # Write Header Metadata
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

        # 1. Bounds Entries
        off = bounds_offset
        for b in buildings:
            struct.pack_into("<4d", buffer, off, *b["bounds"])
            off += 32

        # 2. Foundation Depths
        off = foundation_depths_offset
        for b in buildings:
            _FLOAT32.pack_into(buffer, off, b["foundationDepth"])
            off += 4

        # 3. Geometry Lookups
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

        # 4. CSR Spatial Index Sequences
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

        # Stream Outputs
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
        # Update this string when Overture releases a new version
        OVERTURE_RELEASE = self._get_latest_overture_release()
        
        buildings_pkl = os.path.join(self.city_dir, "buildings.pkl")
        self.buildings_geojson = os.path.join(self.city_dir, 
                                              "buildings.geojson")

        # Check if we already have the data to avoid expensive re-downloads
        if not os.path.exists(buildings_pkl) or self.REFETCH_BUILDINGS:
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
        response = requests.get("https://stac.overturemaps.org/catalog.json")
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
        
    def process_roads_and_aeroways(self):
        """
        Extracts roads and aeroways, applies JQ filters and buffering.
        """
        if self.verb:
            print("***** Processing Roads and Aeroways *****")
        city_pbf = os.path.join(self.city_dir, f"{self.city.lower()}.osm.pbf")
        roads_pbf = os.path.join(self.city_dir, "roads.pbf")
        roads_geojson = os.path.join(self.city_dir, "roads.geojson")
        
        # 1. Roads
        roads_list = "motorway,motorway_link,trunk,trunk_link,primary,"\
                     "primary_link,secondary,secondary_link,tertiary,"\
                     "tertiary_link,unclassified,residential"
        self._run_command(["osmium", "tags-filter", city_pbf, 
                           f"w/highway={roads_list}", "-o", roads_pbf, 
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
        raw_mbtiles = f"{path_prefix}.mbtiles"
        clean_mbtiles = f"{path_prefix}-clean.mbtiles"
        fixed_mbtiles = f"{path_prefix}-fixed.mbtiles"
        merged_mbtiles = f"{path_prefix}-merged.mbtiles"
        self.buildings_mbtiles = os.path.join(self.city_dir, "buildings.mbtiles")
        final_pmtiles = os.path.join(self.city_dir, self.city+"-nolabels.pmtiles")

        # 1. Planetiler
        bounds_str = ",".join(map(str, self.bbox))
        self._run_command([
            "java", "-Xmx16g", "-jar", self.planetiler_path,
            f"--osm-path={city_pbf}", 
            f"--output={raw_mbtiles}",
            f"--bounds={bounds_str}",
            "--download", 
            "--minzoom=0", 
            "--maxzoom=15", 
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
            "-o", clean_mbtiles, raw_mbtiles
        ])
        
        if self.cleanup_files:
            os.remove(raw_mbtiles)
        
        # 3. Fix the tiles as SB expects
        self.fix_mbtiles() # Turns clean_mbtiles into fixed_mbtiles
        
        if self.cleanup_files:
            os.remove(clean_mbtiles)
        

        # 4. Building Overlays
        self._generate_building_tiles()

        # 5. Merge buildings
        self._update_mbtiles_metadata(fixed_mbtiles)
        self._update_mbtiles_metadata(self.buildings_mbtiles)
        
        self._run_command([
            "tile-join", "--force", 
            "-o", merged_mbtiles,
            fixed_mbtiles, self.buildings_mbtiles
        ])
        
        if self.cleanup_files:
            os.remove(fixed_mbtiles)
            os.remove(self.buildings_mbtiles)
        
        # 6. Metadata and PMTiles Convert
        self._update_mbtiles_metadata(merged_mbtiles)
        self._run_command(["pmtiles", "convert", merged_mbtiles, 
                           final_pmtiles])
        
        if self.cleanup_files:
            os.remove(merged_mbtiles)
        
    def _apply_jq(self, filepath, filter_str):
        """
        Internal helper for JQ operations.
        """
        tmp_file = filepath + ".tmp"
        with open(tmp_file, 'w') as out_f:
            subprocess.run(["jq", "-c", filter_str, filepath], stdout=out_f, 
                            check=True)
        os.replace(tmp_file, filepath)
    
    def _buffer_linestrings(self, filepath, buffer_width=0.00015):
        """
        Internal helper to convert LineStrings to Polygons (buffer fix).
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        for feat in data['features']:
            if feat['geometry']['type'] == 'LineString':
                geom = shape(feat['geometry'])
                buffered = geom.buffer(buffer_width, cap_style=2)
                feat['geometry'] = mapping(buffered)
                feat['geometry']['type'] = 'Polygon'
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def _get_kind_and_rank(self, val):
        """
        Helper to map OSM/Planetiler tags to game-engine specific kinds and 
        ranks.
        """
        priority = {
            'aeroway': 400, 'river': 200, 'park': 189, 'wood': 180,
            'forest': 180, 'scrub': 180, 'grass': 50, 'aerodrome': 189
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
                                'zoo', 'grass', 'wood']):
            return 'park', None, priority['park']
        if 'aerodrome' in v or \
           ('military' in v and self.color_military_like_aerodrome):
            return 'aerodrome', None, priority['aerodrome']
        if 'scrub' in v:
            return 'scrub', None, priority['scrub']
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

        for layer_name, layer_content in decoded.items():
            is_bldg_layer = 'building' in layer_name.lower()
            for feature in layer_content['features']:
                old_props = feature.get('properties', {})
                kind, detail, rank = self._get_kind_and_rank(
                    old_props.get('aeroway') or old_props.get('class') or ""
                )
                
                water_kinds = ['ocean', 'river', 'canal', 'drain',
                               'swimming_pool', 'lake', 'cenote', 'lagoon',
                               'oxbow', 'rapids', 'stream', 'stream_pool',
                               'canal', 'pond', 'reflecting_pool',
                               'reservoir']

                if kind in water_kinds:
                    dest = "water"
                    final_kind = kind
                    final_rank = rank
                    
                    geom = shape(feature['geometry'])
                    
                    # Turn linestrings into polygons:
                    if 'LineString' in feature['geometry']['type']:
                        target_meters = 10
                        extent = 4096
                        meters_per_tile = 40075016.686 / (2**z)
                        units_per_meter = extent / meters_per_tile
                        target_buffer = (target_meters / 2) * units_per_meter
                        safe_buffer = max(target_buffer, 4.0)
                        geom = geom.buffer(safe_buffer, cap_style=2)
                        if not geom.is_empty:
                            if geom.geom_type == 'Polygon':
                                new_coords = [list(geom.exterior.coords)]
                            else:
                                new_coords = [list(p.exterior.coords) for p in geom.geoms]
                            feature['geometry']['coordinates'] = new_coords
                            feature['geometry']['type'] = 'Polygon'
                    if not geom.is_empty:
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                        water_geoms_to_dissolve.append(geom)
                        water_id_map.append((feature.get('id'), geom))
                    continue # These features will be added after the loop
                elif (kind == 'aeroway' or \
                      'runway' in str(old_props).lower() or \
                      'taxiway' in str(old_props).lower()):
                    dest, final_kind, final_rank = "roads", "aeroway", 400
                    if not detail:
                        detail = (
                            'runway' if 'runway' in str(old_props).lower() 
                            else 'taxiway'
                        )
                elif kind == 'aerodrome':
                    dest, final_kind, final_rank = "landuse", "aerodrome", 189
                elif is_bldg_layer or kind == 'building':
                    dest, final_kind, final_rank = "buildings", "building", 400
                elif layer_name in ["transportation", "roads", "navigation"]:
                    dest, final_kind, final_rank = "roads", kind, rank
                else:
                    dest, final_kind, final_rank = "landuse", kind, rank

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
            if "landuse" in new_layers_data and merged_result is not None \
                    and not merged_result.is_empty:
                kept = []
                for feat in new_layers_data["landuse"]:
                    if feat["properties"].get("kind") != "park":
                        kept.append(feat)
                        continue
                    geom = shape(feat["geometry"])
                    if not geom.intersects(merged_result):
                        kept.append(feat)
                        continue
                    clipped = geom.difference(merged_result)
                    if not clipped.is_valid:
                        clipped = clipped.buffer(0)
                    if clipped.is_empty or clipped.area < 1.0:
                        continue  # whole feature was over water — drop
                    # difference() can yield GeometryCollections; keep only
                    # Polygon/MultiPolygon parts (matches the spec the rest
                    # of this worker enforces).
                    if clipped.geom_type not in ("Polygon", "MultiPolygon"):
                        parts = [
                            g for g in clipped.geoms
                            if g.geom_type in ("Polygon", "MultiPolygon")
                        ]
                        if not parts:
                            continue
                        clipped = unary_union(parts) if len(parts) > 1 \
                                  else parts[0]
                    feat["geometry"] = mapping(clipped)
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

        if not layers_to_encode:
            return (z, x, y, data)
        
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
            results = list(executor.map(self._process_tile_worker, 
                                        all_tiles))

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
        
        # Add default building height where needed
        self._set_default_building_height()
        
        # Convert to Vector Tiles with Tippecanoe
        tippe_cmd = [
            "tippecanoe", "-o", self.buildings_mbtiles,
            "--layer=buildings", "--include=height", "--drop-smallest-as-needed",
            f"--maximum-tile-bytes={self.max_building_tile_size}", 
            "-Z12", "-z15", self.buildings_zoom_geojson, "--force"
        ]
        self._run_command(tippe_cmd)
    
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
            if val is None or val == "":
                props['height'] = float(default_height)
            else:
                try:
                    props['height'] = float(val)
                except (ValueError, TypeError):
                    props['height'] = float(default_height)
        
        # Overwrite it it
        with open(self.buildings_zoom_geojson, 'w') as f:
            json.dump(data, f)
    
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
            "tippecanoe", "-Z", "6", "-z", "15", "-r", "1", "-y", "name",
            "-o", labels_only_pmtiles,
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
                           '--force'])
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
