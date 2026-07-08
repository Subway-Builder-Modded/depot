"""
All things related to demand data.

Classes
-------
DemandData: Handles demand data.

Functions
---------
process_home_node: Calculate routing when using OSMnx.
compute_centroid: Calculate a (possibly weighted) centroid.
weighted_mean: Calculate a weighted mean.
haversine: Calculate the Haversine distance between two points
haversine_travel_time: Calculate the Haversine distance and travel time given 
                       a speed in kph.
in_cbd: Determine if a point is located within the central business district.
merge_points: Efficiently merge points.  Used as a base for functools partial.

"""

import sys, os
from collections import defaultdict
import copy
import shutil
from datetime import datetime, timezone
import functools
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np
import osmnx as ox
import requests
import time
from tqdm import tqdm
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from sklearn.cluster import AgglomerativeClustering
from unidecode import unidecode
import inflect

import depot.utils as U


# Defines
km2m   = 1000 # km -> meter
hr2sec = 3600 # hour -> seconds


class DemandData(dict):
    """
    Handles demand data.
    
    Methods
    -------
    save: Saves demand data.
    load: Loads demand data.
    print_stats: Prints summary statistics about the demand data.
    enforce_max_pop_size: Splits populations that exceed a maximum pop size 
                          into smaller chunks.
    calculate_routes: Calculates driving distances and durations for population
                      paths using either OSMnx (parallelized) or a local OSRM 
                      server.
    scale_demand: Scales raw job demand by a constant factor.
                  Special demand are not impacted.
    consolidate_pops: Merges job demand below threshold pop sizes into larger 
                      pops that have the same home (or work) node and a nearby 
                      work (or home) node.
    merge_identical_commutes: Merges any pops that have the same exact home 
                              and work nodes.
    cluster_points: Merges points together according to size and distance.
                    General version of Colin's point merging code.
    agglomerate_pops: Agglomerates pops below a threshold size into new 
                      super-origin points.
    move_points: Move the point nearest to some coordinate to a new coordinate.
    add_points: Creates a new demand point and assigns pops.
    del_points: Deletes specified point(s) and all pops associated with it.
    _load_schema: Helper method to read data into structures.
    get_exponent: Looks up the decay exponent for a given point of interest 
                  type ID.
    save_schemas: Save out special demand types and points schema files.
    create_config: Creates the config.json file needed for Railyard import.
    create_description: Create a Markdown file with the contents for the map 
                        description during Railyard submission.
    """
    def __init__(self, fdemand, map_code, bbox=None,
                 outputdir=None, HUMAN_READABLE=False, verb=True):
        """
        Inputs
        ------
        fdemand: str. Path to demand file. Can be named anything, but 
                      Subway Builder requires it to be demand_data.json to be  
                      used in game.
        map_code: str. 4 character map code. Any numbers must occur at the end 
                       of the code (e.g., 4LAX is not valid, but LAX4 is).
        outputdir: (optional) str. Output directory to save files. 
                                   If None, defaults to the `map_code` 
                                   directory relative to where you are running 
                                   the code from.
                                   Default: None
        HUMAN_READABLE: (optional) bool. Determines whether to save the demand 
                                         file with line breaks and indentation 
                                         for readability.
                                         Default: False
        verb: (optional) bool. Controls verbosity of output print statements.
                               Default: True
        """
        super().__init__()
        
        self.HUMAN_READABLE = bool(HUMAN_READABLE)
        self.verb = bool(verb)
        
        self.map_code = map_code
        self.bbox = bbox
        self.outputdir = outputdir
        if self.outputdir is None:
            self.outputdir = self.map_code
        
        self.fdemand = fdemand
        self.load()
        if not str(self.fdemand).endswith("demand_data.json"):
            print(f"WARNING: Demand file '{self.fdemand}' is not named " \
                   "demand_data.json.  Note that the file must be named " \
                   "demand_data.json to be recognized by Subway Builder.")
        
        self._load_schema()
        
        # Check if the demand already has any special demand
        # If it does, then the points schema file must already exist
        dest_dir = os.path.join(self.outputdir, ".railyard_map")
        os.makedirs(dest_dir, exist_ok=True)
        self.ftypes_schema  = os.path.join(dest_dir, "special_demand_types.json")
        self.fpoints_schema = os.path.join(dest_dir, "special_demand_points.json")
        self.has_existing_special_demand = False
        for p in self["points"]:
            if p['id'].split('_')[0] in self.special_demand_ids.keys():
                self.has_existing_special_demand = True
                break
        if self.has_existing_special_demand:
            if not os.path.exists(self.fpoints_schema):
                raise FileNotFoundError("Provided demand_data.json file has "
                        f"special demand points, but the points schema file "
                        f"does not exist at {self.fpoints_schema}")
            if not os.path.exists(self.ftypes_schema):
                raise FileNotFoundError("Provided demand_data.json file has "
                        f"special demand points, but the types schema file "
                        f"does not exist at {self.ftypes_schema}")
        self.added_special_demand_points = []

    def save(self, fdemand=None):
        """
        Saves the current dictionary contents to the specified JSON file.
        
        Inputs
        ------
        fdemand: str. Path to demand file. Can be named anything, but 
                      Subway Builder requires it to be demand_data.json to be  
                      used in game.
                      If None, falls back to self.fdemand selected on initialization.
                      Default: None
        """
        fdemand = fdemand or self.fdemand
        if not fdemand:
            raise ValueError("Attempted to save a demand file, but none "
                  "specified in the object's fdemand attribute or the "
                  "function's fdemand input.")
        
        # Ensure consistent points data
        self.update(self.sanitize(self))
        
        # Convert self to a standard dict to ensure clean JSON serialization
        with open(fdemand, "w") as json_file:
            if self.HUMAN_READABLE:
                json.dump(dict(self), json_file, indent=4)
            else:
                json.dump(dict(self), json_file, indent=None, separators=(',', ':'))
        
        self.save_schemas()

    def load(self, fdemand=None):
        """Loads data from the JSON file and updates the dictionary.
        If the file doesn't exist, it leaves the current values intact.
        If the file exists, it loads pops and/or points if the key exists.
        
        Inputs
        ------
        fdemand: str. Path to demand file. Can be named anything, but 
                      Subway Builder requires it to be demand_data.json to be  
                      used in game.
                      If None, falls back to self.fdemand selected on initialization.
                      Default: None
        """
        fdemand = fdemand or self.fdemand
        if not isinstance(fdemand, str):
            raise ValueError("Attempted to load a demand file, but a string "
                             "was not provided.")
        
        if not os.path.exists(fdemand):
            raise FileNotFoundError(f"File '{fdemand}' does not exist.")

        with open(fdemand, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Sanitize the input data
        self.update(self.sanitize(data))
    
    @staticmethod
    def sanitize(data):
        """
        Sanitizes demand data by
        - recalculating points' job and resident counts from the pops list,
        - dropping any points and pops of size 0, and
        - dropping any pops that aren't assigned to a point.
        """
        assert isinstance(data, dict), "`data` input must be a dictionary."\
                                      f"\nReceived: {type(data)}"
        assert 'pops' in data.keys() and 'points' in data.keys(), "`data` "\
                        f"input dict must have keys for 'pops' and 'points'."\
                        f"\nReceived: {data.keys()}"
        # Clear out empty pops and pops that don't have a residence and/or job
        points_by_id = {p['id']: p for p in data['points']}
        point_ids = [p['id'] for p in data['points']]
        data['pops'] = [p for p in data['pops'] if p['size'] > 0 and \
                        p['residenceId'] in point_ids and \
                        p['jobId'] in point_ids]
        
        # Update point sizes and pop IDs to ensure consistency
        for p in data['points']:
            p['popIds'] = []
            p['jobs'] = 0
            p['residents'] = 0
        for p in data['pops']:
            points_by_id[p['residenceId']]['popIds'].append(p['id'])
            points_by_id[p['residenceId']]['residents'] += p['size']
            points_by_id[p['jobId']]['popIds'].append(p['id'])
            points_by_id[p['jobId']]['jobs'] += p['size']
        # Clear out any points that now have no pops
        data['points'] = [p for p in data['points'] if p['jobs'] + p['residents'] > 0]
        
        return data
        
        
    
    def print_stats(self):
        """
        Prints summary statistics about the demand data.
        """
        print("Points:", len(self['points']))
        print("Pops:", len(self['pops']))
        print("Total pop size:", np.sum([p['size'] for p in self['pops']]))
        print("Workers:", np.sum([p['jobs'] for p in self['points']]))
        print("Residents:", np.sum([p['residents'] for p in self['points']]))
        print("Median point size:", int(np.median([p['jobs'] + p['residents'] for p in self['points']])))
        print("Mean point size:", round(float(np.mean([p['jobs'] + p['residents'] for p in self['points']])), 1))
        print("Median pop size:", int(np.median([p['size'] for p in self['pops']])))
        print("Mean pop size:", round(float(np.mean([p['size'] for p in self['pops']])), 1))
        print("Median commute distance (km):", round(float(np.median([p['drivingDistance'] / 1000 for p in self['pops']])), 2))
        print("Mean commute distance (km):", round(float(np.mean([p['drivingDistance'] / 1000 for p in self['pops']])), 2))
        print("Median commute time (min):", round(float(np.median([p['drivingSeconds'] / 60 for p in self['pops']])), 1))
        print("Mean commute time (min):", round(float(np.mean([p['drivingSeconds'] / 60 for p in self['pops']])), 1))
    
    def enforce_max_pop_size(self, MAXPOPSIZE):
        """
        Splits populations that exceed MAXPOPSIZE into smaller chunks.
        
        Inputs
        ------
        MAXPOPSIZE: int. Maximum pop size before needing to be split into 
                         smaller pops.
        """
        if self.verb:
            print(f"Pops before enforcing size <= {MAXPOPSIZE}: {len(self['pops'])}")

        points_by_id = {
            pt["id"]: pt for pt in self.get("points", []) if "id" in pt
        }

        counter = 0

        # Iterating over a shallow copy list(...) prevents bugs from appending
        # items to self['pops'] while looping over it.
        for p in list(self["pops"]):
            p["size"] = int(p["size"])
            if p["size"] > MAXPOPSIZE:
                niter = int(np.ceil(p["size"] / MAXPOPSIZE))

                for n in range(1, niter):
                    pop = copy.deepcopy(p)
                    pop["id"] += "_" + str(counter)
                    counter += 1

                    if n < niter - 1:
                        # More than MAXPOPSIZE pops remain - cap at MAXPOPSIZE
                        pop["size"] = MAXPOPSIZE
                    else:
                        # <= MAXPOPSIZE remains - put all remaining into this pop
                        pop["size"] = p["size"] - (MAXPOPSIZE * n)

                    self["pops"].append(pop)

                    if pop["jobId"] in points_by_id:
                        points_by_id[pop["jobId"]]["popIds"].append(pop["id"])
                    if pop["residenceId"] in points_by_id:
                        points_by_id[pop["residenceId"]]["popIds"].append(pop["id"])

                # Update the original pop
                p["size"] = MAXPOPSIZE
        
        if self.verb:
            print(f"Pops after enforcing size <= {MAXPOPSIZE}: {len(self['pops'])}")
    
    def calculate_routes(self, routing_method="osmnx", bbox=None, 
                               max_workers=1, osrm_port=5000):
        """
        Calculates driving distances and durations for population paths using
        either OSMnx (parallelized) or an already-running local OSRM server.
        
        Inputs
        ------
        routing_method: str. Method to calculate routes.
                             Options: osmnx, osrm
                             OSRM is recommended as it is faster, but it 
                             requires the user to set up manually.
                             OSMnx does not require any user set up, but is 
                             much slower and requires more resources.
                             Default: osmnx
        bbox: list of floats OR dict with "type" and "bounds" (list of floats) 
              or "coordinates" (list of list of floats).
              ONLY required if using 'osmnx' routing_method.
              "type" must be either "box" or "polygon".
              If "type" = "box":
                  "bounds": The [min_lon, min_lat, max_lon, max_lat] boundary 
                  for the map. 
              If "type" = "polygon":
                  "coordinates": [[lon1, lat1], [lon2, lat2], ..., [lonN, latN]] 
                  that defines the polygon boundary for the map.
                  The first and last coordinate pairs must be identical.
        max_workers: int.  Number of parallel route calculations to do when 
                           using osmnx.
                           Note: Higher values increase RAM usage.
                           Default: 1
        osrm_port: int. Port number for local OSRM server.
                        Not used if using OSMnx for routing.
                        Default: 5000
        """
        assert routing_method in ['osmnx', 'osrm'], "Invalid routing_method.  "\
                                                    "Must be 'osmnx' or 'osrm'."
        # Ensure valid bbox if using osmnx
        if routing_method == 'osmnx':
            if bbox is None:
                if self.bbox is not None:
                    bbox = self.bbox
                else:
                    print("Warning: specified osmnx routing, but no bbox "
                          "provided. Routes will not be calculated.")
                return
            if isinstance(bbox, list):
                assert len(bbox) == 4, "If bbox is a list, it must be 4 values of "\
                                       "min_lon, min_lat, max_lon, max_lat"
                bbox = {"type" : "box", "bounds" : bbox}
            else:
                assert isinstance(bbox, dict), "bbox must be a list or dictionary. "\
                                               "Received: "+str(type(bbox))
                assert "type" in bbox.keys() and "bounds" in bbox.keys()
                assert bbox["type"] in ["box", "polygon"], "Invalid bbox type provided.  "\
                                                           "Must be 'box' or 'polygon'."
                if bbox["type"] == "polygon":
                    assert np.all(bbox["coordinates"][0] == bbox["coordinates"][-1]), \
                        "'polygon' bounding box specified, but the first and last "\
                        "coordinates do not match."
        
        # Build reference lookup mappings using internal dictionary keys
        points_by_id = {p["id"]: p for p in self["points"]}
        pops_by_id = {p["id"]: p for p in self["pops"]}

        if routing_method == "osmnx":
            if self.verb:
                print("Initializing OSM drive network graph")
            if bbox and bbox.get("type") == "box":
                G = ox.graph_from_bbox(bbox["bounds"], network_type="drive")
            elif bbox and bbox.get("type") == "polygon":
                G = ox.graph_from_polygon(poly, network_type="drive")
            else:
                raise ValueError("A valid bbox or polygon configuration must "
                                 "be provided for OSMnx routing.")

            G = ox.truncate.largest_component(G, strongly=True)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)

            # Parallelize the route calculations over home nodes
            if self.verb:
                print("Calculating driving paths for each home node. "
                      "This may take a while.")
            
            # process_home_node must be defined globally in your script for Pool to pick it up
            process_home_node_worker = functools.partial(
                process_home_node, demand=dict(self), G=G, 
                points_by_id=points_by_id
            )

            num_points = len(self["points"])
            with Pool(processes=max_workers) as pool:
                results = []
                for r in tqdm(
                    pool.imap(process_home_node_worker, range(num_points)),
                    total=num_points,
                ): results.append(r)

            # Flatten results and update internal populations dictionary
            for ret in results:
                for pop in ret:
                    pops_by_id[pop["id"]]["drivingSeconds"] = pop["drivingSeconds"]
                    pops_by_id[pop["id"]]["drivingDistance"] = pop[
                        "drivingDistance"
                    ]

        elif routing_method == "osrm":
            if self.verb:
                print("Calculating routes using local OSRM server.")
            num_points = len(self["points"])

            for ipoint in range(num_points):
                if self.verb:
                    print(f"  Point {ipoint + 1} / {num_points}", end="\r")
                home_point = self["points"][ipoint]
                home_id = home_point["id"]

                # Get nearest point coordinates from OSRM
                time.sleep(0.002)
                url_nearest_home = f"http://localhost:{osrm_port}/nearest/v1/driving/{home_point['location'][0]},{home_point['location'][1]}"
                response = requests.get(url_nearest_home)

                if response.status_code != 200:
                    if self.verb:
                        print(f"Invalid response for home point {home_id}. Skipping.")
                    continue

                home_node_loc = response.json()["waypoints"][0]["location"]
                pops = [p for p in self["pops"] if p["residenceId"] == home_id]

                for p in pops:
                    if p["drivingSeconds"] > 0:
                        # Already calculated - skip
                        continue
                    job_id = p["jobId"]
                    job_point = points_by_id[job_id]

                    # Get nearest job point coordinates from OSRM
                    time.sleep(0.002)
                    url_nearest_job = f"http://localhost:{osrm_port}/nearest/v1/driving/{job_point['location'][0]},{job_point['location'][1]}"
                    response = requests.get(url_nearest_job)

                    if response.status_code != 200:
                        if self.verb:
                            print(f"Invalid response for home point {home_id} "
                                  f"at job point {job_id}. Skipping.")
                        continue

                    job_node_loc = response.json()["waypoints"][0]["location"]

                    # Request actual route path calculations between home and job nodes
                    time.sleep(0.002)
                    url_route = (
                        f"http://localhost:{osrm_port}/route/v1/driving/"
                        f"{home_node_loc[0]},{home_node_loc[1]};{job_node_loc[0]},{job_node_loc[1]}?overview=false"
                    )
                    response = requests.get(url_route)

                    if response.status_code != 200:
                        if self.verb:
                            print(f"Invalid response for route between home point "
                                  f"{home_id} and job point {job_id}. Skipping.")
                        continue

                    resp = response.json()
                    if resp.get("code") == "NoRoute":
                        if self.verb:
                            print(f"No route between home point {home_id} and "
                                  f"job point {job_id}. Using straight-line "
                                  "distance.")

                        dist, duration = self._haversine_travel_time(
                            home_node_loc[0],
                            home_node_loc[1],
                            job_node_loc[0],
                            job_node_loc[1],
                        )

                        p["drivingDistance"] = int(dist)
                        p["drivingSeconds"] = int(duration)
                    else:
                        p["drivingSeconds"] = int(resp["routes"][0]["duration"])
                        p["drivingDistance"] = int(resp["routes"][0]["distance"])
            print("")
    
    def scale_demand(self, demand_factor=1):
        """
        Scales raw job demand by a constant factor.
        Special demand are not impacted.
            Note: Current implementation only excludes jobs at point IDs that 
                  begin with "UNI_", "ENT_", "AIR_", "MIL_", "FER_", or "BOR_"
        
        Inputs
        ------
        demand_factor: int or float. Scaling factor for all pop demand.
                                     Must be > 0.
        """
        demand_factor = float(demand_factor)
        assert demand_factor > 0, "demand_factor must be greater than 0."
        if demand_factor != 1:
            if self.verb:
                print("Applying a demand scaling factor of", demand_factor)

            points_by_id = {p["id"]: p for p in self["points"]}

            for p in list(self['pops']):
                if p['jobId'][:4] in ["UNI_", "ENT_", "AIR_", "MIL_", "FER_", "BOR_"]:
                    continue
                size = p['size']
                new_size = int(size * demand_factor)
                addtl = new_size - size
                points_by_id[p['residenceId']]['residents'] += addtl
                points_by_id[p['jobId']]['jobs'] += addtl
                p['size'] = new_size
    
    def consolidate_pops(self, 
                         consolidate_max_size=[25, 10, 5, 2], 
                         consolidate_distance=[2000, 4000, 80000, 16000]):
        """
        Merges job demand below threshold pop sizes into larger pops that have 
        the same home (or work) node and a nearby work (or home) node.
        
        Inputs
        ------
        consolidate_max_size : list of int. Threshold population sizes where all 
                               pops below the size are eligible for consolidation 
                               within the corresponding `consolidate_distance`.
                               Default: [25, 10, 5, 2]
        consolidate_distance : list of int or float. Distances in meters for 
                               consolidation of pops less than the corresponding 
                               `consolidate_max_size`.
                               Default: [2000, 4000, 80000, 16000]
        """
        print("Consolidating pops of sizes <", consolidate_max_size, 
              "among points within", consolidate_distance, "meters")
        if not isinstance(consolidate_max_size, list):
            consolidate_max_size = [consolidate_max_size]
        if not isinstance(consolidate_distance, list):
            consolidate_distance = [consolidate_distance]
        assert len(consolidate_max_size) == len(consolidate_distance), "Must provide the same number of values for both consolidate_max_size and consolidate_distance. Received:\nconsolidate_max_size = " + str(consolidate_max_size) + "\nconsolidate_distance = " + str(consolidate_distance)
            
        point_map = {p['id']: p for p in self['points']}
        point_ids = [p['id'] for p in self['points']]
        id_to_idx = {pid: i for i, pid in enumerate(point_ids)}
        
        # Calculate distance matrix between points
        coords = np.array([p['location'] for p in self['points']])
        lons, lats = coords[:, 0], coords[:, 1]
        dist_matrix = U.haversine(lons[:, None], lats[:, None], lons[None, :], lats[None, :])
        
        delta_res = {pid: 0 for pid in point_ids}
        delta_job = {pid: 0 for pid in point_ids}
        removed_pop_ids = set()
        
        # Helper for target selection
        def get_target_id(id1, id2, size, delta_dict):
            d1, d2 = delta_dict[id1], delta_dict[id2]
            if d1 < 0 and d2 >= 0: return id1
            if d2 < 0 and d1 >= 0: return id2
            if abs(d1 - d2) > size:
                return id1 if d1 < d2 else id2
            s1 = point_map[id1]['jobs'] + point_map[id1]['residents']
            s2 = point_map[id2]['jobs'] + point_map[id2]['residents']
            return id1 if s1 < s2 else id2
        
        for ic in range(len(consolidate_max_size)):
            # Pass 1: Group pops by job to consolidate residences
            if self.verb:
                print("  Consolidating residences")
            job_groups = defaultdict(list)
            for pop in self['pops']:
                job_groups[pop['jobId']].append(pop)
            
            pop_map = {p['id']: p for p in self['pops']}
            
            for job_id, group in job_groups.items():
                if job_id[:4] in ["UNI_", "ENT_", "AIR_", "MIL_", "FER_", "BOR_"]:
                    continue
                if len(group) < 2: continue
                # Sort by size so we try to merge smaller ones first
                group.sort(key=lambda x: pop_map[x['id']]['size'])
                
                for i in range(len(group)):
                    p_i = pop_map[group[i]['id']]
                    if p_i['id'] in removed_pop_ids or pop_map[p_i['id']]['size'] >= consolidate_max_size[ic]: continue
                    
                    for j in range(i + 1, len(group)):
                        p_j = pop_map[group[j]['id']]
                        if p_j['id'] in removed_pop_ids or pop_map[p_j['id']]['size'] >= consolidate_max_size[ic]: continue
                        
                        # Spatial check
                        if dist_matrix[id_to_idx[p_i['residenceId']], id_to_idx[p_j['residenceId']]] <= consolidate_distance[ic]:
                            target = get_target_id(p_i['residenceId'], p_j['residenceId'], pop_map[p_j['id']]['size'], delta_res)
                            survivor, victim = (p_i, p_j) if p_i['residenceId'] == target else (p_j, p_i)
                            
                            # Update metadata
                            v_size = pop_map[victim['id']]['size']
                            source_id = victim['residenceId']
                            point_map[source_id]['residents'] -= v_size
                            point_map[target]['residents'] += v_size
                            delta_res[source_id] -= v_size
                            delta_res[target] += v_size
                            
                            if victim['id'] in point_map[source_id]['popIds']:
                                point_map[source_id]['popIds'].remove(victim['id'])
                            if survivor['id'] not in point_map[target]['popIds']:
                                point_map[target]['popIds'].append(survivor['id'])
                            
                            # Merge logic
                            pop_map[survivor['id']]['size'] += v_size
                            pop_map[survivor['id']]['residenceId'] = target
                            removed_pop_ids.add(victim['id'])
                            if victim['id'] == p_i['id']:
                                break
                            
                            if pop_map[survivor['id']]['size'] >= consolidate_max_size[ic]: break
            
            # Pass 2: Group pops by residence to consolidate jobs
            if self.verb:
                print("  Consolidating jobs")
            # Refresh pop list to exclude removed ones
            remaining_pops = [p for p in self['pops'] if p['id'] not in removed_pop_ids]
            res_groups = defaultdict(list)
            for pop in remaining_pops:
                res_groups[pop['residenceId']].append(pop)
            
            for res_id, group in res_groups.items():
                if res_id[:4] in ["UNI_", "MIL_"]:
                    continue
                if len(group) < 2: continue
                group.sort(key=lambda x: pop_map[x['id']]['size'])
                
                for i in range(len(group)):
                    p_i = pop_map[group[i]['id']]
                    if p_i['id'] in removed_pop_ids or pop_map[p_i['id']]['size'] >= consolidate_max_size[ic]: continue
                    
                    for j in range(i + 1, len(group)):
                        p_j = pop_map[group[j]['id']]
                        if p_j['id'] in removed_pop_ids or pop_map[p_j['id']]['size'] >= consolidate_max_size[ic]: continue
                        
                        if dist_matrix[id_to_idx[p_i['jobId']], id_to_idx[p_j['jobId']]] <= consolidate_distance[ic]:
                            target = get_target_id(p_i['jobId'], p_j['jobId'], pop_map[p_j['id']]['size'], delta_job)
                            survivor, victim = (p_i, p_j) if p_i['jobId'] == target else (p_j, p_i)
                            
                            v_size = pop_map[victim['id']]['size']
                            source_id = victim['jobId']
                            point_map[source_id]['jobs'] -= v_size
                            point_map[target]['jobs'] += v_size
                            delta_job[source_id] -= v_size
                            delta_job[target] += v_size
                            
                            if victim['id'] in point_map[source_id]['popIds']:
                                point_map[source_id]['popIds'].remove(victim['id'])
                            if survivor['id'] not in point_map[target]['popIds']:
                                point_map[target]['popIds'].append(survivor['id'])
                            
                            pop_map[survivor['id']]['size'] += v_size
                            pop_map[survivor['id']]['jobId'] = target
                            removed_pop_ids.add(victim['id'])
                            if victim['id'] == p_i['id']:
                                break
                            
                            if pop_map[survivor['id']]['size'] >= consolidate_max_size[ic]: break
        
        self['pops'] = [p for p in self['pops'] if p['id'] not in removed_pop_ids]
        
        # Ensure consistent points data
        self.update(self.sanitize(self))
        """
        # Re-build popIds list for points
        for p in self['points']:
            p['popIds'] = []
        points_by_id = {p["id"]: p for p in self["points"]}
        for p in self['pops']:
            points_by_id[p['residenceId']]['popIds'].append(p['id'])
            points_by_id[p['jobId']]['popIds'].append(p['id'])
        # Clear out any points that now have no pops
        self['points'] = [p for p in self['points'] if p['jobs'] + p['residents'] > 0]
        """

    def merge_identical_commutes(self):
        """
        Merges any pops that have the same exact home and work nodes.
        """
        if self.verb:
            print("Merging pops with identical commutes")
        index_map = defaultdict(list)
        for idx, entry in enumerate(self['pops']):
            key = (entry["residenceId"], entry["jobId"])
            index_map[key].append(idx)
        new_pops = []
        keys = list(index_map.keys())
        points_by_id = {p["id"]: p for p in self["points"]}
        for k in keys:
            imerge = index_map[k]
            nmerge = len(imerge)
            if nmerge > 1:
                pop = {
                    "id" : self['pops'][imerge[0]]["id"],
                    "residenceId" : self['pops'][imerge[0]]["residenceId"],
                    "jobId" : self['pops'][imerge[0]]["jobId"],
                    "size" : int(np.sum([self['pops'][imerge[i]]["size"] for i in range(nmerge)])),
                    "drivingSeconds"  : weighted_mean([self['pops'][imerge[i]]["drivingSeconds"] for i in range(nmerge)], 
                                                      [self['pops'][imerge[j]]["size"] for j in range(nmerge)]),
                    "drivingDistance" : weighted_mean([self['pops'][imerge[i]]["drivingDistance"] for i in range(nmerge)], 
                                                      [self['pops'][imerge[j]]["size"] for j in range(nmerge)])
                }
                # Update points to forget about old pops that no longer exist
                for i in range(1,nmerge):
                    points_by_id[k[0]]["popIds"].remove(self['pops'][imerge[i]]["id"])
                    if k[0] != k[1]:
                        points_by_id[k[1]]["popIds"].remove(self['pops'][imerge[i]]["id"])
            else:
                pop = self['pops'][imerge[0]]
            new_pops.append(pop)
        self['pops'] = new_pops
    
    def cluster_points(self,
                       max_pop_threshold=[25, 50, 75, 200, 500, 5000, 15000, np.inf],
                       buffer_meters=[1500, 1000, 500, 250, 200, 150, 125, 100],
                       max_workers=4,
                       max_rounds=5):
        """
        Clusters spatial demand points based on size and distance thresholds.
        
        Inputs
        ------
        max_pop_threshold : (optional) list of int. demand point size 
                            thresholds for Colin's clustering approach.
                            Colin uses: [200, 500, 5000, 15000, np.inf]
                            Default: [25, 50, 75, 200, 500, 5000, 15000, np.inf]
        buffer_meters : (optional) list of int or float. distance from demand 
                        points to merge when using Colin's clustering approach.
                        Must match max_pop_threshold in length.
                        Colin uses: [250, 200, 150, 125, 100]
                        Default: [1500, 1000, 500, 250, 200, 150, 125, 100]
        """
        if self.verb:
            print("Clustering points based on Colin's method")
        merged_points = []
        pts = copy.deepcopy(self["points"])
        counter = 0

        while (counter < max_rounds):  # Fail-safe
            # Order points by size
            size_of_points = np.array([p["residents"] + p["jobs"] for p in pts])

            unique_locs = np.empty((0, 2), dtype=float)
            loc_assignments = []
            isort = np.argsort(size_of_points)[::-1]  # Largest -> smallest
            sorted_points = [pts[ip] for ip in isort]

            size_of_points = size_of_points[isort]  # Reorder to match

            # First go from largest -> smallest to figure out which are merged where
            for ipoint, p in enumerate(sorted_points):
                if self.verb:
                    print(f"  ({counter + 1}) Determining mergers: "
                          f"{ipoint + 1} / {len(sorted_points)}", end="\r")

                # Determine the buffer size for merging this point
                merge_buffer = buffer_meters[size_of_points[ipoint] <= max_pop_threshold][0]
                if not ipoint:
                    unique_locs = np.vstack([unique_locs, p["location"]])
                    loc_assignments.append(0)
                else:
                    # Determine if any existing locations are close enough to this one
                    dists = U.haversine(p["location"][0], p["location"][1],
                                      unique_locs[:, 0], unique_locs[:, 1])
                    iloc = dists.argmin()
                    if dists[iloc] > merge_buffer:
                        # New point
                        loc_assignments.append(len(unique_locs))
                        unique_locs = np.vstack([unique_locs, p["location"]])
                    else:
                        # Existing point
                        loc_assignments.append(iloc)
            if self.verb:
                print("")

            merged_points = []
            pops_by_id = {p["id"]: p for p in self["pops"]}

            # Then merge the points
            if self.verb:
                print((len(str(counter + 1)) + 4) * " " + " Merging")

            merge_points_worker = functools.partial(
                merge_points,
                loc_assignments=loc_assignments,
                sorted_points=sorted_points,
                size_of_points=size_of_points,
                pops_by_id=pops_by_id
            )

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(merge_points_worker, enumerate(unique_locs)))

            # Process results
            merged_points = []
            for merged_point, updated_pops in results:
                merged_points.append(merged_point)
                for pop in updated_pops:
                    if pop["residenceId"] == merged_point["id"]:
                        pops_by_id[pop["id"]]["residenceId"] = merged_point["id"]
                    if pop["jobId"] == merged_point["id"]:
                        pops_by_id[pop["id"]]["jobId"] = merged_point["id"]

            if len(merged_points) == len(pts):
                # It is already in a steady state, no need to continue
                break
            pts = copy.deepcopy(merged_points)  # Update `pts` for next round
            counter += 1

        self["points"] = merged_points
    
    def agglomerate_pops(self, SMALL_THRESHOLD=100, 
                         DISTANCE_THRESHOLD_NONCBD=0.1, 
                         DISTANCE_THRESHOLD_CBD=0.05, cbd_bbox=None):
        """
        Agglomerates pops below a threshold with the same job site into 
        super-origins.
        
        Inputs
        ------
        SMALL_THRESHOLD: int.  Pops with size below this threshold are eligible 
                               for agglomeration.
        DISTANCE_THRESHOLD_NONCBD: float.  Agglomeration parameter in degrees 
                                   for pops outside of the CBD.  If no CBD is 
                                   specified, then it is applied globally.
        DISTANCE_THRESHOLD_CBD: float.  Like `DISTANCE_THRESHOLD_NONCBD`, but 
                                        for pops in the CBD.
        cbd_bbox: list, floats.  Bounding box for the CBD. 
                                 [min_lon, min_lat, max_lon, max_lat]
        """
        if self.verb:
            print("Agglomerating pops below a threshold size of", SMALL_THRESHOLD)
        points_by_id = {p['id'] : p for p in self['points']}
        flows_by_dest = defaultdict(list)
        super_origin_counter_pt  = 0
        super_origin_counter_pop = 0
        for pop in self['pops']:
            flows_by_dest[pop["jobId"]].append(pop)
            if pop['id'].startswith('agg_'):
                # Set last counter - there is +=1 before using below
                super_origin_counter_pop = max(super_origin_counter_pop, 
                                               int(pop['id'].replace('agg_', '')))
        for pt in self['points']:
            if pt['id'].startswith('SO_'):
                # Set last counter - there is +=1 before using below
                super_origin_counter_pt = max(super_origin_counter_pt, 
                                              int(pt['id'].replace('SO_', '')))

        new_pops = []
        new_points = list(self['points'])

        for dest_id, flows in flows_by_dest.items():
            orig_total = sum(f["size"] for f in flows)
            large_flows = [f.copy() for f in flows if f["size"] >= SMALL_THRESHOLD]
            small_flows = [f.copy() for f in flows if f["size"] < SMALL_THRESHOLD]
            new_flows_for_dest = []

            # cluster small origins spatially
            #origin_coords = []
            #small_indices_with_coords = []
            cbd_coords, cbd_indices = [], []
            noncbd_coords, noncbd_indices = [], []
            for i, f in enumerate(small_flows):
                loc = points_by_id[f["residenceId"]]["location"]
                if loc is not None:
                    #origin_coords.append(loc)
                    #small_indices_with_coords.append(i)
                    if in_cbd(loc, cbd_bbox):
                        cbd_coords.append(loc)
                        cbd_indices.append(i)
                    else:
                        noncbd_coords.append(loc)
                        noncbd_indices.append(i)
            #origin_coords = np.array(origin_coords) if origin_coords else np.empty((0, 2))
            cbd_coords = np.array(cbd_coords) if cbd_coords else np.empty((0, 2))
            noncbd_coords = np.array(noncbd_coords) if noncbd_coords else np.empty((0, 2))

            initial_clusters = []
            # Non-CBD regime
            if len(noncbd_indices) > 1:
                clustering_non = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=DISTANCE_THRESHOLD_NONCBD,  # larger threshold
                    linkage="ward"
                ).fit(np.array(noncbd_coords))
                labels_non = clustering_non.labels_
                label_to_indices_non = defaultdict(list)
                for lbl, idx_in_valid in zip(labels_non, range(len(noncbd_indices))):
                    label_to_indices_non[lbl].append(noncbd_indices[idx_in_valid])
                initial_clusters.extend(label_to_indices_non.values())
            elif len(noncbd_indices) == 1:
                initial_clusters.append([noncbd_indices[0]])
            
            # CBD regime
            if len(cbd_indices) > 1:
                clustering_cbd = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=DISTANCE_THRESHOLD_CBD,  # smaller threshold
                    linkage="ward"
                ).fit(np.array(cbd_coords))
                labels_cbd = clustering_cbd.labels_
                label_to_indices_cbd = defaultdict(list)
                for lbl, idx_in_valid in zip(labels_cbd, range(len(cbd_indices))):
                    label_to_indices_cbd[lbl].append(cbd_indices[idx_in_valid])
                initial_clusters.extend(label_to_indices_cbd.values())
            elif len(cbd_indices) == 1:
                initial_clusters.append([cbd_indices[0]])
            
            used = set(sum(initial_clusters, []))
            no_coord = [i for i in range(len(small_flows)) if i not in used]
            initial_clusters.extend([[i] for i in no_coord])
            
            # emit clusters as SuperOrigins
            for idxs in initial_clusters:
                cl_flows = [small_flows[i] for i in idxs]
                sizes = [f["size"] for f in cl_flows]
                seconds_vals  = [f["drivingSeconds" ] for f in cl_flows]
                distance_vals = [f["drivingDistance"] for f in cl_flows]
                agg_size = sum(sizes)
                agg_seconds = int(round(weighted_mean(seconds_vals, sizes)))
                agg_distance = int(round(weighted_mean(distance_vals, sizes)))
                centroid = compute_centroid([points_by_id[f["residenceId"]]["location"] \
                                               for f in cl_flows], 
                                              [points_by_id[f["residenceId"]]["residents"] + \
                                               points_by_id[f["residenceId"]]["jobs"]        \
                                               for f in cl_flows])
                residents_sum = np.sum([points_by_id[f["residenceId"]]["residents"] \
                                        for f in cl_flows])

                super_origin_counter_pt  += 1
                super_origin_counter_pop += 1
                super_id = f"SO_{super_origin_counter_pt}"
                pop_id = f"agg_{super_origin_counter_pop}"

                new_flows_for_dest.append({
                    "residenceId": super_id,
                    "jobId": dest_id,
                    "drivingSeconds": agg_seconds,
                    "drivingDistance": agg_distance,
                    "size": agg_size,
                    "id": pop_id
                })
                # Update the destination point's pops
                new_points.append({
                    "id": super_id,
                    "location": centroid,
                    "jobs": 0,
                    "residents": agg_size, 
                    "popIds": [pop_id]
                })

            new_flows_for_dest.extend(large_flows)

            # conservation check
            new_total = sum(f["size"] for f in new_flows_for_dest)
            if new_total != orig_total:
                raise ValueError(f"Commuter mismatch at destination {dest_id}: {orig_total} vs {new_total}")

            new_pops.extend(new_flows_for_dest)
        
        if self.verb:
            print("  Re-calculating points' residents and popIds")
        point_dict = {}
        for ip, p in enumerate(new_points):
            p['popIds'] = []
            p['residents'] = 0
            p['jobs'] = 0
            point_dict[p['id']] = ip

        for ip, p in enumerate(new_pops):
            new_points[point_dict[p['residenceId']]]['residents'] += p['size']
            new_points[point_dict[p['residenceId']]]['popIds'].append(p['id'])
            new_points[point_dict[p['jobId']]]['jobs'] += p['size']
            if p['residenceId'] != p['jobId']:
                new_points[point_dict[p['jobId']]]['popIds'].append(p['id'])

        self['points'] = new_points
        self['pops'] = new_pops

    def move_points(self, point_ids=None, coords=None, new_coords=None, 
                    dist_tol=500):
        """
        Move multiple specified points to new locations.
        
        Inputs
        ------
        point_ids: list-like of str. Point IDs to be moved.
        coords: list of lists/tuples of floats. e.g., [[lon1, lat1], [lon2, lat2]].
                Used to select the closest points to the provided coordinates.
        new_coords: list of lists/tuples of floats.
                New coordinates where the points are moved to.
        dist_tol: int or float. Distance tolerance in meters when matching `coords`. 
        
        Notes
        -----
        You must provide either point_ids or coords, but not both.
        Must have the same number of entries in your targets (point_ids or coords) 
        and `new_coords`.
        """
        # Sanity checks for mutual exclusivity
        if point_ids is None and coords is None:
            raise ValueError("move_points: must provide either `point_ids` "
                             "OR `coords`.")
        elif point_ids is not None and coords is not None:
            raise ValueError("move_points: must provide either `point_ids` "
                             "OR `coords`, but not both.")
        
        if new_coords is None:
            raise ValueError("move_points: must provide `new_coords`.")

        # Ensure new_coords is a list of lists/tuples
        if not isinstance(new_coords[0], (list, tuple)):
            new_coords = [new_coords]

        # Validate new_coords format
        for ic, coord in enumerate(new_coords):
            if len(coord) != 2:
                raise ValueError("Provided new_coords must have [lon, lat] format.\n"
                                 f"Received '{coord}' at index {ic} for new_coords")

        # Ordered list of point indices (in self['points']) to be moved
        target_indices = []

        # Identify points to be moved
        if point_ids is not None:
            if not isinstance(point_ids, (list, tuple)):
                point_ids = [point_ids]
            # Match by IDs while preserving input order
            existing_ids_to_idx = {p['id']: i for i, p in enumerate(self['points'])}
            missing_ids = [pid for pid in point_ids if pid not in existing_ids_to_idx]
            
            if missing_ids:
                raise ValueError(f"The following point_ids were not found: "
                                 f"{set(missing_ids)}")
            
            if len(point_ids) != len(new_coords):
                raise ValueError(f"Must provide the same number of entries "
                                 f"for `point_ids` ({len(point_ids)}) "
                                 f"and `new_coords` ({len(new_coords)})")
            target_indices = [existing_ids_to_idx[pid] for pid in point_ids]
        else:
            # Ensure coords is a list of lists/tuples
            if not isinstance(coords[0], (list, tuple)):
                coords = [coords]

            # Validate coords format
            for ic, coord in enumerate(coords):
                if len(coord) != 2:
                    raise ValueError("Provided coords must have [lon, lat] format.\n"
                                     f"Received '{coord}' at index {ic} for coords")

            if len(coords) != len(new_coords):
                raise ValueError(f"Must provide the same number of coordinate "
                                 f"pairs for both `coords` ({len(coords)}) "
                                 f"and `new_coords` ({len(new_coords)})")

            # Find closest points for all provided coordinates
            point_locs = np.array([p['location'] for p in self['points']])
            for coord in coords:
                dists = U.haversine(coord[0], coord[1], point_locs[:, 0], point_locs[:, 1])
                min_idx = dists.argmin()
                if dists[min_idx] > dist_tol:
                    raise ValueError(f"The nearest point to coordinate {coord} is "
                                     f"{dists[min_idx]} meters away, exceeding "
                                     f"tolerance ({dist_tol} m).")
                target_indices.append(min_idx)

        # Move the points
        for i, idx in enumerate(target_indices):
            self['points'][idx]['location'] = new_coords[i]
    
    def add_points(self, poi):
        """
        Generates special demand points and splits pops between forced required locations

        and randomly distributed resident locations. Expects [lon, lat] coordinates.

        Inputs
        ------
        poi : dict or list of dicts
            A list of dictionaries representing the POIs to generate.
            Each must contain:
                - type: str. Type of special demand point.
                             To see a list of available types, instantiate 
                             a DemandData object, then print 
                             obj.special_demand_codes.keys()
                - name: str. Name of the point (e.g., 'Haneda', 'Tokyo Dome')
                - code: (optional) str. Use short code to make more 
                                        size-efficient demand files.
                - location: list/tuple of [lon, lat]. Coordinates for the point.
                - total_capacity: int. Total number of people to be assigned to this point.
                - pop_size: int. Size of each created pop.
                - required_locs: (optional) list of [lon, lat] coordinates. Selects 
                                 the point closest to each coordinate pair and 
                                 assigns a single pop to it.
                - pop_size_req: (optional) int. Size of assigned pops. 
                                Falls back to pop_size if not specified.
                - pop_size_remain: (optional) int. Size of remaining pops assigned 
                                   automatically by the code.
                                   Falls back to pop_size if not specified.
                - residential_split: (optional) float, 0.0 to 1.0.
                                     Percentage of total_capacity allocated to 
                                     pops that live at the new point and work 
                                     elsewhere.
                                     Default: 0
                - exponent: (optional) float, >=0.0. Distance decay exponent 
                            when assigning pops.
                            If not specified, falls back to pre-determined 
                            defaults.
                - max_distance: (optional) int or float. Maximum commute 
                                           distance in meters for most 
                                           generated pops. Beyond this 
                                           distance, there is a 10x reduction 
                                           in probability.
                                           Depot has a hard-coded limit of 
                                           200 km for any generated commutes.
                - merge_within: (optional) int or float. 
        """
        new_demand_points = []
        counter = 0
        
        if isinstance(poi, dict):
            poi = [poi]

        if not self["points"]:
            raise ValueError("No existing baseline points found in DemandData.")

        # To later exclude from some pop generation
        valid_special_prefixes = set(self.special_demand_ids.keys())

        for item in poi:
            poi_type = item["type"]
            poi_name = item["name"]

            # Resolve type code from taxonomy schema
            schema_code = self.special_demand_codes.get(poi_type)
            type_code = schema_code if schema_code else poi_type.replace(" ", "_").upper()

            # Resolve identifier (code or name)
            if "code" in item and item["code"]:
                identifier = item["code"]
            else:
                identifier = poi_name.replace(" ", "_").strip()

            # Construct the point ID
            poi_point_id = f"{type_code}_{identifier}"
            item['point_id'] = poi_point_id

            # Resolve distance decay exponent
            exponent = item.get("exponent")
            if exponent is None:
                exponent = self.get_exponent(poi_type)

            # Setup residents/jobs for this point
            total_capacity = item["total_capacity"]
            res_split = item.get("residential_split", 0.0)

            resident_capacity = int(total_capacity * res_split)
            job_capacity = int(total_capacity - resident_capacity)

            # Handle pop sizes
            base_pop_size = item["pop_size"]
            psize_req = item.get("pop_size_req", base_pop_size)
            psize_remain = item.get("pop_size_remain", base_pop_size)
            required_coords = item.get("required_locs", [])
            max_dist = item.get("max_distance")
            merge_within = item.get("merge_within")
            
            if self.verb:
                print(f"Adding {poi_type} demand for {poi_point_id}\n"
                      f"  Jobs: {job_capacity}\n  Residents: {resident_capacity}")

            # Initialize the destination/origin point entity
            point = {
                "id": poi_point_id,
                "location": item["location"],  # [lon, lat]
                "jobs": 0,
                "residents": 0,
                "popIds": [],
            }

            # Extract all existing point locations. Shape: (N, 2) -> [lon, lat]
            point_locs = np.array([p["location"] for p in self["points"]])
            # Distance matrix for gravity calculations
            dist_of_points = U.haversine(
                point["location"][0],
                point["location"][1],
                point_locs[:, 0],
                point_locs[:, 1],
            )
            
            if merge_within is not None:
                # Do not consider other special demand infrastructure points
                for idx, p in enumerate(self["points"]):
                    p_id_parts = p["id"].split("_", 1)
                    if len(p_id_parts) > 1 and p_id_parts[0] in valid_special_prefixes:
                        dist_of_points[idx] = 1e10 # do not merge this point
                inds = np.arange(len(self['points']), dtype=int)
                iloc_merge = inds[dist_of_points <= merge_within][::-1] # largest to smallest, so that points are deleted from the end of the array to the front
                pops_by_id = {p["id"]: p for p in self["pops"]}
                for iloc in iloc_merge:
                    point['jobs'] += self['points'][iloc]['jobs']
                    point['residents'] += self['points'][iloc]['residents']
                    point['popIds'] += self['points'][iloc]['popIds']
                    for popid in self['points'][iloc]['popIds']:
                        if pops_by_id[popid]['residenceId'] == self['points'][iloc]['id']:
                            pops_by_id[popid]['residenceId'] = point['id']
                        if pops_by_id[popid]['jobId'] == self['points'][iloc]['id']:
                            pops_by_id[popid]['jobId'] = point['id']
                    del self['points'][iloc]
                # Update locs and dists arrays
                point_locs = np.array([p["location"] for p in self["points"]])
                dist_of_points = U.haversine(
                    point["location"][0],
                    point["location"][1],
                    point_locs[:, 0],
                    point_locs[:, 1],
                )

            # Extract baseline pops
            base_residents = np.array([p["residents"] for p in self["points"]], dtype=float)
            base_jobs = np.array([p["jobs"] for p in self["points"]], dtype=float)

            # Identify closest existing points to the "required" locations
            ilocs_req = np.zeros(len(required_coords), dtype=int)
            for i, req_coord in enumerate(required_coords):
                distances = U.haversine(
                    req_coord[0],
                    req_coord[1],
                    point_locs[:, 0],
                    point_locs[:, 1],
                )
                ilocs_req[i] = distances.argmin()

            if len(ilocs_req) > 0:
                base_residents[ilocs_req] = 0.0
                base_jobs[ilocs_req] = 0.0

            # Do not consider other special demand infrastructure points
            for idx, p in enumerate(self["points"]):
                p_id_parts = p["id"].split("_", 1)
                if len(p_id_parts) > 1 and p_id_parts[0] in valid_special_prefixes:
                    base_residents[idx] = 0.0
                    base_jobs[idx] = 0.0

            # Calculate gravity weights
            # Create a mask where distance is greater than 0 to avoid ZeroDivisionError
            with np.errstate(divide='ignore', invalid='ignore'):
                distance_decay = dist_of_points**exponent
                job_weights = base_residents / distance_decay
                resident_weights = base_jobs / distance_decay

            # Explicitly force any point with a distance of exactly 0 to have a weight of 0.0
            job_weights[dist_of_points == 0] = 0.0
            resident_weights[dist_of_points == 0] = 0.0
            
            # Clean up any potential NaNs generated by 0 / 0 scenarios
            job_weights = np.nan_to_num(job_weights, nan=0.0, posinf=0.0, neginf=0.0)
            resident_weights = np.nan_to_num(resident_weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            if max_dist is not None:
                job_weights[dist_of_points > max_dist] /= 10
                resident_weights[dist_of_points > max_dist] /= 10
            
            # Hard cap gravity model to <= 200 km
            job_weights[dist_of_points > 200000] = 0
            resident_weights[dist_of_points > 200000] = 0
            
            if job_weights.max() == 0 and resident_weights.max() == 0:
                raise ValueError(f"No points are within 200 km")
            
            # Generate point jobs
            total_req_pop_mass = psize_req * len(required_coords)
            remain_job_capacity = job_capacity - total_req_pop_mass
            ntarget_jobs = max(0, int(remain_job_capacity / psize_remain))

            if job_weights.sum() > 0 and ntarget_jobs > 0:
                ilocs_job_remain = np.random.choice(
                    job_weights.size,
                    size=ntarget_jobs,
                    replace=True,
                    p=job_weights / job_weights.sum(),
                )
            else:
                ilocs_job_remain = np.array([], dtype=int)

            # Generate point residents
            ntarget_residents = max(0, int(resident_capacity / psize_remain))
            
            if resident_weights.sum() > 0 and ntarget_residents > 0:
                ilocs_resident_remain = np.random.choice(
                    resident_weights.size,
                    size=ntarget_residents,
                    replace=True,
                    p=resident_weights / resident_weights.sum(),
                )
            else:
                ilocs_resident_remain = np.array([], dtype=int)

            # Loop and write job pops
            for stage in range(2):
                if stage == 0:
                    psize = psize_req
                    locs_arr = ilocs_req
                else:
                    psize = psize_remain
                    locs_arr = ilocs_job_remain

                for iloc in locs_arr:
                    counter += 1
                    pop_id = f"{poi_point_id}_{counter}"

                    pop = {
                        "id": pop_id,
                        "residenceId": self["points"][iloc]["id"],
                        "jobId": point["id"],
                        "size": psize,
                        "drivingSeconds": 0,
                        "drivingDistance": 0,
                    }

                    self["pops"].append(pop)
                    self["points"][iloc]["residents"] += pop["size"]
                    point["jobs"] += pop["size"]
                    self["points"][iloc]["popIds"].append(pop["id"])
                    point["popIds"].append(pop["id"])

            # Loop and write resident pops
            for iloc in ilocs_resident_remain:
                counter += 1
                pop_id = f"{poi_point_id}_{counter}"

                pop = {
                    "id": pop_id,
                    "residenceId": point["id"],
                    "jobId": self["points"][iloc]["id"],
                    "size": psize_remain,
                    "drivingSeconds": 0,
                    "drivingDistance": 0,
                }

                self["pops"].append(pop)
                point["residents"] += pop["size"]
                self["points"][iloc]["jobs"] += pop["size"]
                self["points"][iloc]["popIds"].append(pop["id"])
                point["popIds"].append(pop["id"])

            new_demand_points.append(point)
            item['pop_ids'] = point['popIds']

        # Merge the new special POIs into the main points list
        self["points"] += new_demand_points
        
        # Log these for output to the points schema
        self.added_special_demand_points += poi
    
    def del_points(self, point_ids=None, coords=None, dist_tol=500):
        """
        Deletes multiple specified points and all pops associated with them.
        
        Inputs
        ------
        point_ids: list-like of str.  Point IDs to be deleted.
        coords: list of lists/tuples of floats.  e.g., [[lon1, lat1], [lon2, lat2]].
                Used to select the closest points to the provided coordinates.
        dist_tol: int or float.  Distance tolerance in meters when matching `coords`. 
        
        Notes
        -----
        You must provide either point_ids or coords, but not both.
        """
        # Sanity checks
        if point_ids is None and coords is None:
            raise ValueError("del_points: must provide either `point_ids` OR `coords`.")
        elif point_ids is not None and coords is not None:
            raise ValueError("del_points: must provide either `point_ids` OR `coords`, but not both.")
        
        # Target IDs set we want to eventually populate and remove
        points_to_remove = set()

        # Identify points to be deleted
        if point_ids is not None:
            if not isinstance(point_ids, (list, tuple)):
                point_ids = [point_ids]
            # Convert input to a set for O(1) lookups
            input_ids = set(point_ids)
            # Verify they exist and add to our removal pool
            existing_ids = {p['id'] for p in self['points']}
            missing_ids = input_ids - existing_ids
            if missing_ids:
                raise ValueError(f"The following point_ids were not found: {missing_ids}")
            points_to_remove = input_ids

        else:
            # Ensure coords is a list of lists/tuples
            if not isinstance(coords[0], (list, tuple)):
                coords = [coords]
            # Find closest points for all provided coordinates
            point_locs = np.array([p['location'] for p in self['points']])
            for coord in coords:
                # Vectorized distance calculation for one coordinate pair against all points
                dists = U.haversine(coord[0], coord[1], point_locs[:, 0], point_locs[:, 1])
                min_idx = dists.argmin()
                if dists[min_idx] > dist_tol:
                    raise ValueError(f"The nearest point to coordinate {coord} is "\
                                     f"{dists[min_idx]} meters away, exceeding "\
                                     f"tolerance ({dist_tol} m).")
                points_to_remove.add(self['points'][min_idx]['id'])

        # Gather all pop IDs associated with the target points
        pops_to_remove = set()
        points_by_id = {p['id'] : p for p in self['points']}
        pops_by_id = {p['id']: p for p in self['pops']}
        for target_id in points_to_remove:
            point = points_by_id[target_id]
            for p_id in point['popIds']:
                pops_to_remove.add(p_id)
                pop = pops_by_id[p_id]
                if pop['residenceId'] != target_id:
                    points_by_id[pop['residenceId']]['popIds'].remove(p_id)
                elif pop['jobId'] != target_id:
                    points_by_id[pop['jobId']]['popIds'].remove(p_id)

        # Bulk filter 'pops' and 'points'
        self['pops']   = [p for p in self['pops']   if p['id'] not in pops_to_remove]
        self['points'] = [p for p in self['points'] if p['id'] not in points_to_remove]
    
    def _load_schema(self, filepath=None):
        """
        Private helper method to read data into structures.
        
        Inputs
        ------
        filepath: str. Path to schema file to load.
                       If None, defaults to the included special demand typing.
        """
        self.special_demand_exp = {
            "AIR": 0.5,
            "AMU": 1,
            "AQU": 1.2,
            "BTH": 1,
            "CNV": 3,
            "CUL": 1,
            "EVT": 1,
            "EXT": 0.5,
            "GOV": 1.2,
            "HER": 0.5,
            "HOS": 1.5,
            "LIB": 2,
            "MIL": 1.2,
            "MUS": 1,
            "NAT": 1,
            "PORT": 0.7,
            "PRK": 1.5,
            "REL": 2,
            "RST": 2,
            "SCH": 2.5,
            "SHP": 1.5,
            "SPO": 1.5,
            "UNI": 2,
            "ZOO": 1,
            "nature_park": 1,
            "lake": 1.5,
            "arena": 1,
            "racetrack": 1,
            "sports_complex": 1.1,
            "sports_park": 1.2,
            "stadium": 1
        }
        self.special_demand_codes = {}
        self.special_demand_descs = {}
        self.special_demand_types = {}
        self.special_demand_subtypes = {}
        self.special_demand_labels = {}
        plural = inflect.engine() # to pluralize labels
        
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), "special_demand_types.json")
        with open(filepath, "r") as f:
            types_data = json.load(f)["types"]
        self.f_types_schema = filepath

        # Using defaultdict locally for easy aggregation
        ids_accumulator = defaultdict(list)

        for stype in types_data:
            parent_id = stype["id"]
            scode = stype.get("code")

            # Handle parent category
            if scode:
                self.special_demand_codes[parent_id] = scode
                ids_accumulator[scode].append(parent_id)

            self.special_demand_descs[parent_id] = stype.get("description", {}).get("__default__", "")
            
            # Log parent types of points schema
            self.special_demand_types[parent_id] = scode
            # And labels
            singular = plural.singular_noun(stype['label']['__default__'].lower())
            if singular:
                self.special_demand_labels[scode] = plural.plural(singular).title()
            else:
                self.special_demand_labels[scode] = plural.plural(stype['label']['__default__'].lower()).title()

            # Handle nested subtypes
            for subtype in stype.get("sub_types", []):
                sub_id = subtype["id"]
                sub_code = subtype.get("code", scode)

                if sub_code:
                    self.special_demand_codes[sub_id] = sub_code
                    ids_accumulator[sub_code].append(sub_id)
                else:
                    print(f"WARNING: No valid code for subtype '{sub_id}'")

                # Classification fallback to default parent description
                classification = subtype.get("metadata", {}).get("classification")
                self.special_demand_descs[sub_id] = (
                    classification if classification
                    else self.special_demand_descs[parent_id]
                )
                
                # Store subtypes -> parent types mapping, for writing points schema
                self.special_demand_subtypes[sub_id] = parent_id

        # Cast back to a clean standard dictionary for the public instance attribute
        self.special_demand_ids = dict(ids_accumulator)

    def get_exponent(self, poi_id):
        """
        Looks up the decay exponent for a given point of interest type ID.
        Checks the specific subtype ID first, falling back to the parent category code.
        
        Inputs
        ------
        poi_id: str. Point of interest special demand type.
        
        Outputs
        -------
        exp: int or float. Exponent for distance decay factor when assigning pops.
        """
        # Direct ID match (e.g. 'stadium', 'lake')
        if poi_id in self.special_demand_exp:
            return self.special_demand_exp[poi_id]

        # Category code fallback match (e.g., 'baseball_field' -> 'SPO')
        category_code = self.special_demand_codes.get(poi_id)
        # Default to 1.0 if not found
        return self.special_demand_exp.get(category_code, 1.0)  
    
    def save_schemas(self):
        """
        Saves out the special demand points and types schema files
        """
        if not os.path.exists(self.ftypes_schema):
            # Copy types schema
            shutil.copy(self.f_types_schema, self.ftypes_schema)
        
        if not os.path.exists(self.fpoints_schema) or \
           not self.has_existing_special_demand:
            # Create points schema
            points_schema = {}
            points_schema["$schema"] = "special_demand_points.schema.json"
            points_schema["version"] = 1
            points_schema["map_code"] = self.map_code
            
            now_utc = datetime.now(timezone.utc)
            datetime_string = now_utc.isoformat()
            points_schema["generated_at"] = datetime_string
            
            points_schema["points"] = []
        else:
            with open(self.fpoints_schema, 'r') as f:
                points_schema = json.load(f)
        
        # Add special demand points from this session into output
        for p in self.added_special_demand_points:
            pt = {}
            pt['point_id'] = p['point_id']
            pt['name'] = {'__default__' : p['name']}
            # Determine if the type is a parent or sub type
            if p['type'] in self.special_demand_types.keys():
                pt['type'] = p['type']
            else:
                pt['type'] = self.special_demand_subtypes[p['type']]
                pt['sub_type'] = p['type']
            pt['pop_ids'] = p['pop_ids']
            points_schema["points"].append(pt)
        
        # Save it
        with open(self.fpoints_schema, "w") as json_file:
            json.dump(points_schema, json_file, indent=4)
    
    def create_config(self, name, bbox=None, 
                      description="", creator="", version="", country="",
                      initial_view_state=None):
        """
        Creates the config.json file needed for Railyard import.
        
        Inputs
        ------
        name: str. Map name as it will appear in Railyard and SB.
        bbox: list, floats. Bounding box for the map, 
                            [min_lon, min_lat, max_lon, max_lat]
                            This should contain the full playable area.
                            If None, this is estimated from the demand points.
                            Default: None
        description: str. Map description that shows in SB.
        creator: str. Map creator's handle (that's you).
        version: str. Version in X.Y.Z format.
        country: str. Two-digit country code.
        initial_view_state: list, floats. [lon, lat] coordinates for the 
                            camera's starting position.
                            If None, it will be estimated from the average of 
                            the bounding box.
                            Default: None
        """
        if self.verb:
            print("Creating config.json file")
            if description=="":
                print("No description provided.")
            if creator=="":
                print("No creator provided.")
            if version=="":
                print("No version provided.")
        
        if bbox is None:
            if self.bbox is not None:
                bbox = self.bbox
            else:
                # Estimate it from the point locations
                if self.verb:
                    print("Estimating the bbox from points locations")
                locs = [p['location'] for p in self['points']]
                bbox = [round(float(v-0.001), 5) for v in np.amin(locs, axis=0)] + \
                       [round(float(v+0.001), 5) for v in np.amax(locs, axis=0)]
        
        total_pop = int(np.sum([p['size'] for p in self['pops']]))
        
        if initial_view_state is None:
            initial_view_state = [round((bbox[0] + bbox[2]) / 2., 5),
                                  round((bbox[1] + bbox[3]) / 2., 5)]
        elif not isinstance(initial_view_state, (list, tuple, np.ndarray)):
            raise ValueError("initial_view_state must be a list of floats in "
                             "[lon, lat] order.")
        
        config = {
            "code" : self.map_code,
            "name" : name,
            "bbox" : bbox,
            "description" : description,
            "population" : total_pop,
            "initialViewState" : {
                "latitude"  : float(initial_view_state[1]),
                "longitude" : float(initial_view_state[0]),
                "zoom" : 12,
                "bearing" : 0
            },
            "creator" : creator,
            "version" : version
        }
        if country != "":
            config["country"] = country
        
        with open(os.path.join(self.outputdir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
    def create_description(self, mapID, methodology, data_sources, 
                           license="GPLv3"):
        """
        Create a Markdown file with the contents for the map description 
        during Railyard submission.  Provides map details, demand statistics, 
        special demand information, and more.
        
        NOTE: If a point of interest is split between multiple points (e.g., 
              a large park, large university campus), if the points are named 
              exactly the same then they will be combined in the produced 
              description.md file.
        
        Inputs
        ------
        mapID: str. The map ID of the map submitted to Railyard.
                    Example: slurry-trondheim-no
        methodology: list of str. A description of how the map was created.
                                  Be as detailed as necessary.
                                  Since Depot and the US Demand Generator are 
                                  documented projects, if you used those, 
                                  you can just simply say that.  Include 
                                  embedded links where appropriate. See TPA 
                                  example.
        data_sources: list of str. Names of data products with embedded links 
                                   used to produce the map.
                                   See TPA example.
        license: str. The license you wish to release your map under.
                      If "GPLv3", text with an embedded link is auto-generated.
                      Otherwise, you must write out the full text with any 
                      embedded links to your chosen license.
                      Default: GPLv3
        """
        # Ensure consistent points data
        self.update(self.sanitize(self))
        
        with open(os.path.join(self.outputdir, "config.json"), "r") as f:
            config = json.load(f)
        
        with open(self.fpoints_schema, "r") as f:
            points = json.load(f)['points']
        
        if license == "GPLv3":
            license = """This map data is released under the <a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU General Public License v3.0</a>."""

        minlon, minlat, maxlon, maxlat = [np.radians(v) for v in config["bbox"]]
        area = (6371**2) * abs(np.sin(minlat) - np.sin(maxlat)) * abs(minlon - maxlon)

        # Header, top-level info
        description = f"""<h1>{config["name"]}</h1>\n"""\
        f"""<h3>{config["code"]} · {config["version"]}</h3>\n"""\
        f"""<p><img src="https://raw.githubusercontent.com/Subway-Builder-Modded/registry/refs/heads/main/maps/{unidecode(mapID)}/gallery/screenshot1.png" alt="Map Preview"></p>\n"""\
        """<h2>Coverage</h2>\n"""\
        """<table style="width: auto">\n"""\
        f"""<tr><td><strong>Bounding box</strong></td><td>{str(config["bbox"]).replace("[","").replace("]","")}</td></tr>\n"""\
        f"""<tr><td><strong>Bounding Box Area</strong></td><td>{int(np.floor(area))} km²</td></tr>\n"""\
        """</table>\n\n"""

        npoints = len(self['points'])
        npops = len(self['pops'])
        npeople = np.sum([p['size'] for p in self['pops']])
        assert npeople == config["population"], "Config mis-match. Config " \
                f"reports {config['population']} people, but loaded demand " \
                f"data has {npeople} people."
        
        # Calculate number per top-level demand category and each unique place
        # Places with the same name are combined in reporting
        categories = {k: {"total" : 0, "entries" : {}} for k in self.special_demand_labels.keys()}
        pops_by_id = {p['id'] : p for p in self['pops']}
        for p in points:
            cat = p['point_id'].split('_')[0]
            name = p['name']['__default__']
            if name not in categories[cat]['entries'].keys():
                pt = {
                    "type" : p['type'],
                    "size" : 0
                }
                if "sub_type" in p.keys():
                    pt['sub_type'] = p['sub_type']
                else:
                    pt['sub_type'] = None
                categories[cat]['entries'][name] = pt
            for pid in p['pop_ids']:
                if p['point_id'] in pid:
                    categories[cat]['entries'][name]['size'] += pops_by_id[pid]['size']
                    categories[cat]['total'] += pops_by_id[pid]['size']
        
        # Remove unused categories
        categories = {k : categories[k] for k in categories.keys() if categories[k]['total']}
        
        # Calculate total special demand
        nspec = int(np.sum([categories[k]['total'] for k in categories.keys()]))
        
        # Helper function for formatting strings of numbers
        def fmt(val):
            return format(val, ",") if val >= 10000 else val

        # Population Summary
        description += """<h2>Population Summary</h2>\n"""\
        """<table style="width: auto">\n"""\
        f"""<tr><td><strong>Total Modeled Demand</strong></td><td align="right">{fmt(config["population"])}</td></tr>\n"""\
        f"""<tr><td><strong>Modeled Normal Demand</strong></td><td align="right">{fmt(config["population"] - nspec)}</td></tr>\n"""\
        f"""<tr><td><strong>Modeled Special Demand</strong></td><td align="right">{fmt(nspec)}</td></tr>\n"""\
        """</table>\n\n"""

        # Map Statistics
        med_point = int(np.median([p['jobs'] + p['residents'] for p in self['points']]))
        avg_point = round(float(np.mean([p['jobs'] + p['residents'] for p in self['points']])), 1)
        med_pop = int(np.median([p['size'] for p in self['pops']]))
        avg_pop = round(float(np.mean([p['size'] for p in self['pops']])), 1)
        med_dist = round(float(np.median([p['drivingDistance'] / 1000 for p in self['pops']])), 2)
        avg_dist = round(float(np.mean([p['drivingDistance'] / 1000 for p in self['pops']])), 2)
        med_time = round(float(np.median([p['drivingSeconds'] / 60 for p in self['pops']])), 1)
        avg_time = round(float(np.mean([p['drivingSeconds'] / 60 for p in self['pops']])), 1)
        
        stats = [
            ("Demand Points", npoints),
            ("Populations", npops),
            ("Median Point Size", med_point),
            ("Mean Point Size", avg_point),
            ("Median Population Size", med_pop),
            ("Mean Population Size", avg_pop),
            ("Median Commute Distance (km)", med_dist),
            ("Mean Commute Distance (km)", avg_dist),
            ("Median Commute Time (min)", med_time),
            ("Mean Commute Time (min)", avg_time),
        ]
        
        description += """<h2>Map Statistics</h2>\n"""\
        """<table style="width: auto">\n"""
        
        for label, value in stats:
            description += f"""<tr><td><strong>{label}</strong></td><td align="right">{fmt(value)}</td></tr>\n"""
        description += """</table>\n\n"""

        description += "<h2>Special Demand</h2>\n"
        
        for cat in sorted(categories.keys()):
            description += """<details>\n"""\
            f"""<summary>{self.special_demand_labels[cat]} — {fmt(categories[cat]['total'])}</summary>\n\n"""\
            """<table style="width: auto">\n"""\
            """<tr><th align="left">Name</th><th align="right">Modeled Demand</th></tr>\n"""
            for entry in categories[cat]['entries'].keys():
                description += f"""<tr><td align="left">&nbsp;&nbsp;{entry}&nbsp;&nbsp;</td><td align="right">&nbsp;&nbsp;{fmt(categories[cat]['entries'][entry]['size'])}&nbsp;&nbsp;</td></tr>\n"""
            description += """</table>\n\n"""\
            """</details>\n\n"""
        
        description = description[:-1] # cut the last \n
        description += "<br>\n\n"

        description += """<h2>Additional Features</h2>\n"""\
        """<ul>\n"""\
        """<li><strong>Building Collision</strong> — A buildings index is included, providing in-game collision geometry for all non-filtered buildings. Buildings have foundations ranging from -10 m to -80 m, calculated based on the building's height and footprint; the in-game foundations map layer visualizes these.</li>\n"""\
        """<li><strong>Water Depths</strong> — A water depth index is included, preventing tracks from being placed within the water.  GEBCO bathymetric data are used where available, and a flat -4 m depth is assumed everywhere else; the in-game ocean foundations map layer visualizes these.</li>\n"""\
        """<li><strong>Place Labels</strong> — The map includes city, suburb, and neighborhood labels extracted from selected OSM place tags.</li>\n"""\
        """</ul>\n"""\
        """<h2>Methodology</h2>\n"""\
        """<ul>\n"""
        
        for m in methodology:
            description += f"""<p>{m}</p>\n"""
        
        description += """</ul>\n"""\
        """<h2>Data Sources</h2>\n"""\
        """<ul>\n"""
        
        for dsource in data_sources:
            description += f"""<p>{dsource}</p>\n"""
        
        description += """</ul>\n"""\
        """<h2>License</h2>\n"""\
        f"""<p>{license}</p>\n"""\
        """<h2>Credits</h2>\n"""\
        f"""<p>Map authored by {config['creator']}</p>\n"""

        with open(os.path.join(self.outputdir, "description.md"), "w") as f:
            f.write(description)

################################################################################

def process_home_node(i, demand, G, points_by_id):
    home_point = demand['points'][i]
    home_id = home_point['id']
    home_node = ox.nearest_nodes(G, Y=home_point['location'][1], X=home_point['location'][0])
    pops = [p for p in demand['pops'] if p['residenceId'] == home_id]
    for p in pops:
        if p['drivingSeconds'] > 0:
            # Already calculated - skip
            continue
        job_id = p['jobId']
        job_point = points_by_id[job_id]
        try:
            job_node = ox.nearest_nodes(G, Y=job_point['location'][1], X=job_point['location'][0])
            path_nodes = nx.shortest_path(G, home_node, job_node, weight='travel_time')
            distance_in_meters = nx.path_weight(G, path_nodes, weight='length')
            travel_time_in_seconds = nx.path_weight(G, path_nodes, weight='travel_time')
        except:
            try:
                # Find closest road segment and project a point onto it
                x, y = job_point['location']
                u, v, key = ox.nearest_edges(G, Y=y, X=x)
                edge_data = G[u][v][key]
                line = edge_data['geometry']
                point = Point(x, y)
                nearest_point = line.interpolate(line.project(point))
                new_node = max(G.nodes) + 1
                G.add_node(new_node, x=nearest_point.x, y=nearest_point.y)
                dist_to_u = Point(G.nodes[u]['x'], G.nodes[u]['y']).distance(nearest_point)
                dist_to_v = Point(G.nodes[v]['x'], G.nodes[v]['y']).distance(nearest_point)
                G.add_edge(new_node, u, length=dist_to_u)
                G.add_edge(new_node, v, length=dist_to_v)
                job_node = ox.nearest_nodes(G, X=x, Y=y)
                path_nodes = nx.shortest_path(G, home_node, job_node, weight='travel_time')
                distance_in_meters = nx.path_weight(G, path_nodes, weight='length')
                travel_time_in_seconds = nx.path_weight(G, path_nodes, weight='travel_time')
            except:
                path_nodes = []
                distance_in_meters = 0
                travel_time_in_seconds = 0
        # Add time penalties for intersections + traffic: 5 seconds per intersection
        travel_time_in_seconds += len(path_nodes) * 5
        
        p['drivingSeconds']  = int(travel_time_in_seconds)
        p['drivingDistance'] = int(np.ceil(distance_in_meters))
    return pops


def compute_centroid(coords, weights=None):
    if weights is None:
        weights = np.ones(len(coords))
    if not coords:
        return [0.0, 0.0]
    lon = float(weighted_mean([c[0] for c in coords], weights))
    lat = float(weighted_mean([c[1] for c in coords], weights))
    return [lon, lat]


def weighted_mean(values, weights):
    if not values:
        return 0.0
    w = np.array(weights, dtype=float)
    v = np.array(values, dtype=float)
    if w.sum() == 0:
        return float(np.mean(v))
    return float((v * w).sum() / w.sum())


def haversine_travel_time(lon1, lat1, lon2, lat2, kph=30):
    dist = U.haversine(lon1, lat1, lon2, lat2)
    speed_m_s = kph * km2m / hr2sec
    duration = dist / speed_m_s
    return dist, duration


def in_cbd(loc, cbd_bbox=None):
    if cbd_bbox is not None:
        if ((loc[0] >= cbd_bbox[0]) and (loc[1] >= cbd_bbox[1]) and \
            (loc[0] <= cbd_bbox[2]) and (loc[1] <= cbd_bbox[3])):
            return True
    return False


def merge_points(inps, loc_assignments, sorted_points, size_of_points, pops_by_id):
    ipoint, unique_loc = inps

    # Convert to NumPy array once
    loc_assignments = np.asarray(loc_assignments)
    mask = loc_assignments == ipoint

    # Extract points using boolean mask
    these_points = [sorted_points[i] for i in np.where(mask)[0]]

    # Pre-extract fields
    pids = [p['id'] for p in these_points]
    sizes = size_of_points[mask]

    # Pick representative ID
    max_idx = np.argmax(sizes)
    merged_id = these_points[max_idx]['id']
    if not merged_id.startswith('m_'):
        merged_id = 'm_' + merged_id

    # Vectorized centroid
    merged_loc = compute_centroid(
        [p['location'] for p in these_points],
        sizes
    )

    # Vectorized sums
    merged_jobs = sum(p['jobs'] for p in these_points)
    merged_residents = sum(p['residents'] for p in these_points)

    # Flatten popIds efficiently
    merged_popIds = np.unique(
        np.concatenate([p['popIds'] for p in these_points])
    ).tolist()

    # Update pops (only deep-copy those that need updating)
    updated_pops = []
    pid_set = set(pids)
    for popid in merged_popIds:
        p = pops_by_id[popid]
        needs_copy = p['residenceId'] in pid_set or p['jobId'] in pid_set
        if needs_copy:
            p = copy.deepcopy(p)
            if p['residenceId'] in pid_set:
                p['residenceId'] = merged_id
            if p['jobId'] in pid_set:
                p['jobId'] = merged_id
        updated_pops.append(p)

    merged_point = {
        "id": merged_id,
        "location": merged_loc,
        "jobs": merged_jobs,
        "residents": merged_residents,
        "popIds": merged_popIds
    }

    return merged_point, updated_pops
