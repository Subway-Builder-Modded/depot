import sys, os
import numpy as np
from depot.maps import MapGen
import requests

# Download the .osm.pbf file we need
if not os.path.exists('finland-latest.osm.pbf'):
    url = 'https://download.geofabrik.de/europe/finland-latest.osm.pbf'
    print("Downloading", url, flush=True)
    resp = requests.get(url)
    with open('finland-latest.osm.pbf', 'wb') as f:
        f.write(resp.content)

# Set up MapGen object
obj = MapGen(city='HEL', bbox=[24.5,60.1,25.3,60.4],
             osmpbf='finland-latest.osm.pbf',
             building_index_filter_size=10, building_tile_filter_size=10, 
             building_index_simplification=1, building_tile_simplification=1, 
             ncores=16, RAM=4, cleanup_files=False,
             cities = ['city', 'borough', 'town'],
             suburbs = ['city', 'borough', 'town', 'village'],
             neighborhoods = ['suburb', 'quarter', 'village'])

# If you want to run each method individually:
obj.extract_base_data()
obj.process_buildings()
obj.process_roads_and_aeroways()
obj.generate_pmtiles()
obj.add_labels()

# Or just run them all consecutively:
#obj.run_all()
