import sys, os
import numpy as np
from depot.maps import MapGen
import requests

# Download the .osm.pbf file we need
if not os.path.exists('socal-latest.osm.pbf'):
    url = 'https://download.geofabrik.de/north-america/us/california/socal-latest.osm.pbf'
    print("Downloading", url, flush=True)
    resp = requests.get('https://download.geofabrik.de/north-america/us/california/socal-latest.osm.pbf')
    with open('socal-latest.osm.pbf', 'wb') as f:
        f.write(resp.content)

# Set up MapGen object
obj = MapGen(city='LAXM', bbox=[-118.66816, 33.70200, -117.96758, 34.33162],
             osmpbf='socal-latest.osm.pbf',
             building_index_filter_size=180, building_tile_filter_size=50, 
             building_index_simplification=4,
             building_tile_simplification=1,
             ncores=16, RAM=8, 
             cities = ['city', 'borough', 'town'],
             suburbs = ['suburb', 'village'],
             neighborhoods = ['neighbourhood', 'hamlet', 'quarter', 'locality'])

# If you want to run each method individually:
#obj.extract_base_data()
#obj.process_buildings()
#obj.process_roads_and_aeroways()
#obj.generate_pmtiles()
#obj.add_labels()

# Or just run them all consecutively:
obj.run_all()
