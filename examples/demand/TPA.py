import sys, os
import numpy as np
from depot.demand import DemandData


# This can go in a separate config.json file (swap out any True -> true, False -> false), 
# then load it in this script. It's included here for clarity
config = {
    "bbox" : [-82.85791, 27.61136, -82.10097, 28.28610],
    "MAXPOPSIZE" : 200,
    "CALCULATE_ROUTES" : True,
    "ROUTING_METHOD": "osrm",
    
    "airport" : ["TPA_T1", "PIE_T1"],
    "air_type" : ["airport", "airport"],
    "airport_name" : ["Tampa International Airport", 
                      "St. Pete-Clearwater International Airport"],
    "airport_daily_passengers" : [20000, 3900], 
    "airport_loc" : [[-82.53474, 27.97930], [-82.69090, 27.90684]],
    "air_pop_size" : [200, 100],
    "air_merge_within": [350, 200],
    
    "universities" : ["USFT", "USFS", "UT", "HCCD", "HCCB", "HCCY", "HCCP", "HCCS", "STET"],
    "univ_name" : ["University of South Florida, Tampa Campus", 
                   "University of South Florida, St. Petersburg Campus",
                   "University of Tampa", 
                   "Hillsborough Community College, Dale Mabry Campus",
                   "Hillsborough Community College, Brandon Campus",
                   "Hillsborough Community College, Ybor City Campus",
                   "Hillsborough Community College, Plant City Campus",
                   "Hillsborough Community College, SouthShore Campus",
                   "Stetson University, Tampa Law Center"],
    "univ_type" : ["university", "university", "university", "community_college", 
                   "community_college", "community_college", "community_college", 
                   "community_college", "university"],
    "univ_max_dist" : [30000, 30000, 20000, 40000, 40000, 40000, 40000, 40000, 
                       20000],
    "univ_loc" : [[-82.41223, 28.05957], [-82.63580, 27.76268], [-82.46421, 27.94671], 
                  [-82.50959, 27.97606], [-82.32867, 27.96938], [-82.44468, 27.96269], 
                  [-82.10304, 28.02609], [-82.40024, 27.72210], [-82.71764, 27.75662]],
    "univ_merge_within" : [350, 250, 300, 300, 250, 200, 200, 200, 150],
    "students" : [40000, 4700, 11400, 7000, 5000, 3500, 2500, 2000, 900],
    "perc_oncampus" : [0.1875, 0.19, 0.4, 0, 0, 0, 0, 0, 0.056],
    "univ_pop_size" : [200, 100, 200, 200, 200, 100, 100, 100, 50],
    "univ_perc_travel" : [0.3, 0.9],
    
    "entertainment" : ["AA", "RJ", "TROP", "BG", "AI", "AQUA", "ZOO", "DALI",
                       "CB", "FDSP", "SPB", "PB", "HISP", "TIB", "MB", "BTDB",
                       "TR1", "TR2", "TR3", "FHP", "CVP", "SKP", "WP", "LSP", "PP",
                       "JCSP", "LLP", "HRSP", "PIP", "EMCP",
                       "IPBS", "BE", "TPO", "TS", "TSW", "CM", "HPV"],
    "ent_name" : ["Benchmark International Arena", "Raymond James Stadium", 
                  "Tropicana Field", "Busch Gardens", "Adventure Island", 
                  "Florida Aquarium", "Lowry Park Zoo", "Dali Museum", 
                  "Clearwater Beach", "Fort De Soto Park", "St. Pete Beach", 
                  "Pass-a-Grille Beach", "Honeymoon Island State Park", 
                  "Treasure Island Beach", "Madeira Beach", 
                  "Ben T. Davis Beach", "Tampa Riverwalk", "Tampa Riverwalk", 
                  "Tampa Riverwalk", "Fred Howard Park", 
                  "Carrollwood Village Park", "Sand Key Park", 
                  "Walsingham Park", "Lake Seminole Park", "Philippe Park", 
                  "John Chesnut Sr. Park", "Lettuce Lake Conservation Park", 
                  "Hillsborough River State Park", "Picnic Island Park", 
                  "Edward Medard Conservation Park", 
                  "International Plaza and Bay Street", "Brandon Exchange", 
                  "Tampa Premium Outlets", "Tyrone Square", 
                  "The Shops at Wiregrass", "Countryside Mall", 
                  "Hyde Park Village"],
    "ent_type" : ["arena", "stadium", "baseball_field", "theme_park", 
                  "theme_park", "aquarium", "zoo", "art_museum", "beach",
                  "park", "beach", "beach", "park", "beach", "beach", "beach",
                  "park", "park", "park", "park", "park", "park", "park", 
                  "park", "park", "park", "nature_park", "nature_park", 
                  "park", "nature_park", "shopping_center", 
                  "shopping_center", "shopping_center", "shopping_center", 
                  "shopping_center", "shopping_center", "shopping_center"],
    "ent_max_dist" : [30000, 30000, 30000, 40000, 40000, 30000, 30000, 30000,
                      None, 50000, None, 50000, 50000, 50000, 50000, 30000,
                      25000, 25000, 25000, 30000, 20000, 30000, 20000, 20000,
                      20000, 20000, 20000, 35000, 35000, 25000, 30000, 30000, 
                      30000, 30000, 30000, 30000, 30000],
    "ent_loc" : [
        [-82.45178, 27.94273], [-82.50333, 27.97602],
        [-82.65329, 27.76819], [-82.42120, 28.03371],
        [-82.41249, 28.04246], [-82.44488, 27.94397],
        [-82.46993, 28.01353], [-82.63146, 27.76600], 
        [-82.82978, 27.97734], [-82.71864, 27.63392], 
        [-82.74204, 27.72110], [-82.73805, 27.69618], 
        [-82.83097, 28.06383], [-82.77430, 27.77168], 
        [-82.79767, 27.79706], [-82.57929, 27.97073], 
        [-82.45883, 27.94479], [-82.45819, 27.94237], 
        [-82.45514, 27.94036], [-82.78981, 28.15591], 
        [-82.52520, 28.07222], [-82.82775, 27.95914], 
        [-82.81366, 27.86862], [-82.77549, 27.84113], 
        [-82.67845, 28.00903], [-82.70149, 28.08862], 
        [-82.37381, 28.07273], [-82.22692, 28.14310], 
        [-82.54662, 27.86096], [-82.16816, 27.92339], 
        [-82.52058, 27.96458], [-82.32577, 27.93220], 
        [-82.39062, 28.19035], [-82.73338, 27.79381], 
        [-82.34933, 28.18871], [-82.73332, 28.01717], 
        [-82.47561, 27.93639]
    ],
    "ent_size" : [1700, 4000, 4400, 10800, 1750, 2700, 3200, 850,
                  11000, 7400, 6800, 2700, 2700, 2000, 2000, 1100,
                  3000, 3000, 3000, 5400, 2700, 2700, 2700, 2700, 2400,
                  2400, 1650, 825, 675, 1000,
                  41000, 32800, 22000, 16400, 15000, 13600, 9600],
    "ent_pop_size" : [50, 200, 200, 200, 50, 100, 100, 50,
                      200, 200, 200, 100, 100, 100, 100, 50,
                      200, 200, 200, 200, 100, 100, 100, 100, 75,
                      75, 50, 25, 25, 50,
                      200, 200, 200, 200, 200, 200, 200],
    "ent_merge_within" : [150, 200, 200, 250, 150, 150, 200, 100, 
                          300, 250, 250, 150, 150, 150, 150, 100,
                          150, 150, 150, 200, 150, 150, 150, 150, 150,
                          150, 150, 100, 100, 150,
                          400, 400, 350, 300, 250, 200, 200],
    
    "bases" : ["MAFB", "USCGC", 
               "USCGSP"],
    "base_name" : ["MacDill Air Force Base", 
                   "US Coast Guard Air Station Clearwater", 
                   "US Coast Guard Sector St. Petersburg"],
    "base_max_dist" : [30000, 20000, 20000],
    "base_loc" : [[-82.48706, 27.85753], [-82.69797, 27.91156],
                  [-82.63087, 27.75716]],
    "personnel" : [16800, 700,
                   850],
    "perc_onbase" : [0.172, 0.0,
                     0.0],
    "base_pop_size" : [200, 50,
                       50],
    "base_merge_within" : [350, 150, 150],
    "base_perc_travel" : [0.4, 1.0]
}

# LODES demand data
tpa = DemandData('TPA/demand_data.json', 'TPA')

### Add special demand
# Airports
for iair in range(len(config['airport'])):
    new_point = {
            "type" : config['air_type'][iair],
            "name" : config['airport_name'][iair],
            "code" : config['airport'][iair],
            "location" : config['airport_loc'][iair],
            "total_capacity" : config['airport_daily_passengers'][iair],
            "pop_size" : config['air_pop_size'][iair],
            "merge_within" : config['air_merge_within'][iair]
    }
    tpa.add_points(new_point)

# Universities
for iuniv in range(len(config['universities'])):
    # Calculate modeled size and residential split
    univ_size_oncampus = config['students'][iuniv] * \
                         config['perc_oncampus'][iuniv] * \
                         config['univ_perc_travel'][0]
    univ_size_offcampus = config['students'][iuniv] * \
                          (1 - config['perc_oncampus'][iuniv]) * \
                          config['univ_perc_travel'][1]
    univ_size_modeled = univ_size_oncampus + univ_size_offcampus
    
    new_point = {
            "type" : config['univ_type'][iuniv],
            "name" : config['univ_name'][iuniv],
            "code" : config['universities'][iuniv],
            "location" : config['univ_loc'][iuniv],
            "total_capacity" : univ_size_modeled,
            "pop_size" : config['univ_pop_size'][iuniv],
            "merge_within" : config['univ_merge_within'][iuniv],
            "residential_split" : univ_size_oncampus / univ_size_modeled,
    }
    if config['univ_max_dist'][iuniv] is not None:
        new_point['max_distance'] = config['univ_max_dist'][iuniv]
    tpa.add_points(new_point)

# Entertainment
for ient in range(len(config['entertainment'])):
    new_point = {
            "type" : config['ent_type'][ient],
            "name" : config['ent_name'][ient],
            "code" : config['entertainment'][ient],
            "location" : config['ent_loc'][ient],
            "total_capacity" : config['ent_size'][ient],
            "pop_size" : config['ent_pop_size'][ient],
            "merge_within" : config['ent_merge_within'][ient]
    }
    if config['ent_max_dist'][ient] is not None:
        new_point['max_distance'] = config['ent_max_dist'][ient]
    tpa.add_points(new_point)

# Military bases
for ibase in range(len(config['bases'])):
    # Calculate modeled size and residential split
    onbase_size = config['personnel'][ibase] * \
                  config['perc_onbase'][ibase] * \
                  config['base_perc_travel'][0]
    offbase_size = config['personnel'][ibase] * \
                   (1 - config['perc_onbase'][ibase]) * \
                   config['base_perc_travel'][1]
    modeled_size = onbase_size + offbase_size
    new_point = {
            "type" : "military_base",
            "name" : config['base_name'][ibase],
            "code" : config['bases'][ibase],
            "location" : config['base_loc'][ibase],
            "total_capacity" : modeled_size,
            "pop_size" : config['base_pop_size'][ibase],
            "merge_within" : config['base_merge_within'][ibase],
            "residential_split" : onbase_size / modeled_size,
    }
    if config['base_max_dist'][ibase] is not None:
        new_point['max_distance'] = config['base_max_dist'][ibase]
    tpa.add_points(new_point)

tpa.calculate_routes(config['ROUTING_METHOD'], config['bbox'])
tpa.print_stats()

tpa.save('TPA/updated_demand_data.json')

tpa.create_config(name="Tampa Bay", bbox=config['bbox'],
                  description="Bring rapid transit to the Big Guava.", 
                  creator="slurry", version="1.3.0", country="US")

tpa.create_description('tampa-bay', ["""<li><a href="https://github.com/rslurry/subwaybuilder-US-demand-data">US Demand Generator</a></li>""", """<li><a href="https://github.com/Subway-Builder-Modded/depot">Depot</a></li>"""], ["""<li><a href="https://lehd.ces.census.gov/data/">United States Census Bureau Longitudinal Employer-Household Dynamics Origin-Destination Employment Statistics</a></li>"""])

