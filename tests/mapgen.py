import sys, os
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from depot.maps import MapGen

class TestMapGen(unittest.TestCase):

    ##### city validation tests #####

    @patch('os.path.exists')
    def test_city_valid(self, mock_exists):
        """Test valid 2, 3, and 4 character city codes."""
        for code in ["LA", "SF4", "NYC1", "la", "sF4"]:
            mg = MapGen(code, [-118.5, 33.9, -118.4, 34.0], 
                        osmpbf="dummy.osm.pbf", verb=False)
            self.assertEqual(mg.city, code.upper())

    def test_city_invalid_start(self):
        """Fail if first two chars are not letters."""
        with self.assertRaises(ValueError):
            MapGen("1A", [-118, 33, -117, 34])
        with self.assertRaises(ValueError):
            MapGen("A1B", [-118, 33, -117, 34])

    def test_city_invalid_length(self):
        """Fail if length is < 2 or > 4."""
        with self.assertRaises(ValueError):
            MapGen("A", [-118, 33, -117, 34])
        with self.assertRaises(ValueError):
            MapGen("ABCDE", [-118, 33, -117, 34])

    def test_city_invalid_chars(self):
        """Fail if characters 3-4 are special symbols."""
        with self.assertRaises(ValueError):
            MapGen("LA!", [-118, 33, -117, 34])

    ##### bbox validation tests #####
    
    @patch('os.path.exists')
    def test_bbox_valid_formats(self, mock_exists):
        """Test list, tuple, and numpy array inputs."""
        valid_data = [-118.5, 33.9, -118.4, 34.0]
        
        # Test List
        mg_list = MapGen("LAX", valid_data, osmpbf="dummy.osm.pbf", verb=False)
        self.assertEqual(mg_list.bbox, valid_data)
        
        # Test Numpy Array
        mg_np = MapGen("LAX", np.array(valid_data), osmpbf="dummy.osm.pbf", 
                       verb=False)
        self.assertEqual(mg_np.bbox, valid_data)

    @patch('os.path.exists')
    def test_bbox_invalid_order(self, mock_exists):
        """Fail if min >= max for lon or lat."""
        # min_lon > max_lon
        with self.assertRaises(ValueError):
            MapGen("LAX", [-110, 33, -118, 34], osmpbf="dummy.osm.pbf")
        # min_lat > max_lat
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118, 35, -117, 34], osmpbf="dummy.osm.pbf")

    @patch('os.path.exists')
    def test_bbox_invalid_length(self, mock_exists):
        """Fail if bbox does not have exactly 4 values."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118, 33, -117], osmpbf="dummy.osm.pbf")
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118, 33, -117, 34, 0], osmpbf="dummy.osm.pbf")

    @patch('os.path.exists')
    def test_bbox_non_numeric(self, mock_exists):
        """Fail if bbox contains strings."""
        with self.assertRaises(TypeError):
            MapGen("LAX", [-118, "33", -117, 34], osmpbf="dummy.osm.pbf")

    @patch('os.path.exists')
    def test_bbox_type_error(self, mock_exists):
        """Fail if bbox is not a collection."""
        with self.assertRaises(TypeError):
            MapGen("LAX", -118.5, osmpbf="dummy.osm.pbf")

    ##### osmpbf tests #####

    def test_osmpbf_none(self):
        """Passing None for osmpbf should raise a ValueError."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf=None)

    def test_osmpbf_invalid_type(self):
        """Fail if osmpbf is not a string (and not None)."""
        with self.assertRaises(TypeError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf=123)

    def test_osmpbf_nonexistent_path(self):
        """Fail if the path does not exist on the file system."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                   osmpbf="/tmp/non_existent_file.pbf")
    
    ##### ncores tests #####
    
    @patch('os.path.exists')
    def test_ncores_posint(self, mock_exists):
        """Positive integers are allowed for ncores."""
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf", ncores=1, verb=False)
        self.assertEqual(mg.ncores, 1)
    
    @patch('os.path.exists')
    def test_ncores_none(self, mock_exists):
        """None is allowed for ncores."""
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf", ncores=None, verb=False)
        self.assertIsNone(mg.ncores)
    
    @patch('os.path.exists')
    def test_ncores_toobig(self, mock_exists):
        """Reduce ncores to os.cpu_count() if it is larger."""
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf", ncores=9999, verb=False)
        self.assertEqual(mg.ncores, os.cpu_count())
    
    @patch('os.path.exists')
    def test_ncores_zero_fails(self, mock_exists):
        """Fail if ncores is zero."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   ncores=0)
    
    @patch('os.path.exists')
    def test_ncores_negint_fails(self, mock_exists):
        """Fail if ncores is a negative integer."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   ncores=-1)
    
    @patch('os.path.exists')
    def test_ncores_float_fails(self, mock_exists):
        """Fail if ncores is a float."""
        with self.assertRaises(TypeError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   ncores=0.5)
    
    @patch('os.path.exists')
    def test_ncores_str_fails(self, mock_exists):
        """Fail if ncores is a string."""
        with self.assertRaises(TypeError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   ncores="4")
    
    ##### RAM tests #####
    
    @patch('os.path.exists')
    def test_ram_positive_int(self, mock_exists):
        """Standard positive integer GB should convert to MB."""
        # 8 GB should become 8000 MB
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf", RAM=8, verb=False)
        self.assertEqual(mg.RAM, 8000)
    
    @patch('os.path.exists')
    def test_ram_positive_float(self, mock_exists):
        """Positive floats should be converted and cast to int."""
        # 4.5 GB should become 4500 MB
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf", RAM=4.5, verb=False)
        self.assertEqual(mg.RAM, 4500)
    
    @patch('os.path.exists')
    def test_ram_zero_fails(self, mock_exists):
        """RAM of 0 should raise a ValueError."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   RAM=0)
    
    @patch('os.path.exists')
    def test_ram_negative_fails(self, mock_exists):
        """Negative RAM should raise a ValueError."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   RAM=-4)

    @patch('os.path.exists')
    def test_ram_minimum_floor(self, mock_exists):
        """RAM of exactly 1 GB should be allowed (the floor)."""
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf",RAM=1, verb=False)
        self.assertEqual(mg.RAM, 1000)

    @patch('os.path.exists')
    def test_ram_below_floor_fails(self, mock_exists):
        """RAM below 1 GB (e.g., 0.5 GB) should raise a ValueError."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   RAM=0.5)

    @patch('os.path.exists')
    def test_ram_string_input_fails(self, mock_exists):
        """
        Passing a string should raise a TypeError during comparison or 
        multiplication.
        """
        with self.assertRaises(TypeError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   RAM="8GB")
    
    ##### places (cities, suburbs, neighborhoods) tests #####
    
    @patch('os.path.exists')
    def test_places_standard_lists(self, mock_exists):
        """Standard lists of strings should be stored correctly."""
        cities = ['city', 'town']
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf", cities=cities, verb=False)
        self.assertEqual(mg.cities, cities)
    
    @patch('os.path.exists')
    def test_places_none(self, mock_exists):
        """None should be allowed to disable labels."""
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    osmpbf="dummy.osm.pbf", cities=None, verb=False)
        self.assertIsNone(mg.cities)
    
    @patch('os.path.exists')
    def test_places_string_fails(self, mock_exists):
        """Passing a single string should raise a TypeError."""
        with self.assertRaises(TypeError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   cities="city")
    
    @patch('os.path.exists')
    def test_places_empty_list_fails(self, mock_exists):
        """An empty list should raise a ValueError."""
        with self.assertRaises(ValueError) as cm:
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
                   cities=[])
        self.assertIn("cannot be an empty list", str(cm.exception))
    
    @patch('os.path.exists')
    def test_places_mixed_list_fails(self, mock_exists):
        """A list containing non-string types should raise a TypeError."""
        # Mixing strings with integers or other types
        mixed_list = ['city', 123, None]
        with self.assertRaises(TypeError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], osmpbf="dummy.osm.pbf",
            cities=mixed_list)
    
    ##### Environment tests #####
    
    @patch('shutil.which')
    def test_missing_binary(self, mock_which):
        mock_which.return_value = None  # Simulate tool not found
        with self.assertRaises(RuntimeError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0])
    
    ##### Building filter validation tests #####

    @patch('os.path.exists')
    def test_building_filter_logic_valid(self, mock_exists):
        """
        Test that building_tile_filter_size defaults correctly and 
        respects hierarchies.
        """
        # Case: Defaults (should equal building_index_filter_size)
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    building_index_filter_size=100, 
                    osmpbf="dummy.osm.pbf", verb=False)
        self.assertEqual(mg.building_index_filter_size, 100)
        self.assertEqual(mg.building_tile_filter_size, 100)

        # Case: Custom hierarchy (150 >= 50)
        mg_custom = MapGen("NYC", [-74.1, 40.6, -73.9, 40.8], 
                           building_index_filter_size=150, 
                           building_tile_filter_size=50, 
                           osmpbf="dummy.osm.pbf", verb=False)
        self.assertEqual(mg_custom.building_tile_filter_size, 50)

    @patch('os.path.exists')
    def test_building_filter_logic_invalid(self, mock_exists):
        """
        Fail if building_tile_filter_size is smaller than 
        building_index_filter_size.
        """
        # building_index_filter_size < building_tile_filter_size
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                   building_index_filter_size=100, 
                   building_tile_filter_size=150, 
                   osmpbf="dummy.osm.pbf")

    ##### Building simplifications #####
    
    @patch('os.path.exists')
    def test_building_simplification_logic_valid(self, mock_exists):
        """
        Test that building_index_simplification and 
        building_tile_simplification assign correctly.
        """
        # Case: int
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    building_index_simplification=2, 
                    building_tile_simplification=2,
                    osmpbf="dummy.osm.pbf", verb=False)
        self.assertEqual(mg.building_index_simplification, 2)
        self.assertEqual(mg.building_tile_simplification, 2)

        # Case: float
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                    building_index_simplification=1.5, 
                    building_tile_simplification=1.5,
                    osmpbf="dummy.osm.pbf", verb=False)
        self.assertEqual(mg.building_index_simplification, 1.5)
        self.assertEqual(mg.building_tile_simplification, 1.5)

    @patch('os.path.exists')
    def test_building_index_simplification_invalid_negative(self, mock_exists):
        """Fail if building_index_simplification is negative."""
        # building_index_filter_size < building_tile_filter_size
        with self.assertRaises(ValueError):
            mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                        building_index_simplification=-1, 
                        building_tile_simplification=1,
                        osmpbf="dummy.osm.pbf", verb=False)

    @patch('os.path.exists')
    def test_building_tile_simplification_invalid_negative(self, mock_exists):
        """Fail if building_tile_simplification is negative."""
        # building_index_filter_size < building_tile_filter_size
        with self.assertRaises(ValueError):
            mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                        building_index_simplification=1, 
                        building_tile_simplification=-1,
                        osmpbf="dummy.osm.pbf", verb=False)

    @patch('os.path.exists')
    def test_building_index_simplification_invalid_str(self, mock_exists):
        """Fail if building_index_simplification is a string."""
        # building_index_filter_size < building_tile_filter_size
        with self.assertRaises(TypeError):
            mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                        building_index_simplification="1", 
                        building_tile_simplification=1,
                        osmpbf="dummy.osm.pbf", verb=False)

    @patch('os.path.exists')
    def test_building_tile_simplification_invalid_str(self, mock_exists):
        """Fail if building_tile_simplification is a string."""
        # building_index_filter_size < building_tile_filter_size
        with self.assertRaises(TypeError):
            mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                        building_index_simplification=1, 
                        building_tile_simplification="1",
                        osmpbf="dummy.osm.pbf", verb=False)

    ##### Directory and path properties #####
    
    @patch('os.path.exists')
    def test_outputdir_valid(self, mock_exists):
        """Should accept valid directories."""
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], outputdir="/tmp",
                    osmpbf="dummy.osm.pbf", verb=False)
        self.assertEqual(mg.outputdir, "/tmp")

    @patch('os.path.exists')
    def test_outputdir_empty(self, mock_exists):
        """Empty string should default to current directory (.)."""
        mg = MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], outputdir="",
                    osmpbf="dummy.osm.pbf", verb=False)
        self.assertEqual(mg.outputdir, ".")

    def test_outputdir_invalid(self):
        """Should fail if path is a file or doesn't exist."""
        with self.assertRaises(ValueError):
            MapGen("LAX", [-118.5, 33.9, -118.4, 34.0], 
                   outputdir="/not/a/real/path")
    
    @patch('os.path.exists')
    def test_city_dir_property(self, mock_exists):
        """Ensure city_dir is dynamically constructed correctly."""
        mg = MapGen("sf", [-122.5, 37.7, -122.3, 37.8], outputdir="/tmp",
                    osmpbf="dummy.osm.pbf", verb=False)
        # Note: city setter uppercases 'sf' to 'SF'
        expected_path = os.path.join("/tmp", "SF")
        self.assertEqual(mg.city_dir, expected_path)

    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_directory_creation(self, mock_exists, mock_makedirs):
        """Test that the class attempts to create the directory on init."""
        mg = MapGen("TEST", [-118.5, 33.9, -118.4, 34.0],
                    osmpbf="dummy.osm.pbf", verb=False)
        mock_makedirs.assert_called()

    ##### ocean depth tile tests #####

    def test_ocean_depth_tiles_without_contours_file(self):
        """Fall back to the depth index when no contours file sits next to it."""
        with tempfile.TemporaryDirectory() as outputdir:
            with patch('os.path.exists'):
                mg = MapGen("TEST", [-118.5, 33.9, -118.4, 34.0], outputdir=outputdir,
                            osmpbf="dummy.osm.pbf", verb=False)
            mg.fdepths_contours = os.path.join(mg.city_dir,
                                               "ocean_depth_index_contours.json.gz")
            mg.bathy_data = {"depths": [{"b": [0, 0, 1, 1], "d": -5,
                                         "p": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}]}
            os.makedirs(mg.city_dir, exist_ok=True)
            with patch.object(MapGen, "_run_command"):
                mg._generate_ocean_depth_tiles()
            with open(mg.ocean_foundations_geojson) as f:
                features = json.load(f)["features"]

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["properties"]["depth_min"], -5)

if __name__ == "__main__":
    unittest.main()
