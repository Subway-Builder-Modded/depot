import sys, os
import unittest
from unittest.mock import patch
import numpy as np
from depot.demand import DemandData

class TestDemandData(unittest.TestCase):
    def test_missing_map_code(self):
        with self.assertRaises(TypeError):
            DemandData("test_demand_data.json")
    
    ### Loading/saving ###
    
    def test_invalid_demand_file_type(self):
        """ Fails if a non-string is given """
        with self.assertRaises(ValueError):
            DemandData(None, 'TEST')
        
        with self.assertRaises(ValueError):
            DemandData(10, 'TEST')
        
        with self.assertRaises(ValueError):
            DemandData(["something"], 'TEST')
    
    def test_invalid_demand_file_path(self):
        """ Fails if provided file path does not exist """
        with self.assertRaises(FileNotFoundError):
            DemandData("some/path/to/demand_data.json", 'TEST')
    
    def test_valid_demand_file_path(self):
        """
        Loads the example demand data file and checks that everything 
        is loaded properly.
        """
        obj = DemandData('test_demand_data.json', 'TEST')
        assert 'pops'   in obj.keys() and \
               'points' in obj.keys() and \
               isinstance(obj['pops'  ], list) and \
               isinstance(obj['points'], list)

    def test_special_demand_schema_loaded(self):
        """ Checks that the special demand schema was loaded """
        obj = DemandData('test_demand_data.json', 'TEST')
        assert isinstance(obj.special_demand_exp, dict) and \
               isinstance(obj.special_demand_codes, dict) and \
               isinstance(obj.special_demand_descs, dict) and \
               isinstance(obj.special_demand_ids, dict) and \
               len(obj.special_demand_exp.keys()) and \
               len(obj.special_demand_codes.keys()) and \
               len(obj.special_demand_descs.keys()) and \
               len(obj.special_demand_ids.keys()) and \
               obj.special_demand_codes.keys() == obj.special_demand_descs.keys()

    def test_move_points_by_ids(self):
        """ Tests move_points() by point IDs """
        obj = DemandData('test_demand_data.json', 'TEST')
        # Test moving single point
        tgt_pt = obj['points'][0]
        tgt_id = tgt_pt['id']
        tgt_coords = tgt_pt['location']
        new_coords = [v/2. for v in tgt_coords]
        obj.move_points(point_ids=tgt_id, new_coords=new_coords)
        assert obj['points'][0]['location'] == new_coords
        # Test moving list of points
        tgt_pts = obj['points'][1:4]
        tgt_ids = [p['id'] for p in tgt_pts]
        tgt_coords = [p['location'] for p in tgt_pts]
        new_coords = [[v/2. for v in coords] for coords in tgt_coords]
        obj.move_points(point_ids=tgt_ids, new_coords=new_coords)
        for i in range(len(tgt_pts)):
            assert obj['points'][1+i]['location'] == new_coords[i] 
    
    def test_move_points_by_locs(self):
        """ Tests move_points() by coordinates """
        obj = DemandData('test_demand_data.json', 'TEST')
        # Test moving single point
        tgt_pt = obj['points'][0]
        tgt_coords = [v + 0.001 for v in tgt_pt['location']]
        new_coords = [v/2. for v in tgt_coords]
        obj.move_points(coords=tgt_coords, new_coords=new_coords)
        assert obj['points'][0]['location'] == new_coords
        # Test moving list of points
        tgt_pts = obj['points'][1:4]
        tgt_coords = [[v + 0.001 for v in p['location']] for p in tgt_pts]
        new_coords = [[v/2. for v in coords] for coords in tgt_coords]
        obj.move_points(coords=tgt_coords, new_coords=new_coords)
        for i in range(len(tgt_pts)):
            assert obj['points'][1+i]['location'] == new_coords[i] 
    
    def test_del_point_by_ids(self):
        """ Tests del_points() by point IDs """
        obj = DemandData('test_demand_data.json', 'TEST')
        # Test deleting single point
        tgt_pt = obj['points'][0]
        tgt_id = tgt_pt['id']
        obj.del_points(point_ids=tgt_id)
        assert all([tgt_id not in p['id'] for p in obj['points']])
        # Test deleting list of points
        tgt_pts = obj['points'][1:4]
        tgt_ids = [p['id'] for p in tgt_pts]
        obj.del_points(point_ids=tgt_ids)
        for tgt_id in tgt_ids:
            assert all([tgt_id not in p['id'] for p in obj['points']])
    
    def test_del_point_by_locs(self):
        """ Tests del_points() by coordinates """
        obj = DemandData('test_demand_data.json', 'TEST')
        # Test deleting single point
        tgt_pt = obj['points'][0]
        tgt_id = tgt_pt['id']
        tgt_coords = [v + 0.001 for v in tgt_pt['location']]
        obj.del_points(coords=tgt_coords)
        assert all([tgt_id not in p['id'] for p in obj['points']])
        # Test deleting list of points
        tgt_pts = obj['points'][1:4]
        tgt_ids = [p['id'] for p in tgt_pts]
        tgt_coords = [[v + 0.001 for v in p['location']] for p in tgt_pts]
        obj.del_points(coords=tgt_coords)
        for tgt_id in tgt_ids:
            assert all([tgt_id not in p['id'] for p in obj['points']])
    
    def test_add_points_fail(self):
        """
        Test adding a point where no other points are within the maximum 
        allowed distance (hard-capped at 200 km).
        """
        obj = DemandData('test_demand_data.json', 'TEST', verb=False)
        # Create a point far away from the rest of the map
        new_point = {
            "type" : "events",
            "name" : "Test Failure Point",
            "code" : 'TEST',
            "location" : [0.0, 0.0],
            "total_capacity" : 200,
            "pop_size" : 200,
            "max_distance" : 1
        }
        with self.assertRaises(ValueError):
            obj.add_points(new_point)
    
    def test_add_points_single_jobs(self):
        """ Tests add_points() for a single job point """
        obj = DemandData('test_demand_data.json', 'TEST', verb=False)
        # Place point in the middle of the map, tons of small pops, so 
        # one pop will go to each other point
        locs = []
        for p in obj['points']:
            locs.append(p['location'])
        tgt_loc = np.mean(locs, axis=0).tolist()
        new_point = {
            "type" : "airport",
            "name" : "Test Point 1",
            "code" : "TEST1",
            "location" : tgt_loc,
            "total_capacity" : 10000,
            "pop_size" : 1,
            "max_distance" : 100000,
            "residential_split" : 0.0
        }
        obj.add_points(new_point)
        # All other points should have at least 1 resident pop from this point
        for p in obj['points']:
            assert any(new_point['code'] in popid or p['residents'] == 0 \
                       for popid in p['popIds'])

    def test_add_points_single_residents(self):
        """ Tests add_points() for a single residence point """
        obj = DemandData('test_demand_data.json', 'TEST', verb=False)
        # Place point in the middle of the map, tons of small pops, so 
        # one pop will go to each other point
        locs = []
        for p in obj['points']:
            locs.append(p['location'])
        tgt_loc = np.mean(locs, axis=0).tolist()
        new_point = {
            "type" : "airport",
            "name" : "Test Point 2",
            "code" : "TEST2",
            "location" : tgt_loc,
            "total_capacity" : 10000,
            "pop_size" : 1,
            "max_distance" : 100000,
            "residential_split" : 1.0
        }
        obj.add_points(new_point)
        # All other points should have at least 1 job pop from this point
        for p in obj['points']:
            assert any(new_point['code'] in popid or p['jobs'] == 0 \
                       for popid in p['popIds'])
       
    def test_add_points_list(self):
        """ Tests add_points() for multiple points """
        obj = DemandData('test_demand_data.json', 'TEST', verb=False)
        # Place point in the middle of the map, tons of small pops, so 
        # one pop will go to each other point
        locs = []
        for p in obj['points']:
            locs.append(p['location'])
        tgt_loc = np.mean(locs, axis=0).tolist()
        new_points = []
        for i in range(3):
            new_points.append({
                "type" : "airport",
                "name" : f"Test Point {i}",
                "code" : f"TEST{i}",
                "location" : tgt_loc,
                "total_capacity" : 10000,
                "pop_size" : 1,
                "max_distance" : 100000,
                "residential_split" : 0.0
            })
        obj.add_points(new_points)
        # All other points should have at least 1 job pop from this point
        for p in obj['points']:
            if 'TEST' in p['id']:
                continue
            if p['residents'] == 0:
                continue
            for new_point in new_points:
                assert any(new_point['code'] in popid for popid in p['popIds'])
    
    def test_enforce_max_pop_size(self, big_pop_size=1000, max_pop_size=200):
        """ Ensures that enforce_max_pop_size properly splits pops """
        obj = DemandData('test_demand_data.json', 'TEST', verb=False)
        # Create a new pop that has a large size
        pop = {
            "id": 'test',
            "residenceId": obj['points'][0]['id'],
            "jobId": obj['points'][1]['id'],
            "size": big_pop_size,
            "drivingSeconds": 0,
            "drivingDistance": 0
        }
        obj['pops'].append(pop)
        obj['points'][0]['popIds'].append(pop['id'])
        obj['points'][1]['popIds'].append(pop['id'])
        obj['points'][0]['residents'] += pop['size']
        obj['points'][1]['jobs']      += pop['size']
        
        obj.enforce_max_pop_size(max_pop_size)
        # Largest pop should now be max_pop_size
        assert np.amax([p['size'] for p in obj['pops']]) == max_pop_size
        # There should be ceil(big_pop_size/max_pop_size) pops w/ ID that start with 'test'
        assert np.sum([p['id'].startswith('test') for p in obj['pops']]) == \
               np.ceil(big_pop_size/max_pop_size)
        # The points should also have this many pops that start with 'test'
        assert np.sum([p.startswith('test') for p in obj['points'][0]['popIds']]) == \
               np.ceil(big_pop_size/max_pop_size)
        assert np.sum([p.startswith('test') for p in obj['points'][1]['popIds']]) == \
               np.ceil(big_pop_size/max_pop_size)

if __name__ == "__main__":
    unittest.main()
