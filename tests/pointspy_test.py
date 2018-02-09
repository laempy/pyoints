import unittest
from .. import pointspy
#from pointpy import *
import numpy as np



class test_pointpy(unittest.TestCase):


    def test_version(self):
        self.assertTrue(hasattr(pointspy,'__version__'))
        self.assertTrue(float(pointspy.__version__)>0)


class test_extent(unittest.TestCase):


    def test_extent(self):
        
        points = [(0,0,0),(1,2,4),(0,1,0),(1,0.5,0)]
                    
        ext = pointspy.Extent(np.array(points)[:,0:2])
        self.assertEqual(ext.dim,2)
        self.assertTrue(np.array_equal(ext.ranges,[1,2]))
        self.assertTrue(np.array_equal(ext.min_corner,[0,0]))
        self.assertTrue(np.array_equal(ext.max_corner,[1,2]))
        self.assertTrue(np.array_equal(ext.center,[0.5,1]))

        ext = pointspy.Extent(points)
        self.assertEqual(ext.dim,3)
        self.assertTrue(np.array_equal(ext.ranges,[1,2,4]))
        self.assertTrue(np.array_equal(ext.min_corner,[0,0,0]))
        self.assertTrue(np.array_equal(ext.max_corner,[1,2,4]))
        self.assertTrue(np.array_equal(ext.split,[(0,0,0),(1,2,4)]))
        self.assertTrue(np.array_equal(ext.center,[0.5,1,2]))



        self.assertTrue(ext.intersects([0.5,1,0.5]))
        self.assertFalse(ext.intersects([-1,0,0]))
        self.assertFalse(ext.intersects([0,0,5]))
        self.assertTrue(np.array_equal(
                ext.intersects([(0,0,-1),(0.5,0.5,0.5),(0.1,0.2,0.3),(0,5,0)]),
                [1,2]
            ))
            


if __name__ == '__main__':
    unittest.main()
