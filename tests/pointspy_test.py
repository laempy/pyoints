import unittest
import doctest

import pointspy
#from pointpy import *
import numpy as np


def load_tests(loader, tests, ignore):
    #tests.addTests(doctest.DocTestSuite(pointspy.fit))
    tests.addTests(doctest.DocTestSuite(pointspy.vector))
    tests.addTests(doctest.DocTestSuite(pointspy.nptools))
    tests.addTests(doctest.DocTestSuite(pointspy.extent))
    
    #filename = '/daten/Seafile/promotion/Projekte/PANtHEOn/Seafile/TestData/Faro_Focus_S70/Outdoor/LAS_preliminary/Scan_006.las'
    #from pointspy.projection import Proj
    #from pointspy import storage
    #lasHandler = storage.LasHandler(Proj(),filename)
    
    
    return tests


class test_pointpy(unittest.TestCase):


    def test_version(self):
        self.assertTrue(hasattr(pointspy,'__version__'))
        self.assertTrue(float(pointspy.__version__)>0)



class test_transformation(unittest.TestCase):


    def test_matrix(self):

        # 2D
        t = [3,2]
        r = 0.5
        s = [2,1]
        T = pointspy.transformation.matrix(t=t,r=r,s=s)
        
        d = pointspy.transformation.decomposition(T)
        self.assertTrue( np.array_equal(d[0],t) )
        self.assertTrue( np.all(np.isclose(d[1],r)) )
        self.assertTrue( np.all(np.isclose(d[2],s)) )
        
        
        # 3D
        t = [3,2,1]
        r = [0.1,0.2,0.3]
        s = [2,1,0.5]
        T = pointspy.transformation.matrix(t=t,r=r,s=s)
        
        d = pointspy.transformation.decomposition(T)
        self.assertTrue( np.array_equal(d[0],t) )
        self.assertTrue( np.all(np.isclose(d[1],r)) )
        self.assertTrue( np.all(np.isclose(d[2],s)) )
        

    def test_t_matrix(self):
        
        for dim in [2,3,4,5]:
        
            t = np.random.rand(dim)
            T = pointspy.transformation.t_matrix(t)
            self.assertTrue( np.array_equal(np.asarray(T)[:-1,-1],t) )
            
            
    def test_s_matrix(self):
        
        for dim in [2,3,4,5]:
        
            s = np.random.rand(dim)
            T = pointspy.transformation.s_matrix(s)
            self.assertTrue( np.array_equal(np.diag(T)[:-1],s) )
            

    def test_transform(self):
        
        coords = np.random.rand(20,3)
        T = pointspy.transformation.matrix(t=[3,2,1],r=[0.1,0.2,0.3],s=[2,3,0.5])
        
        # rotate coordinates
        rCoords = pointspy.transformation.transform(coords,T)
        
        # the coordinate sets should differ
        self.assertFalse( np.all(np.isclose(coords,rCoords)) )
        
        iCoords = pointspy.transformation.transform(rCoords,T,inverse=True)
        
        # check if the coordinate sets correspond to each other
        self.assertTrue( np.all(np.isclose(coords,iCoords)) )
        
 


class test_IndexKD(unittest.TestCase):
    
    def test_init(self):
        
        for dim in [2,3,4,5]:
            coords = np.random.rand(2,dim)
            T = pointspy.transformation.i_matrix(dim)
            indexKD = pointspy.IndexKD(coords,transform=T)

            self.assertEqual(indexKD.dim,dim)
            self.assertTrue(np.array_equal(indexKD.transform,T))
    
    
    def test_knn(self):
        
        # test different dimensions
        for dim in [2,3,4,5]:
            coords = np.random.rand(2,dim)
            indexKD = pointspy.IndexKD(coords)

            # test knn
            self.assertEqual(indexKD.dim,dim)
            dists,nIds=indexKD.knn(coords,k=3)
            
            # point always closest to itself
            self.assertTrue(np.array_equal(nIds[:,0],[0,1]))
            self.assertTrue(np.all(dists[:,0]==0))
            
            # infinite distances if k>dim
            self.assertTrue(np.all(dists[:,2]==float('inf')))
            
  
    def test_ball(self):
        
        # create regular coords
        coords = np.indices((20,3))[0]
        
        # TODO
        
            


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
        self.assertTrue(np.array_equal(ext.split(),[(0,0,0),(1,2,4)]))
        self.assertTrue(np.array_equal(ext.center,[0.5,1,2]))



        self.assertTrue(ext.intersection([0.5,1,0.5]))
        self.assertFalse(ext.intersection([-1,0,0]))
        self.assertFalse(ext.intersection([0,0,5]))
        self.assertTrue(np.array_equal(
                ext.intersection([(0,0,-1),(0.5,0.5,0.5),(0.1,0.2,0.3),(0,5,0)]),
                [1,2]
            ))
            

if __name__ == '__main__':
    
    print 'main'
    
    #doctest.testmod()
    
    #import unittest
    #import doctest

 

    unittest.main()