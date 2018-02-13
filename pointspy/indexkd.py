import numpy as np
import itertools

from scipy.spatial import cKDTree
import rtree.index as rTreeIndex
from rtree import Rtree as RTree

import transformation
import bisect


class IndexKD(object):
    """Wrapper class for spatial indices to speed up spatial queries and ease
    calulation.
    
    Parameters
    ----------
    coords: (n,k), `array_like`
        Represents n data points of k dimensions.
    transform: (k+1,k+1) `array_like`, optional
        Represents any kind of transformation matrix applied to the coordinates 
        before index computation.
    """

    LEAFSIZE = 20

    def __init__(self, coords, transform=None):
        assert hasattr(coords, '__len__')
        assert len(coords) > 0, 'Empty coordinate lists are not allowed.'
        
        if transform is None:
            self._transform = transformation.i_matrix(coords.shape[1])
            self._coords = np.copy(coords)
        else:
            self._transform = np.matrix(transform)
            self._coords = transformation.transform(
                np.copy(coords), self.transform())

    def __len__(self):
        """Number of points."""
        return self.coords.shape[0]

    def __iter__(self):
        """Iterate over coordinates."""
        return enumerate(self.coords)

    @property
    def coords(self):
        """Provides transformation matrix.
        
        Returns
        -------
        transform: (n,k) `np.ndarray`
            Represents n data points with k dimensions.
        """
        return self._coords

    @property
    def transform(self):
        """Provides transformation matrix
        
        Returns
        -------
        transform: (dim+1,dim+1) `np.matrix`
        """
        return self._transform

    @property
    def dim(self):
        """Provides number of dimensions of coordinates.
        
        Returns
        -------
        dim: `uint`
        """
        return self.coords.shape[1]

    @property
    def kdTree(self):
	"""Provides a kd-tree for rapid neighbourhood queries.
        
        Returns
        -------
        kdTree: `cKDTree`
	"""
        if not hasattr(self, '_kdTree'):
            self._kdTree = cKDTree(self.coords, leafsize=self.LEAFSIZE)
        return self._kdTree

    @property
    def rTree(self):
	""" Provides a r-tree for rapid neighbourhood queries.
        
        Returns
        -------
        rTree: `RTree`
	"""
        
        if not hasattr(self, '_rTree'):
            
	    # define properties
            p = rTreeIndex.Property()
            p.dimension = self.dim()
            p.variant = rTreeIndex.RT_Star
            index = np.concatenate((range(self.dim()), range(self.dim())))

	    # Generator provides required point format
            def gen():
                for id, coord in self:
                    yield (id, coord[index], id)
            self._rTree = RTree(gen(), properties=p)

        return self._rTree

    def ball(self, coords, r, bulk=100000, **kwargs):
        """Find all points within distance r of point(s) coords.
        
        Parameters
        ----------
        coords: (n,k), `array_like`
            Represents n data points of k dimensions.
        r: positive `float`
            Radius of ball.
        bulk: positive `int`
            Reduces required memory by performing bulk queries.
        **kwargs: optional
            Additional parameters similar to 
            scipy.spatial.cKDTree.query_ball_point
        
        Returns
        -------
        nIds: `list or array of lists` 
            If coords is a single point, returns a list neighbours. If coords 
            is an list of points, returns a list containing lists of neighbours.
        """
        if hasattr(r,'__iter__'):
            # query multiple radii
            return list(self.balls_iter(coords, r, **kwargs))
        elif isinstance(coords,np.ndarray) and len(coords.shape)>1:
            # bulk queries
            return list(self.ball_iter(coords, r, bulk=bulk, **kwargs))
        else:
            # single point query
            return self.kdTree.query_ball_point(coords[:,:self.dim], r, **kwargs)
        
    def ball_iter(self, coords, r, bulk=100000, **kwargs):
        """Similar to `ball`, but yields lists of neighbours.
        
        Yields
        -------
        nIds: `list` 
            Lists of indices of neighbouring points.
        """
        assert bulk > 0
        for i in range(coords.shape[0] / bulk + 1):
            # bulk query
            nIds = self.kdTree.query_ball_point(
                coords[bulk * i:bulk * (i + 1),:self.dim], r, **kwargs)
            # yield neighbours
            for nId in nIds:
                yield nId

    def balls_iter(self, coords, radii, **kwargs):
        """Similar to `ball`, but yields lists of neighbours.
        
        Parameters
        ----------
        radii: `iterable`
            Iterable radii to query.
        
        Yields
        -------
        nIds: `list` 
            Lists of indices of neighbouring points.
        """
        for coord,r in itertools.izip(coords,radii):
            nIds = self.kdTree.query_ball_point(coord[:,:self.dim], r, **kwargs)
            yield nIds
            

    def ball_count(self, r, coords=None, p=2):
        """Counts numbers of neighbours within radius.
        
        Parameters
        ----------
        r: `float` or `iterable`
            Iterable radii to query.
        coords: (n,k), `array_like`, optional
            Represents n points of k dimensions. If None it is set to
            self.coords.
        
        Returns
        -------
        count: `list` 
            Number of neigbours for each point.
        """
        if coords is None:
            coords = self.coords
        if hasattr(r, '__iter__'):
            nIdsGen = self.balls_iter(coords, r, p=p)
        else:
            nIdsGen = self.ball_iter(coords, r, p=p)
        return np.array(map(len, nIdsGen), dtype=int)
          
          
    def sphere(self, coord, r_min, r_max, **kwargs):
        """Counts numbers of neighbours within radius.
        
        Parameters
        ----------
        r_min: `float`
            Inner radius of the sphere.
        r_max: `float`
            Outer radius of the sphere.
        coord: (k), `array_like`
            Center of sphere.
        **kwargs:
            Additional parameters similar to self.ball.
            
        Returns
        -------
        sIds: `list` 
            Point indices within sphere.
        """
        assert r_min < r_max
        inner = self.ball(coord[:self.dim], r_min, **kwargs)
        outer = self.ball(coord[:self.dim], r_max, **kwargs)
        return np.intersect1d((outer,inner))
            

    def knn(self, coords, k=1, bulk=100000, **kwargs):
        """Query for k nearest neighbours.
        
        Parameters
        ----------
        coords: (n,k), `array_like`
            Represents n data points of k dimensions.
        k: positive, `int`, optional
            Number of nearest numbers to return.
        **kwargs: optional
            Additional parameters similar to scipy.spatial.cKDTree.query
        
        Returns
        -------
        dists: (n,k), `list` 
            Distances to nearest neihbours.
        indices: (n,k), `list` 
            Indices of nearest neihbours.
        """        
        if hasattr(k,'__iter__'):
            # query multiple radii
            dists,nIds = list(self.knns_iter(coords, k=k, **kwargs))
        elif isinstance(coords,np.ndarray) and len(coords.shape)>1:
            # bulk queries
            dists,nIds = zip(*self.knn_iter(coords, k=k, bulk=bulk, **kwargs))
        else:
            # single point query
            dists,nIds =  self.kdTree.query_ball_point(coords[:,:self.dim], r, **kwargs)
        return np.array(dists),np.array(nIds)
        

    def knn_iter(self, coords, k=1, bulk=100000, **kwargs):
        """Similar to `knn`, but yields lists of neighbours.
        
        Yields
        -------
        dists: (k), `list` 
            Distances to nearest neihbours.
        indices: (k), `list` 
            Indices of nearest neihbours.
        """
        assert bulk > 0
        for i in range(coords.shape[0] / bulk + 1):
            dists, nIds = self.kdTree.query(
                coords[bulk * i:bulk * (i + 1),:self.dim], k=k, **kwargs)
            for d, n in itertools.izip(dists, nIds):
                yield d, n
                
    def knns_iter(self, coords, ks, **kwargs):
        """Similar to `knn`, but yields lists of neighbours.
        
        Parameters
        ----------
        ks: (n) `iterable`
            Iterable numbers of neighbours to query.
        
        Yields
        -------
        dists: (k), `list` 
            Distances to nearest neihbours.
        indices: (k), `list` 
            Indices of nearest neihbours.
        """
        for coord,k in itertools.izip(coords,ks):
            dists, nIds = self.kdTree.query(coord[:,:self.dim], k=k, **kwargs)
            yield dists, nIds
            
                
    
    def nn(self, p=2):
        """Provides nearest neighbours.
        
        Parameters
        ----------
        p: `float`, 1<=p<=infinity, optional
            Minkowski p-norm to use.
        """
        if not hasattr(self, '_NN'):
            dists, ids = self.knn(self.coords, k=2, p=p)
            self._NN = dists[:, 1], ids[:, 1]
        return self._NN

    def closest(self, id, p=2):
        """Provides nearest neighbour of a specific point.
        
        Parameters
        ----------
        p: `float`, 1<=p<=infinity, optional
            Minkowski p-norm to use.
            
        Returns:
        --------
        distance: `float`
            Distance to closest point.
        id: `int`
            Index of closest point.
        """
        dists, ids = self.knn(self.coords[id, :], k=2, p=p)
        return dists[1], ids[1]

    
    def cube(self, coords, r):
        """Provides points within a cube. Wrapper to self.ball with p=infinity
        """
        return self.ball(coords, r, p=float('inf'))
    

    def box(self, extent):
        """Select points within extent.
        
        Parameters
        ----------
        extent: (2*self.dim), `array_like`
            Specifies the points to return. A point p is returned, if
            p>=extent[0:dim] and p>=extent[dim+1:2*dim] 
            
        Returns:
        --------
        indices: `list`
            Indices of points within the extent.
        """
        return self.rTree.intersection(extent, objects='raw')

    # TODO Documentation
    def countBox(self, extent):
        return self.tTree.count(extent)

    # TODO Documentation
    def ballCut(self,
                coord,
                delta,
                p=2,
                filter=lambda pA,pB: pA[-1] < pB[-1]):
        nIds = self.ball(coord, delta, p=p)
        coords = self.coords
        return [nId for nId in nIds if filter(coord, coords[nId, :])]

    # TODO Documentation
    def upperBall(self, coord, delta, p=2, axis=-1):
        return self.ballCut(
            coord,
            delta,
            p=p,
            filter=lambda pA,pB: pA[axis] > pB[axis])

    # TODO Documentation
    def lowerBall(self, coord, delta, p=2, axis=-1):
        return self.ballCut(
            coord,
            delta,
            p=p,
            filter=lambda pA,pB: pA[axis] < pB[axis])

    # TODO Documentation
    def slice(self, minVal, maxVal, axis=-1):
        values = self.coords[:, axis]
        order = np.argsort(values)
        iMin = bisect.bisect_left(values[order], minVal) - 1
        iMax = bisect.bisect_left(values[order], maxVal)

        # return original order for performance reasons
        ids = np.sort(order[iMin:iMax])
        return ids



