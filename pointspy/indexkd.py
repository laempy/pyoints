import numpy as np
import itertools
from numbers import Number

from scipy.spatial import cKDTree
import rtree.index as r_treeIndex
from rtree import Rtree

import bisect

from . import assertion
from . import transformation

# TODO module description
# TODO nested IndexKD?
# ==>


class IndexKD(object):
    """Wrapper class of serveral spatial indices to speed up spatial queries
    and ease usage.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.
    transform : optional, np.matrix(Number, shape=(k+1, k+1))
        Represents any kind of transformation matrix applied to the coordinates
        before index computation.
    leafsize : optional, positive, int
        Leaf size of KD-Tree.


    Attributes
    ----------
    dim : positive int
        Number of coordinate dimensions
    transform : (self.dim+1, self.dim+1), array_like
        Transformation matrix.
    coords : (len(self), self.dim), array_like
        Coordinates of the spatial index.
    kd_tree : `scipy.spatial.cKDTree`
        KD-tree for rapid neighbourhood queries. Generated on first demand.
    r_tree : `rtree.Rtree`
        R-tree for rapid box queries. Generated on first demand.

    """

    def __init__(self, coords, transform=None, leafsize=16, quickbuild=True):

        coords = assertion.ensure_coords(coords).copy()

        assert isinstance(leafsize, int) and leafsize > 0
        assert isinstance(quickbuild, bool)

        self._leafsize = leafsize
        self._balanced = not quickbuild
        self._compact = not quickbuild

        if transform is None:
            self._coords = coords
            self._transform = transformation.i_matrix(self._coords.shape[1])
        else:
            self._transform = assertion.ensure_tmatrix(transform)
            self._coords = transformation.transform(coords, self._transform)

    def __len__(self):
        return self.coords.shape[0]

    def __iter__(self):
        return enumerate(self.coords)

    @property
    def dim(self):
        return self.coords.shape[1]

    @property
    def coords(self):
        return self._coords

    @property
    def transform(self):
        return self._transform

    @property
    def kd_tree(self):
        if not hasattr(self, '_kd_tree'):
            self._kd_tree = cKDTree(
                self.coords,
                leafsize=self._leafsize,
                copy_data=False,
                balanced_tree=self._balanced,
                compact_nodes=self._compact
            )
        return self._kd_tree

    @property
    def r_tree(self):
        if not hasattr(self, '_r_tree'):

            # define properties
            p = r_treeIndex.Property()
            p.dimension = self.dim()
            p.variant = r_treeIndex.RT_Star
            index = np.concatenate((range(self.dim()), range(self.dim())))

            # Generator provides required point format
            def gen():
                for id, coord in self:
                    yield (id, coord[index], id)
            self._r_tree = Rtree(gen(), properties=p)

        return self._r_tree

    def ball(self, coords, r, bulk=100000, **kwargs):
        """Find all points within distance `r` of point or points `coords`.

        Parameters
        ----------
        coords : (n, k), array_like
            Represents `n` data points of `k` dimensions.
        r : positive float
            Radius of ball.
        bulk : positive int
            Reduce required memory by performing bulk queries.
        **kwargs : optional
            Additional parameters similar to
            `scipy.spatial.cKDTree.query_ball_point`


        Returns
        -------
        nIds: `list or array of lists`
            If coords is a single point, returns a list neighbours. If coords
            is an list of points, returns a list containing lists of
            neighbours.
        """

        if hasattr(r, '__iter__'):
            # query multiple radii
            return list(self.balls_iter(coords, r, **kwargs))
        elif isinstance(coords, np.ndarray) and len(coords.shape) > 1:
            # bulk queries
            return list(self.ball_iter(coords, r, bulk=bulk, **kwargs))
        else:
            # single point query
            return self.kd_tree.query_ball_point(
                coords[:self.dim], r, **kwargs)

    def ball_iter(self, coords, r, bulk=10000, **kwargs):
        """Similar to `ball`, but yields lists of neighbours.

        Yields
        ------
        nIds : list of ints
            Lists of indices of neighbouring points.

        See Also
        --------
        ball, balls_iter

        """
        if not isinstance(bulk, int) and bulk > 0:
            raise ValueError("bulk size has to be a integer greater zero")

        for i in range(coords.shape[0] // bulk + 1):
            # bulk query
            nIds = self.kd_tree.query_ball_point(
                coords[bulk * i: bulk * (i + 1),
                       : self.dim],
                r, n_jobs=-1, **kwargs)
            # yield neighbours
            for nId in nIds:
                yield nId

    def balls_iter(self, coords, radii, **kwargs):
        """Yields lists of neighbours.

        Parameters
        ----------
        radii: iterable of floats
            Radii to query.

        Yields
        ------
        nIds : list
            Lists of indices of neighbouring points.

        See Also
        --------
        ball, ball_iter

        """
        for coord, r in zip(coords, radii):
            nIds = self.kd_tree.query_ball_point(coord[:self.dim], r, **kwargs)
            yield nIds

    def ball_count(self, r, coords=None, p=2):
        """Counts numbers of neighbours within radius.

        Parameters
        ----------
        r : float or iterable of floats
            Iterable radii to query.
        coords : (n, k), array_like, optional
            Represents `n` points of `k` dimensions. If None it is set to
            `self.coords`.

        Returns
        -------
        list if ints
            Number of neigbours for each point.

        See Also
        --------
        ball, ball_iter, balls_iter

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
        r_min : float
            Inner radius of the sphere.
        r_max : float
            Outer radius of the sphere.
        coord: (k), `array_like`
            Center of sphere.
        **kwargs:
            Additional parameters similar to `self.ball`.

        Returns
        -------
        list of ints
            Indices of points within sphere.
        """
        if not isinstance(r_min, Number) and r_min > 0:
            raise ValueError("r_min has to be numeric and greater zero")

        if not isinstance(r_max, Number) and r_max > r_min:
            raise ValueError("r_max has to be numeric and greater 'r_max'")

        inner = self.ball(coord[:self.dim], r_min, **kwargs)
        outer = self.ball(coord[:self.dim], r_max, **kwargs)
        return np.intersect1d((outer, inner))

    def knn(self, coords, k=1, bulk=100000, **kwargs):
        """Query for k nearest neighbours.

        Parameters
        ----------
        coords : (n, k), array_like
            Represents n data points of k dimensions.
        k : positive int, optional
            Number of nearest numbers to return.
        **kwargs : optional
            Additional parameters similar to `scipy.spatial.cKDTree.query`.

        Returns
        -------
        dists, dist : (n, k), list
            Distances to nearest neihbours.
        indices : (n, k), list
            Indices of nearest neihbours.

        See Also
        --------
        knn, knns_iter, scipy.spatial.cKDTree.query

        """
        if hasattr(k, '__iter__'):
            # query multiple radii
            dists, nIds = list(self.knns_iter(coords, k=k, **kwargs))
        elif isinstance(coords, np.ndarray) and len(coords.shape) > 1:
            # bulk queries
            dists, nIds = zip(*self.knn_iter(coords, k=k, bulk=bulk, **kwargs))
        else:
            # single point query
            dists, nIds = self.kd_tree.query(
                coords[:self.dim], k, **kwargs)
        return np.array(dists), np.array(nIds)

    def knn_iter(self, coords, k=1, bulk=100000, **kwargs):
        """Similar to `knn`, but yields lists of neighbours.

        Yields
        -------
        dists, indices : (k), list
            List of distances to nearest neighbours and corresponding point
            indices.

        See Also
        --------
        knn, knns_iter

        """
        if not isinstance(bulk, int) and bulk > 0:
            raise ValueError("bulk size has to be a integer greater zero")

        for i in range(coords.shape[0] // bulk + 1):
            dists, nIds = self.kd_tree.query(
                coords[bulk * i:bulk * (i + 1), :self.dim], k=k, **kwargs)
            for d, n in zip(dists, nIds):
                yield d, n

    def knns_iter(self, coords, ks, **kwargs):
        """Similar to `knn`, but yields lists of neighbours.

        Parameters
        ----------
        ks : (n), iterable
            Iterable numbers of neighbours to query.

        Yields
        -------
        dists: (k), list
            Distances to nearest neihbours.
        indices: (k), list
            Indices of nearest neihbours.
        """
        for coord, k in zip(coords, ks):
            dists, nIds = self.kd_tree.query(
                coord[:, :self.dim], k=k, **kwargs)
            yield dists, nIds

    def nn(self, p=2):
        """Provides nearest neighbours.

        Parameters
        ----------
        p : float, 1<=p<=infinity, optional
            Minkowski p-norm to use.
        """
        if not hasattr(self, '_NN'):
            dists, ids = self.knn(self.coords, k=2, p=p)
            self._NN = dists[:, 1], ids[:, 1]
        return self._NN

    def closest(self, id, **kwargs):
        """Provides nearest neighbour of a specific point.

        Parameters
        ----------
        id : int
            Index of a point in `self.coords`.

        Returns:
        --------
        distance : positive float
            Distance to closest point.
        id : positive int
            Index of closest point.

        See Also
        --------
        knn

        """
        dists, ids = self.knn(self.coords[id, :], k=2, **kwargs)
        return dists[1], ids[1]

    def cube(self, coords, r, **kwargs):
        """Provides points within a cube. Wrapper to self.ball with `p=infinity`

        See Also
        --------
        ball
        """
        return self.ball(coords, r, p=float('inf'))

    def box(self, extent):
        """Select points within extent.

        Parameters
        ----------
        extent : (2*self.dim), array_like
            Specifies the points to return. A point p is returned, if
            p>=extent[0:dim] and p>=extent[dim+1:2*dim]

        Returns
        -------
        indices : list of ints
            Indices of points within the extent.

        See Also
        --------
        ball, cube, slice

        """
        return self.r_tree.intersection(extent, objects='raw')

    # TODO Documentation
    def slice(self, min_th, max_th, axis=-1):
        """Select points with coordinate value of axis `axis` within range the
        [`min_th`, `max_th`].

        Parameters
        ----------
        min_th, max_th : float
            A point `p` is returned `min_th <= p[axis] <= max_th`.
        axis : int
            Axis to evaluate.

        Returns
        -------
        list of ints
            Indices of points within the slice.

        See Also
        --------
        box, cube, slice

        """
        if not isinstance(min_th, Number) and min_th > 0:
            raise ValueError("'min_th' has to be numeric and greater zero")
        if not isinstance(max_th, Number) and max_th > min_th:
            raise ValueError("'max_th' has to be numeric and greater 'min_th'")
        if not isinstance(axis, int) and abs(axis) < self.dim:
            raise ValueError("'axis' has to be an integer and smaller 'dim'")

        values = self.coords[:, axis]
        order = np.argsort(values)
        iMin = bisect.bisect_left(values[order], min_th) - 1
        iMax = bisect.bisect_left(values[order], max_th)

        # return original order (for performance reasons)
        ids = np.sort(order[iMin:iMax])
        return ids

    # TODO Documentation
    def countBox(self, extent):
        return self.tTree.count(extent)

    # TODO Documentation
    def ballCut(self,
                coord,
                delta,
                p=2,
                filter=lambda pA, pB: pA[-1] < pB[-1]):
        nIds = self.ball(coord, delta, p=p)
        coords = self.coords
        return [nId for nId in nIds if filter(coord, coords[nId, :])]

    # TODO Documentation
    def upperBall(self, coord, delta, p=2, axis=-1):
        return self.ballCut(
            coord,
            delta,
            p=p,
            filter=lambda pA, pB: pA[axis] > pB[axis])

    # TODO Documentation
    def lowerBall(self, coord, delta, p=2, axis=-1):
        return self.ballCut(
            coord,
            delta,
            p=p,
            filter=lambda pA, pB: pA[axis] < pB[axis])
