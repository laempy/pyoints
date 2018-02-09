import numpy as np
from scipy.spatial import cKDTree

import rtree.index as RT
from rtree import Rtree as RTree

from . import transformation
import bisect


INF = float('inf')


class IndexKD(object):

    def __init__(self, coords, transform=None):
        assert hasattr(coords, '__len__') and len(
            coords) > 0, 'Empty coordinate list?'
        # Copying of coords required because of memory issues
        if transform is None:
            self._transform = transformation.iMatrix(coords.shape[1])
            self._coords = np.copy(coords)
        else:
            self._transform = np.matrix(transform)
            self._coords = transformation.transform(
                np.copy(coords), self.transform())

    @property
    def kdTree(self):
        # KDTree for neighbourhood queries
        if not hasattr(self, '_kdTree'):
            # print 'generate kD-Tree'
            # self._kdTree=cKDTree(self.coords(), leafsize=50)
            self._kdTree = cKDTree(self.coords(), leafsize=20)
        return self._kdTree

    @property
    def rTree(self):
        # RTree for box queries
        if not hasattr(self, '_rTree'):
            print 'generate R-Tree'
            p = RT.Property()
            p.dimension = self.dim()
            p.variant = RT.RT_Star
            index = np.concatenate((range(self.dim()), range(self.dim())))

            def generator():
                for id, coord in self:
                    yield (id, coord[index], id)
            self._rTree = RTree(generator(), properties=p)
            print 'ok'
        return self._rTree

    def ballIter(self, coords, r, bulk=100000, **kwargs):
        for i in range(coords.shape[0] / bulk + 1):
            nIds = self.kdTree.query_ball_point(
                coords[bulk * i:bulk * (i + 1), :], r, **kwargs)
            for nId in nIds:
                yield nId

    def ball(self, coords, r, bulk=100000, **kwargs):
        if len(coords.shape) == 1:
            return self.kdTree.query_ball_point(coords, r, **kwargs)
        return list(self.ballIter(coords, r, bulk=bulk, **kwargs))

    def ballsIter(self, coords, r, **kwargs):
        assert hasattr(
            r, '__len__'), 'No iterable radii given. Call ballIter instead.'
        assert coords.shape[0] == len(
            r), 'number of coordinates and number of radii have to match.'
        for i in range(coords.shape[0]):
            nIds = self.kdTree.query_ball_point(coords[i, :], r[i], **kwargs)
            yield nIds

    def balls(self, coords, r, **kwargs):
        return list(self.ballsIter(coords, r, bulk=bulk, **kwargs))

    def kNN(self, coords, bulk=100000, **kwargs):
        if len(coords.shape) == 1:
            dists, nIds = self.kdTree.query(coords, **kwargs)
            return dists, nIds
        distsLists = []
        nIdsLists = []
        for i in range(coords.shape[0] / bulk + 1):
            dists, nIds = self.kdTree.query(
                coords[bulk * i:bulk * (i + 1), :], **kwargs)
            distsLists.append(dists)
            nIdsLists.append(nIds)
        return np.concatenate(distsLists), np.concatenate(nIdsLists)

    def kNNIter(self, coords, bulk=100000, **kwargs):
        if len(coords.shape) == 1:
            coords = np.array([coords])
        for i in range(coords.shape[0] / bulk + 1):
            dists, nIds = self.kdTree.query(
                coords[bulk * i:bulk * (i + 1), :], **kwargs)
            for d, n in zip(dists, nIds):
                yield d, n

    def NN(self, p=2):
        if not hasattr(self, '_NN'):
            dists, ids = self.kNN(self.coords(), k=2, p=p)
            self._NN = dists[:, 1], ids[:, 1]
        return self._NN

    def closest(self, id, p=2):
        dists, ids = self.kNN(self.coords()[id, :], k=2, p=p)
        return dists[1], ids[1]

    def countBall(self, r=1, coords=None, p=2):
        if coords is None:
            coords = self.coords()
        if hasattr(r, '__getitem__'):
            nIdsGen = self.ballsIter(coords, r, p=p)
        else:
            nIdsGen = self.ballIter(coords, r, p=p)
        return np.array(map(len, nIdsGen), dtype=int)

    def sphere(self, coord, r1, r2, p=2):
        outerIds = self.ball(coord, max(r1, r2), p=p)
        innerIds = self.ball(coord, min(r1, r2), p=p)
        mask = np.zeros(len(self), dtype=bool)
        mask[outerIds] = True
        mask[innerIds] = False
        return np.where(mask)[0]

    def box(self, extent):
        return self.rTree.intersection(extent, objects='raw')

    def cube(self, coord, r):
        return self.ball(coord, r, p=INF)

    def ballCut(self,
                coord,
                delta,
                p=2,
                filter=lambda p0,
                pI: p0[-1] < pI[-1]):
        nIds = self.ball(coord, delta, p=p)
        coords = self.coords()
        return [nId for nId in nIds if filter(coord, coords[nId, :])]

    def upperBall(self, coord, delta, p=2, axis=-1):
        return self.ballCut(
            coord,
            delta,
            p=p,
            filter=lambda p0,
            pI: p0[axis] > pI[axis])

    def lowerBall(self, coord, delta, p=2, axis=-1):
        return self.ballCut(
            coord,
            delta,
            p=p,
            filter=lambda p0,
            pI: p0[axis] < pI[axis])

    def slice(self, minVal, maxVal, axis=None):
        if axis is None:
            axis = self.dim() - 1
        values = self.coords()[:, axis]
        order = np.argsort(values)
        iMin = bisect.bisect_left(values[order], minVal) - 1
        iMax = bisect.bisect_left(values[order], maxVal)

        # return original order for performance reasons
        ids = np.sort(order[iMin:iMax])
        return ids

    def countBox(self, extent):
        if not hasattr(self, '_rTree'):
            self.generateRTree()
        self.countBox = self._rTree.count
        return self.countBox(extent)

    def coords(self):
        return self._coords

    def transform(self):
        return self._transform

    def __len__(self):
        return self.coords().shape[0]

    def dim(self):
        return self.coords().shape[1]

    def __iter__(self):
        return enumerate(self.coords())
