import numpy as np
import math
from sklearn.decomposition import PCA

import distance
import fit
import transformation


def angle(v, w):
    a = (np.array(v) * np.array(w)).sum()
    b = math.sqrt(distance.sNorm(v) * distance.sNorm(w))
    if(b > 0):
        a = math.acos(a / b) * 180.0 / math.pi
    else:
        a = float('inf')
    return a


def zenith(v, axis=None):
    if axis is None:
        axis = len(v) - 1
    length = distance.norm(v)
    return math.acos(v[axis] / length) * 180.0 / math.pi


def orthogonal(v, w):
    return scalarProduct(v, w) == 0


def scalarProduct(v, w):
    return np.dot(v, w)


def basisFromVector(vec):

    vec = np.array(vec)
    dim = len(vec)

    # Rotationsmatrix generieren mit vec als 1. Hauptkomponente
    sCoords = np.array([vec * (float(i) / (dim - 1)) for i in range(dim)]).T
    covM = np.cov(sCoords)
    U = np.linalg.svd(covM)[0]

    T = np.matrix(np.identity(dim + 1))
    T[0:dim, 0:dim] = -U.T

    return T


class Vector(transformation.LocalSystem, object):

    def __init__(self, coords):
        #__new__(coords)

        k = self.k(coords)
        #-min(k)==max(k)!
        self._pos = self(min(k))
        self._target = self(max(k))

    def __new__(cls, coords):
        coords = np.array(coords)
        assert len(coords.shape) == 2, 'coordinates needed!'
        assert coords.shape[0] >= 2, 'at least two points!'

        localSystem = fit.PCA(coords)

        # Second fit (==> centering)
        if coords.shape[0] > 2:
            k = localSystem.toLocal(coords)[:, 0]
            lCoords = np.zeros((2, localSystem.dim))
            lCoords[0, 0] = min(k)
            lCoords[1, 0] = max(k)
            localSystem = fit.PCA(localSystem.toGlobal(lCoords))

        return localSystem.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._pos = getattr(obj, '_pos', None)
        self._target = getattr(obj, '_target', None)

    @property
    def pos(self):
        return self._pos

    @property
    def target(self):
        return self._target

    @property
    def origin(self):
        return self(0)

    @property
    def vec(self):
        return self.target - self.pos

    @property
    def length(self):
        return distance.norm(self.vec)

    def k(self, globalCoords):
        coords = np.array(globalCoords)
        if len(coords.shape) < 2:
            return self.toLocal(coords)[0]
        else:
            return self.toLocal(coords)[:, 0]

    def __call__(self, k):
        if hasattr(k, '__len__'):
            v = np.zeros((len(k), self.dim))
            v[:, 0] = k
        else:
            v = np.zeros(self.dim)
            v[0] = k
        return self.toGlobal(v)

    def __str__(self):
        return 'pos: %s; vec: %s' % (str(self.pos), str(self.vec))

    def intersection(self, surface, eps=0.001, maxIter=20):

        assert isinstance(eps, float)

        # Fixpunktiteration
        coord = np.copy(self.target)

        h0 = surface(coord)
        for i in range(maxIter):
            if np.abs(h0 - coord[-1]) < eps:
                break
            coord[-1] = h0
            k = self.k([coord])[0]
            coord = self(k)
            h0 = surface(coord)

        return coord

    def angles(self):
        return np.array([zenith(self.vec, axis=i) for i in range(self.dim)])
