import numpy as np
import math

from . import distance
from . import fit
from . import transformation


def to_degree(a):
    return a * 180.0 / math.pi


def to_rad(a):
    return a * math.pi / 180.0


def angle(v, w, deg=False):
    """Angle between two vectors.

    Parameters
    ----------
    v, w : (k), array_like
        Vector of length k.

    Returns
    -------
    float
        Angle between vectors v and w.

    Examples
    --------

    2D

    >>> angle([0,1],[1,0],deg=True)
    90.0
    >>> round(angle([0,1],[1,1],deg=True),1)
    45.0
    >>> angle([2,2],[1,1],deg=True)
    0.0
    >>> angle([0,0],[1,1],deg=True)
    inf

    3D

    >>> angle([1,0,1],[0,1,0],deg=True)
    90.0

    4D
    >>> angle([1,0,0,0],[0,1,1,0],deg=True)
    90.0

    """
    assert hasattr(v, '__len__')
    assert hasattr(w, '__len__')
    assert len(v) == len(w)

    a = (np.array(v) * np.array(w)).sum()
    b = math.sqrt(distance.snorm(v) * distance.snorm(w))
    if(b > 0):
        a = math.acos(a / b)
        if deg:
            a = to_degree(a)
    else:
        a = float('inf')
    return a


def zenith(v, axis=-1, deg=False):
    """Angle between a vector and the coordinate axes.

    Parameters
    ----------
    v : (k), `array_like`
        Vector of length k.
    axis : int, optional
        Defines which axis to compare the vector with.

    Returns
    -------
    float
        Angle between the vector and the selected coordinate axis.

    Examples
    --------

    >>> zenith([1,0],deg=True)
    90.0
    >>> zenith([1,0],axis=0)
    0.0
    >>> round( zenith([1,1],deg=True),1)
    45.0
    >>> round( zenith([1,0,1],2,deg=True) ,1)
    45.0

    """
    assert hasattr(v, '__len__')
    assert isinstance(axis, int)
    assert abs(axis) < len(v)

    length = distance.norm(v)
    a = math.acos(v[axis] / length)
    if deg:
        a = to_degree(a)
    return a


def scalarproduct(v, w):
    """Calculates the scalar product between two vectors.

    Parameters
    ----------
    Parameters
    ----------
    v, w : (k), array_like
        Vector of length k.

    Returns
    -------
    float
        Scalar product or dot product of the vectors.

    Examples
    --------

    >>> scalarproduct([1,2],[3,4])
    11
    >>> scalarproduct([1,2,3],[4,5,6])
    32

    othoogonal vectors

    >>> scalarproduct([1,1],[1,-1])
    0
    """

    assert hasattr(v, '__len__')
    assert hasattr(w, '__len__')
    assert len(v) == len(w)

    return np.dot(v, w)


def orthogonal(v, w):
    """Check wether two vectors are orthogonal.

    Parameters
    ----------
    v, w : (k), array_like
        Vector of length k.

    Returns
    -------
    boolean
        True, if v is orthogonal to w.

    Examples
    --------

    >>> orthogonal([1,1],[1,-1])
    True
    >>> orthogonal([1,1],[1,0])
    False

    """
    return scalarproduct(v, w) == 0


def basis(vec):
    # TODO

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
    """ Vector class.

    Parameters
    ----------
    coords : (n,k), array_like
        Coordinates

    Examples
    --------

    TODO

    """

    # TODO vereinfachen?

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

    def intersection(self, surface, eps=0.001, max_iter=20):
        """ Approximates the intersection point between the vector and a surface
        iteratively.

        Parameters
        ----------
        TODO

        Returns
        -------
        coord:
            Approximate intersection point between the vector and a surface.

        Examples
        --------
        TODO


        """
        assert hasattr(surface, '__call__')
        assert isinstance(eps, float)

        coord = np.copy(self.target)

        h0 = surface(coord)
        for i in range(max_iter):

            # check residual
            if np.abs(h0 - coord[-1]) < eps:
                break

            # set new coordinate
            coord[-1] = h0
            k = self.k([coord])[0]
            coord = self(k)

            h0 = surface(coord)

        return coord

    def angles(self, deg=False):
        return np.array([zenith(self.vec, axis=i, deg=deg)
                         for i in range(self.dim)])
