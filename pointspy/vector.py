"""Handling of vectors.
"""
import numpy as np
import math

from . import (
    assertion,
    distance,
    fit,
    transformation,
)


# TODO module documentation

def to_degree(a):
    return a * 180.0 / np.pi


def to_rad(a):
    return a * np.pi / 180.0


def angle(v, w, deg=False):
    """Angle between two vectors.

    Parameters
    ----------
    v, w : array_like(shape=(k))
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
    # TODO ValueError
    if not isinstance(deg, bool):
        raise ValueError("'deg' has to be boolean")

    v = assertion.ensure_vector(v)
    w = assertion.ensure_vector(w)
    if not len(v) == len(w):
        raise ValueError("'v' has to have the same length as 'w'")

    a = (v * w).sum()
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
    v = assertion.ensure_vector(v)
    if not (isinstance(axis, int) and abs(axis) < len(v)):
        raise ValueError("'axis' neets to be an integer smaller len(v)")

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

    v = assertion.ensure_vector(v)
    w = assertion.ensure_vector(w)
    if not len(v) == len(w):
        raise ValueError("'v' has to have the same length as 'w'")

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
    coords : array_like(Number, shape=(n,k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.

    Attributes
    ----------
    origin, target, vec : np.ndarray(Number, shape=(k))
        The vector `vec` starts point `origin` and points at point `target`.
    length : positive float
        Length of the vector `vec`.


    Examples
    --------

    Two dimensional case.

    >>> v = Vector([(3,4),(5,4)])
    >>> print v
    origin: [3. 4.]; vec: [2. 0.]
    >>> print v.origin
    [3. 4.]
    >>> print v.target
    [5. 4.]
    >>> print v.length
    2.0

    Three dimensional case.

    >>> v = Vector([(1,1,1),(2,3,4)])
    >>> print v
    origin: [1. 1. 1.]; vec: [1. 2. 3.]
    >>> print v.origin
    [1. 1. 1.]
    >>> print v.target
    [2. 3. 4.]


    """
    def __init__(self, coords):
        #__new__(coords)
        self._max_k = max(self.k(coords))

    def __new__(cls, coords):
        coords = assertion.ensure_coords(coords)
        if not coords.shape[0] >= 2:
            ValueError('at least two points needed')

        localSystem = fit.PCA(coords)

        # Second fit (==> centering)
        if True or coords.shape[0] > 2:
            k = localSystem.to_local(coords)[:, 0]
            lCoords = np.zeros((2, localSystem.dim))
            lCoords[0, 0] = -3*max(k)
            lCoords[1, 0] = max(k)
            gcoords = localSystem.to_global(lCoords)
            localSystem = fit.PCA(gcoords)

        return localSystem.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._target = getattr(obj, '_target', None)

    @property
    def target(self):
        return self(self._max_k)

    @property
    def origin(self):
        return self(0)

    @property
    def vec(self):
        return self.target - self.origin

    @property
    def length(self):
        return distance.norm(self.vec)

    def k(self, gcoords):
        lcoords = self.to_local(gcoords)
        if len(lcoords.shape) < 2:
            return lcoords[0]
        else:
            return lcoords[:, 0]

    def __call__(self, k):
        if hasattr(k, '__len__'):
            v = np.zeros((len(k), self.dim))
            v[:, 0] = k
        else:
            v = np.zeros(self.dim)
            v[0] = k
        return self.to_global([v])[0]

    def __str__(self):
        return 'origin: %s; vec: %s' % (str(self.origin), str(self.vec))

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
