"""Module to handle vector operations.
"""

import numpy as np
import math

from . import (
    assertion,
    distance,
    fit,
    nptools,
    transformation,
)


def rad2deg(rad):
    """Converts angles from radiant to degree.

    Parameters
    ----------
    rad : Numeric or array_like(Number, shape=(k, ))
        Angle or angles in radiant.

    Returns
    -------
    float
        Angle or angles in degree.

    See Also
    --------
    deg2rad

    Examples
    --------

    >>> rad2deg(0.5*np.pi)
    90.0
    >>> print rad2deg([0, np.pi/4, np.pi, 2*np.pi])
    [  0.  45. 180. 360.]

    """
    if nptools.isarray(rad):
        rad = assertion.ensure_numarray(rad)
    elif not assertion.isnumeric(rad):
        raise ValueError("'rad' neets to be numeric")
    return rad * 180.0 / np.pi


def deg2rad(deg):
    """Converts angles from degree to radiant.

    Parameters
    ----------
    deg : Number or array_like(Number, shape=(k, ))
        Angle or angles in degree.

    Returns
    -------
    Number or np.ndarray(Number, shape=(k, ))
        Angle or angles in radiant.

    See Also
    --------
    rad2deg

    Examples
    --------

    >>> round(deg2rad(90), 3)
    1.571
    >>> rad = deg2rad([0, 45, 180, 360])
    >>> print np.round(rad, 3)
    [0.    0.785 3.142 6.283]

    """
    if nptools.isarray(deg):
        deg = assertion.ensure_numarray(deg)
    elif not assertion.isnumeric(deg):
        raise ValueError("'deg' neets to be numeric")
    return deg * np.pi / 180.0


def angle(v, w, deg=False):
    """Angle between two vectors.

    Parameters
    ----------
    v, w : array_like(Number, shape=(k, ))
        Vector of length k.

    Returns
    -------
    float
        Angle between vectors v and w.

    Examples
    --------

    2D

    >>> angle([0, 1], [1, 0], deg=True)
    90.0
    >>> round(angle([0, 1], [1, 1], deg=True), 1)
    45.0
    >>> angle([2, 2], [1, 1], deg=True)
    0.0
    >>> angle([0, 0], [1, 1], deg=True)
    inf

    3D

    >>> angle([1,0, 1], [0, 1, 0], deg=True)
    90.0

    4D
    >>> angle([1, 0, 0, 0], [0, 1, 1, 0], deg=True)
    90.0

    """
    # TODO ValueError
    if not isinstance(deg, bool):
        raise ValueError("'deg' has to be boolean")

    v = assertion.ensure_numvector(v)
    w = assertion.ensure_numvector(w)
    if not len(v) == len(w):
        raise ValueError("'v' has to have the same length as 'w'")

    a = (v * w).sum()
    b = math.sqrt(distance.snorm(v) * distance.snorm(w))
    if(b > 0):
        a = math.acos(a / b)
        if deg:
            a = rad2deg(a)
    else:
        a = float('inf')
    return a


def zenith(v, axis=-1, deg=False):
    """Angle between a vector and the coordinate axes.

    Parameters
    ----------
    v : array_like(Number, shape=(k, ))
        Vector of length `k`.
    axis : optional, int
        Defines which axis to compare the vector with.

    Returns
    -------
    float
        Angle between the vector and the selected coordinate axis.

    Examples
    --------

    >>> zenith([1, 0], deg=True)
    90.0
    >>> zenith([1, 0], axis=0)
    0.0
    >>> round(zenith([1, 1], deg=True), 1)
    45.0
    >>> round(zenith([1, 0, 1], 2, deg=True), 1)
    45.0

    """
    v = assertion.ensure_numvector(v)
    if not (isinstance(axis, int) and abs(axis) < len(v)):
        raise ValueError("'axis' neets to be an integer smaller len(v)")

    length = distance.norm(v)
    a = math.acos(v[axis] / length)
    if deg:
        a = rad2deg(a)
    return a


def scalarproduct(v, w):
    """Calculates the scalar product between two vectors.

    Parameters
    ----------
    Parameters
    ----------
    v, w : array_like(Number, shape=(k, ))
        Vector of length k.

    Returns
    -------
    float
        Scalar product or dot product of the vectors.

    Examples
    --------

    >>> scalarproduct([1, 2], [3, 4])
    11
    >>> scalarproduct([1, 2, 3], [4, 5, 6])
    32

    othoogonal vectors

    >>> scalarproduct([1, 1], [1, -1])
    0

    """
    v = assertion.ensure_numvector(v)
    w = assertion.ensure_numvector(w)
    if not len(v) == len(w):
        raise ValueError("'v' has to have the same length as 'w'")

    return np.dot(v, w)


def orthogonal(v, w):
    """Check wether two vectors are orthogonal.

    Parameters
    ----------
    v, w : array_like(Number, shape=(k, ))
        Vector of length k.

    Returns
    -------
    boolean
        True, if v is orthogonal to w.

    Examples
    --------

    >>> orthogonal([1, 1], [1, -1])
    True
    >>> orthogonal([1, 1], [1, 0])
    False

    """
    return scalarproduct(v, w) == 0


def basis(vec):
    """Generates a local coordinate system based on the provided vector. The
    local coordinate system is represented by a rotation matrix.

    Parameters
    ----------
    vec : array_like(Number, shape=(k))
        Vector of `k` dimensions to defines the direction of the first
        coordinate axis of the new coordinate system. The other `k-1` axes are
        build perpendicular to it and each other.

    Returns
    -------
    np.array(Number, shape=(k+1, k+1))
        Rotation matrix representing the desired coordinate system. Since the
        norm  of the vector `vec` has no influence on the scale, the norm of
        all basic vectors of the coordinate system is one.

    Examples
    --------

    Two dimensional case.

    >>> from pointspy import transformation
    >>> corners = [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5), (-1, -1)]

    X-vector changes

    >>> b = basis([1, 0])
    >>> print(np.round(b, 2))
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    >>> local_coords = transformation.transform(corners, b)
    >>> print(np.round(local_coords, 3))
    [[ 0.   0. ]
     [ 0.   1. ]
     [ 1.   0. ]
     [ 1.   1. ]
     [ 0.5  0.5]
     [-1.  -1. ]]

    >>> b = basis([0, 2])
    >>> print(np.round(b, 2))
    [[0. 1. 0.]
     [1. 0. 0.]
     [0. 0. 1.]]
    >>> local_coords = transformation.transform(corners, b)
    >>> print(np.round(local_coords, 3))
    [[ 0.   0. ]
     [ 1.   0. ]
     [ 0.   1. ]
     [ 1.   1. ]
     [ 0.5  0.5]
     [-1.  -1. ]]

    >>> b = basis([1, 1])
    >>> print(np.round(b, 2))
    [[ 0.71  0.71  0.  ]
     [ 0.71 -0.71  0.  ]
     [ 0.    0.    1.  ]]
    >>> local_coords = transformation.transform(corners, b)
    >>> print(np.round(local_coords, 3))
    [[ 0.     0.   ]
     [ 0.707 -0.707]
     [ 0.707  0.707]
     [ 1.414  0.   ]
     [ 0.707  0.   ]
     [-1.414  0.   ]]

    Three dimensional case.

    >>> b = basis([1, 1, 0])
    >>> print(np.round(b, 2))
    [[ 0.71  0.71 -0.    0.  ]
     [-0.   -0.   -1.    0.  ]
     [ 0.71 -0.71 -0.    0.  ]
     [ 0.    0.    0.    1.  ]]

    """
    vec = assertion.ensure_numvector(vec)
    dim = len(vec)

    # TODO replace by PCA([np.zeros(len(vec)), vec])
    return fit.PCA([-vec, vec])

    # generate a rotation matrix with vec as 1th princiopal component
    #vec = vec / distance.norm(vec)
    sCoords = np.array([vec * (float(dim - i - 1) / (dim - 1)) for i in range(dim)]).T
    covM = np.cov(sCoords)

    U = np.linalg.svd(covM)[0]

    pc1 = U[0, :]
    mIndex = np.argmax(np.abs(pc1))
    if pc1[mIndex] < 0:
        U = -U

    # TODO signum?
    T = np.matrix(np.identity(dim + 1))
    T[0:dim, 0:dim] = U.T

    return T


class Vector(object):
    """Vector class.

    Parameters
    ----------
    origin, vec : array_like(Number, shape=(k, ))
        The arrays `origin` and `vec` define the location and orientation of
        the vector in a `k`-dimensional vector space. The vector `vec` starts
        at point `origin` and points at point `target`.

    Attributes
    ----------
    origin, target, vec : np.ndarray(Number, shape=(k, ))
        The vector `vec` starts at point `origin` and points at point `target`.
    length : positive float
        Length of the vector `vec`.
    dim : positive int
        Number of coordinate dimensions `k` of the vector.
    base : np.matrix(Number, shape = (k+1, k+1))
        Transformation matrix representation of the local coordinate system
        defined by the vector.

    Examples
    --------

    Two dimensional case.

    >>> v = Vector((5, 7), (3, 4))
    >>> print v
    origin: [5 7]; vec: [3 4]
    >>> print v.target
    [ 8 11]
    >>> print v.length
    5.0

    Three dimensional case.

    >>> v = Vector((1, 1, 1), (2, 3, 4))
    >>> print v
    origin: [1 1 1]; vec: [2 3 4]
    >>> print v.target
    [3 4 5]

    Edit vector.

    >>> v = Vector((1, 1), (3, 4))
    >>> print v
    origin: [1 1]; vec: [3 4]
    >>> v.vec = (5, 2)
    >>> print v
    origin: [1 1]; vec: [5 2]

    >>> v.origin = (-1, -2)
    >>> print v
    origin: [-1 -2]; vec: [5 2]

    >>> v.target = (5, 4)
    >>> print v
    origin: [-1 -2]; vec: [6 6]

    """
    def __init__(self, origin, vec):
        self.origin = origin
        self.vec = vec

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._target = getattr(obj, '_target', None)

    @property
    def dim(self):
        return len(self.origin)

    @property
    def length(self):
        return distance.norm(self.vec)

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        self._origin = assertion.ensure_numvector(origin)
        self._clear_cache()

    @property
    def vec(self):
        return self._vec

    @vec.setter
    def vec(self, vec):
        self._vec = assertion.ensure_numvector(vec)
        self._clear_cache()

    @property
    def target(self):
        return self.origin + self.vec

    @target.setter
    def target(self, target):
        self.vec = assertion.ensure_numvector(target) - self.origin

    def _clear_cache(self):
        if hasattr(self, '_base'):
            del self._base

    @property
    def base(self):
        if not hasattr(self, '_base'):
            v = self.origin - self.vec
            w = self.origin + self.vec
            self._base = fit.PCA([v, w])
            assert np.all(np.isclose(self._base.pc(1) * self.length, self.vec))
        return self._base

    def __str__(self):
        return 'origin: %s; vec: %s' % (str(self.origin), str(self.vec))

    def __mul__(self, s):
        """Scale a vector by multiplication.

        Parameters
        ----------
        s : Number
            Scale factor.

        Returns
        -------
        Vector
            Scaled vector.

        Examples
        --------

        >>> v = Vector((1, 1, 1), (2, 3, 4))
        >>> print v*3
        origin: [1 1 1]; vec: [ 6  9 12]

        """
        if not assertion.isnumeric(s):
            raise ValueError("'s' needs to be a scalar")
        return Vector(self.origin, self.vec*s)

    def __div__(self, s):
        """Scale a vector by division.

        Parameters
        ----------
        s : Number
            Scale factor.

        Returns
        -------
        Vector
            Scaled vector.

        Examples
        --------

        >>> v = Vector((1, 1, 1), (2, 3, 4))
        >>> print v / 2.0
        origin: [1 1 1]; vec: [1.  1.5 2. ]

        """
        if not assertion.isnumeric(s):
            raise ValueError("'s' needs to be a scalar")
        return Vector(self.origin, self.vec/s)

    def k(self, gcoords):
        """Calculates the relative position of points in vector direction.

        Parameters
        ----------
        gcoords : array_like(Number, shape=(n, k))
            Represents `n` data points of `k` dimensions in a global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(k, ))
            Relative relative position of points in vector direction. The
            `origin` of the vector defines zero and the `target` defines one.

        Examples
        --------

        >>> v = Vector((1, 1, 1), (2, 3, 4))
        >>> ks = v.k([v.origin, v.target, v.origin - 2 * v.vec])
        >>> print np.round(ks, 2)
        [ 0.  1. -2.]

        """
        lcoords = self.base.to_local(gcoords)
        return lcoords[:, 0] / self.length

    def __call__(self, k):
        """Convert a relative position in vector direction to a global
        coordinate.

        Parameters
        ----------
        k : Number or array_like(Number, shape=(n, ))
            Represents `n` data points of `k` dimensions in a global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(n, self.dim))
            Global coordinates.

        See Also
        --------
        Vector.k

        Examples
        --------

        >>> v = Vector((1, 1, 1), (2, 3, 4))
        >>> print v(2)
        [5 7 9]
        >>> print v([0, 1, -2, 3])
        [[ 1  1  1]
         [ 3  4  5]
         [-3 -5 -7]
         [ 7 10 13]]

        """
        if assertion.isnumeric(k):
            return self.origin + k * self.vec
        else:
            ks = assertion.ensure_numvector(k)
            v = np.array([self.vec * l for l in ks])
            return v + self.origin

    def angles(self, deg=False):
        """Calculates the angles of the vector to the coordinate axes.

        Parameters
        ----------
        deg : bool
            Indicates weather or not to provide the angles in degree.

        Returns
        -------
        np.ndarray(Number, shape=(self.dim))
            Angles according to the coordinate axes.

        Examples
        --------

        >>> v = Vector((1, 1, 1), (2, 3, 4))
        >>> angles = v.angles(deg=True)
        >>> print np.round(angles, 3)
        [68.199 56.145 42.031]

        """
        return np.array([zenith(self.vec, axis=i, deg=deg)
                         for i in range(self.dim)])

    def distance(self, gcoords):
        """Calculate the distance between points and the vector.

        Parameters
        ----------
        gcoords : array_like(Number, shape=(n, self))
            Represents `n` data points.

        Returns
        -------
        np.ndarray(Number, shape=(n))
            Distances of the points to the vector.

        Examples
        --------

        >>> v = Vector((1, -1), (1, 2))
        >>> dist = v.distance([(1, -1), (0, -3), (2, 1), (2, -2), (0, 0)])
        >>> print np.round(dist, 3)
        [0.    0.    0.    1.342 1.342]
        >>> print np.linalg.inv(v.base).origin
        [ 1. -1.]
        >>> print v.base.pc(1) * v.length
        [1. 2.]
        >>> print v.base.pc(2) * v.length
        [-2.  1.]

        """
        lcoords = self.base.to_local(gcoords)
        return distance.norm(lcoords[:, 1:self.dim])

    def surface_intersection(self, surface, eps=0.001, max_iter=20):
        """ Approximates the intersection point between the vector and a surface
        iteratively.

        Parameters
        ----------
        surface : function


        Returns
        -------
        coord:
            Approximate intersection point between the vector and a surface.

        Examples
        --------

        >>> from pointspy import surface, interpolate
        >>> method = interpolate.LinearInterpolator
        >>> surface = surface.Surface(
        ...         [(0, 0, 0), (0, 2, 0), (2, 1, 4)],
        ...         method=method
        ...     )
        >>> vec = Vector((1, 1, -1), (0, 0, 1))
        >>> print vec.surface_intersection(surface)
        [1. 1. 2.]

        """
        if not hasattr(surface, '__call__'):
            raise ValueError("'surface' is not callable")
        if not (assertion.isnumeric(eps) and eps > 0):
            raise ValueError("'eps' needs to be a number greater zero")
        if not (isinstance(max_iter, int) and max_iter > 0):
            raise ValueError("'max_iter' needs to be an integer greater zero")

        coord = np.copy(self.target)

        for i in range(max_iter):

            h0 = surface([coord])

            # check residual
            if np.abs(h0 - coord[-1]) < eps:
                break

            # set new coordinate
            coord[-1] = h0
            k = self.k([coord])[0]
            coord = self(k)

        return coord
