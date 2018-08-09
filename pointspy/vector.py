# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Various vector operations.
"""

import math
import numpy as np

from . import (
    assertion,
    distance,
    nptools,
    transformation,
)


def rad_to_deg(rad):
    """Convert angles from radiant to degree.

    Parameters
    ----------
    rad : Number or array_like(Number, shape=(k))
        Angle or angles in radiant.

    Returns
    -------
    Number or np.ndarray(Number, shape=(k))
        Angle or angles in degree.

    See Also
    --------
    deg_to_rad

    Examples
    --------

    >>> rad_to_deg(0.5*np.pi)
    90.0
    >>> print(rad_to_deg([0, np.pi/4, np.pi, 2*np.pi]))
    [  0.  45. 180. 360.]

    """
    if nptools.isarray(rad):
        rad = assertion.ensure_numarray(rad)
    elif not assertion.isnumeric(rad):
        raise ValueError("'rad' neets to be numeric")
    return rad * 180.0 / np.pi


def deg_to_rad(deg):
    """Converts angles from degree to radiant.

    Parameters
    ----------
    deg : Number or array_like(Number, shape=(k))
        Angle or angles in degree.

    Returns
    -------
    Number or np.ndarray(Number, shape=(k))
        Angle or angles in radiant.

    See Also
    --------
    rad_to_deg

    Examples
    --------

    >>> round(deg_to_rad(90), 3)
    1.571
    >>> rad = deg_to_rad([0, 45, 180, 360])
    >>> print(np.round(rad, 3))
    [0.    0.785 3.142 6.283]

    """
    if nptools.isarray(deg):
        deg = assertion.ensure_numarray(deg)
    elif not assertion.isnumeric(deg):
        raise ValueError("'deg' neets to be numeric")
    return deg * np.pi / 180.0


def angle(v, w, deg=False):
    """Calculate angle between two vectors.

    Parameters
    ----------
    v, w : array_like(Number, shape=(k))
        Vector of 'k' dimensions.
    deg : optional, bool
        Indicates whether or not the angle is returned in degree.

    Returns
    -------
    Number
        Angle between vectors v and w.

    Examples
    --------

    Angle between two dimensional vectors.

    >>> angle([0, 1], [1, 0], deg=True)
    90.0
    >>> round(angle([0, 1], [1, 1], deg=True), 1)
    45.0
    >>> angle([2, 2], [1, 1], deg=True)
    0.0
    >>> angle([0, 0], [1, 1], deg=True)
    inf

    Angle between three dimensional vectors.

    >>> angle([1, 0, 1], [0, 1, 0], deg=True)
    90.0

    4D
    >>> angle([1, 0, 0, 0], [0, 1, 1, 0], deg=True)
    90.0

    """
    if not isinstance(deg, bool):
        raise TypeError("'deg' has to be boolean")

    v = assertion.ensure_numvector(v)
    w = assertion.ensure_numvector(w)
    if not len(v) == len(w):
        raise ValueError("vectors 'v' and 'w' have to have the same length")

    a = (v * w).sum()
    b = math.sqrt(distance.snorm(v) * distance.snorm(w))
    if(b > 0):
        a = math.acos(a / b)
        if deg:
            a = rad_to_deg(a)
    else:
        a = float('inf')
    return a


def axes_angles(v, deg=False):
    """Calculates the angles of a vetor to all coordinate axes.

    Parameters
    ----------
    v : array_like(Number, shape=(k))
        Vector of `k` dimensions.
    deg : optional, bool
        Indicates whether or not the angles are returned in degree.

    Returns
    -------
    np.ndarray(Number, shape=(k))
        Rotation angles.

    Examples
    --------
    
    >>> v = [1, 1]
    >>> print(axes_angles(v, deg=True))
    [45. 45.]
    
    >>> v = [0, 1, 0]
    >>> print(axes_angles(v, deg=True))
    [90.  0. 90.]

    """
    v = assertion.ensure_numvector(v)
    e = np.eye(len(v))
    return np.array([angle(v, e[:, i], deg=deg) for i in range(len(v))])


def direction(v, deg=False):
    """Calculate the direction angles of a vector. This direction can be used
    to create a rotation matrix.

    Parameters
    ----------
    v : array_like(Number, shape=(k))
        Vector of `k` dimensions.
    deg : optional, bool
        Indicates whether or not the direction is returned in degree.

    Returns
    -------
    Number or np.ndarray(Number, shape=(k))
        Rotation angles.
    
    Examples
    --------
    
    >>> v = [0, 1]
    >>> print(direction(v, deg=True))
    90.0

    >>> v = [0, 1, 1]
    >>> print(direction(v, deg=True))
    [90. 45.]
    
    """
    v = assertion.ensure_numvector(v)
    v = v / distance.norm(v)

    if not isinstance(deg, bool):
        raise TypeError("'deg' has to be boolean")

    if len(v) == 2:
        direction = np.arcsin(v[1])
    elif len(v) == 3:
        phi = np.arctan2(v[1], v[0])
        theta = np.arccos(v[2])
        direction = np.array([phi, theta])
    else:
        raise ValueError("%i dimensions are not supported yet" % len(v))

    if deg:
        direction = direction * 180.0 / np.pi

    return direction


def zenith(v, axis=-1, deg=False):
    """Angle between a vector and a specific coordinate axes.

    Parameters
    ----------
    v : array_like(Number, shape=(k))
        Vector of `k` dimensions.
    axis : optional, int
        Defines which axis to compare the vector with. If not provided, the
        last dimension is used.

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
        a = rad_to_deg(a)
    return a


def scalarproduct(v, w):
    """Calculates the scalar product or dot product of two vectors.

    Parameters
    ----------
    v,w : array_like(Number, shape=(k))
        Vector of `k` dimensions.

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
    >>> scalarproduct([1, 1], [1, -1])
    0

    """
    v = assertion.ensure_numvector(v)
    w = assertion.ensure_numvector(w)
    if not len(v) == len(w):
        raise ValueError("'v' has to have the same length as 'w'")

    return np.dot(v, w)


def orthogonal(v, w):
    """Check whether or not two vectors are orthogonal.

    Parameters
    ----------
    v,w : array_like(Number, shape=(k))
        Vector of `k` dimensions.

    Returns
    -------
    boolean
        True, if `v` is orthogonal to `w`.

    Examples
    --------

    >>> orthogonal([1, 1], [1, -1])
    True
    >>> orthogonal([1, 1], [1, 0])
    False

    """
    return scalarproduct(v, w) == 0


def basis(vec, origin=None):
    """Generates a local coordinate system based on a vector. The local 
    coordinate system is represented by a rotation matrix.

    Parameters
    ----------
    vec : array_like(Number, shape=(k))
        Vector of `k` dimensions to defines the direction of the first
        coordinate axis of the new coordinate system. The other `k-1` axes are
        build perpendicular to it and each other.
    origin : optional, array_like(Number, shape=(k))
        Defines the origin of the local coordinate system. If None, no shift
        is assumed.

    Returns
    -------
    np.ndarray(Number, shape=(k+1, k+1))
        Transformation matrix representing the desired coordinate system. Since
        the norm of vector `vec` has no influence on the scale, the norm of all
        basic vectors is one.

    Examples
    --------


    Create some two dimensional coordinates.
    
    >>> coords = [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5), (-1, -1)]

    Flip the basic axes.

    >>> b = basis([0, 1])
    >>> print(b)
    [[0. 1. 0.]
     [1. 0. 0.]
     [0. 0. 1.]]
    
    >>> local_coords = transformation.transform(coords, b)
    >>> print(local_coords)
    [[ 0.   0. ]
     [ 1.   0. ]
     [ 0.   1. ]
     [ 1.   1. ]
     [ 0.5  0.5]
     [-1.  -1. ]]
    
    Keep the original orientation, but set a new origin.

    >>> b = basis([2, 0], origin=[2, 3])
    >>> print(np.round(b, 2))
    [[ 1.  0. -2.]
     [ 0.  1. -3.]
     [ 0.  0.  1.]]
    >>> local_coords = transformation.transform(coords, b)
    >>> print(local_coords)
    [[-2.  -3. ]
     [-2.  -2. ]
     [-1.  -3. ]
     [-1.  -2. ]
     [-1.5 -2.5]
     [-3.  -4. ]]

    Use a diagonal basis.

    >>> b = basis([3, 4])
    >>> print(b)
    [[ 0.6  0.8  0. ]
     [-0.8  0.6  0. ]
     [ 0.   0.   1. ]]
    >>> local_coords = transformation.transform(coords, b)
    >>> print(local_coords)
    [[ 0.   0. ]
     [ 0.8  0.6]
     [ 0.6 -0.8]
     [ 1.4 -0.2]
     [ 0.7 -0.1]
     [-1.4  0.2]]

    Three dimensional case.

    >>> b = basis([3, -4, 0], origin=[1, 2, 3])
    >>> print(b)
    [[-0.6  0.8  0.  -1. ]
     [ 0.   0.   1.  -3. ]
     [-0.8 -0.6  0.   2. ]
     [ 0.   0.   0.   1. ]]

    """
    vec = assertion.ensure_numvector(vec)
    if origin is None:
        origin = np.zeros(len(vec))
    else:
        origin = assertion.ensure_numvector(origin, length=len(vec))
    return transformation.PCA([origin - vec, origin + vec])


class Vector(object):
    """Handle vectors conveniently.

    Parameters
    ----------
    origin,vec : array_like(Number, shape=(k))
        The arrays `origin` and `vec` define the location and orientation of
        the vector in a `k` dimensional vector space. 

    Attributes
    ----------
    origin,target,vec : np.ndarray(Number, shape=(k))
        The vector `vec` starts at point `origin` and points to `target`.
    length : positive float
        Length of the vector `vec`.
    dim : positive int
        Number of coordinate dimensions of the vector.
    base : PCA(Number, shape=(k+1, k+1))
        Transformation matrix representation of the local coordinate system
        defined by the vector.

    Examples
    --------

    Two dimensional case.

    >>> v = Vector((5, 7), (3, 4))
    >>> print(v)
    origin: [5 7]; vec: [3 4]
    >>> print(v.target)
    [ 8 11]
    >>> print(v.length)
    5.0

    Three dimensional case.

    >>> v = Vector((1, 1, 1), (2, 3, 4))
    >>> print(v)
    origin: [1 1 1]; vec: [2 3 4]
    >>> print(v.target)
    [3 4 5]

    Edit the vector.

    >>> v = Vector((1, 1), (3, 4))
    >>> print(v)
    origin: [1 1]; vec: [3 4]
    >>> v.vec = (5, 2)
    >>> print(v)
    origin: [1 1]; vec: [5 2]

    >>> v.origin = (-1, -2)
    >>> print(v)
    origin: [-1 -2]; vec: [5 2]

    >>> v.target = (5, 4)
    >>> print(v)
    origin: [-1 -2]; vec: [6 6]

    """
    def __init__(self, origin, vec):
        self.origin = origin
        self.vec = vec
        self.t

    @classmethod
    def from_anges(cls, origin, rot):
        x = np.sin(rot[0]) * np.cos(rot[1])
        z = np.sin(rot[0]) * np.sin(rot[1])
        y = np.cos(rot[1])
        vec = [x, y, z]
        return cls(origin, vec)

    @classmethod
    def from_coords(cls, coords):
        """Vector creation from coordinates using Principal Component Analysis.

        Parameters
        ----------
        coords : array_like(Number, shape=(n, k))
            Represents `n` data points of `k` dimensions. These coordinates are
            used to fit a PCA.

        See Also
        --------
        transformation.PCA

        Examples
        --------

        >>> v = Vector.from_coords([(1, 1), (1, 2), (1, 6)])
        >>> print(v)
        origin: [1. 3.]; vec: [0. 3.]
        >>> print(v.t)
        [[ 0.  1. -3.]
         [ 1.  0. -1.]
         [ 0.  0.  1.]]

        """
        coords = assertion.ensure_coords(coords)
        pca = transformation.PCA(coords)

        length = pca.to_local(coords)[:, 0].max()
        vec = cls(coords.mean(0), pca.pc(1) * length)
        vec._t = pca

        return vec

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

    @property
    def t(self):
        if not hasattr(self, '_t'):
            self._t = basis(self.vec, origin=self.origin)

            # check reflection case
            e = np.eye(1, self.dim)[0] * self.length
            t = self._t.to_local(self.target)
            if not np.all(np.isclose(t, e)):
                self._t.reflect()

            # double check target
            t = self._t.to_local(self.target)
            assert np.all(np.isclose(t, e)), (
                "target '%s' and '%s' differ" % (
                    np.round(t, 2), np.round(e, 2)
                )
            )

            # double check origin
            e = np.zeros(self.dim)
            t = self._t.to_local(self.origin)
            assert np.all(np.isclose(t, e)), (
                "origin '%s' and '%s' differ" % (
                    np.round(t, 2), np.round(e, 2)
                )
            )

            # double check PC1
            t = self._t.pc(1) * self.length
            assert np.all(np.isclose(t, self.vec)), (
                "PC1 '%s' and vector '%s' differ unexpectedly" % (
                    np.round(t, 2), np.round(self.vec, 2)
                )
            )

        return self._t

    def _clear_cache(self):
        if hasattr(self, '_t'):
            del self._t

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
        >>> print(v*3)
        origin: [1 1 1]; vec: [ 6  9 12]

        """
        if not assertion.isnumeric(s):
            raise ValueError("'s' needs to be a scalar")
        return Vector(self.origin, self.vec * s)

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
        >>> print(v / 2.0)
        origin: [1 1 1]; vec: [1.  1.5 2. ]

        """
        return self.__truediv__(s)

    def __truediv__(self, s):
        if not assertion.isnumeric(s):
            raise ValueError("'s' needs to be a scalar")
        return Vector(self.origin, self.vec / s)

    def k(self, global_coords):
        """Calculates the relative position of points in vector direction.

        Parameters
        ----------
        global_coords : array_like(Number, shape=(n, k))
            Represents `n` points of `k` dimensions in a global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(k))
            Relative relative position of points in vector direction. The
            `origin` of the vector defines zero and the `target` defines one.

        Examples
        --------

        >>> v = Vector((1, 1, 1), (2, 3, 4))
        >>> ks = v.k([v.origin - v.vec, v.target, v.origin - 2 * v.vec])
        >>> print(np.round(ks, 2))
        [-1.  1. -2.]

        """
        local_coords = self.t.to_local(global_coords)
        return local_coords[:, 0] / self.length

    def __call__(self, k):
        """Convert a relative position in vector direction to a global
        coordinate.

        Parameters
        ----------
        k : Number or array_like(Number, shape=(n))
            Relative location of a point or points in vector direction.

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
        >>> print(v(2))
        [5 7 9]
        >>> print(v([0, 1, -2, 3]))
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
            Indicates whether or not to provide the angles in degree.

        Returns
        -------
        np.ndarray(Number, shape=(self.dim))
            Angles according to the coordinate axes.

        Examples
        --------

        >>> v = Vector((1, 1, 1), (2, 3, 4))
        >>> angles = v.angles(deg=True)
        >>> print(np.round(angles, 3))
        [68.199 56.145 42.031]

        """
        return np.array([zenith(self.vec, axis=i, deg=deg)
                         for i in range(self.dim)])

    def distance(self, global_coords):
        """Calculate the distance between points and the vector.

        Parameters
        ----------
        global_coords : array_like(Number, shape=(n, self))
            Represents `n` data points.

        Returns
        -------
        np.ndarray(Number, shape=(n))
            Distances of the points to the vector.

        Examples
        --------

        >>> v = Vector((1, -1), (1, 2))
        >>> dist = v.distance([(1, -1), (0, -3), (2, 1), (2, -2), (0, 0)])
        >>> print(np.round(dist, 3))
        [0.    0.    0.    1.342 1.342]
        >>> print(np.linalg.inv(v.t).origin)
        [ 1. -1.]
        >>> print(v.t.pc(1) * v.length)
        [1. 2.]
        >>> print(v.t.pc(2) * v.length)
        [-2.  1.]

        """
        local_coords = self.t.to_local(global_coords)
        return distance.norm(local_coords[:, 1:self.dim])

    def surface_intersection(self, surface, eps=0.001, max_iter=20):
        """Approximates the intersection point between of the vector and a
        surface iteratively.

        Parameters
        ----------
        surface : callable
            Surface model. The model needs to recieve coordinates as an 
            argument and needs to return the distance to the surface. 
            
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
        >>> print(vec.surface_intersection(surface))
        [1. 1. 2.]

        """
        if not hasattr(surface, '__call__'):
            raise ValueError("'surface' is not callable")
        if not (assertion.isnumeric(eps) and eps > 0):
            raise ValueError("'eps' needs to be a number greater zero")
        if not (isinstance(max_iter, int) and max_iter > 0):
            raise TypeError("'max_iter' needs to be an integer greater zero")

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
