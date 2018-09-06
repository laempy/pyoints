# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University,
# lamprecht@uni-trier.de
#
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
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
    """Class to represent vectors and handle them conveniently.

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
    t : PCA(Number, shape=(k+1, k+1))
        Roto-translation matrix representation of the local coordinate system
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
        self._vec = assertion.ensure_numvector(vec, length=len(self.origin))
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



class Plane(object):
    """Class to represent two hyperplanes and handle them conveniently.
    
    Parameters
    ----------
    origin : np.ndarray(Number, shape=(k))
        Defines the origin of the planes `k` dimensional local coordinate 
        system.
    \*vec : np.ndarray(Number, shape=(k))
        The `k`-1 vectors of `vec` define the orientation of the plane. The 
        missing  axis (perpenticular to the vectors) are calculated 
        automatically using principal component analysis. So, paralell
        vectors in `vec` are valid, but might result in unexpected results.
        
    Attributes
    ----------
    origin : np.ndarray(Number, shape=(k))
        Origin of the plane.
    vec : np.ndarray(Number, shape=(k-1, k))
        Vectors defining the orientation of the plane.
    dim : positive int
        Number of coordinate dimensions of the vector.
    t : PCA(Number, shape=(k+1, k+1))
        Roto-translation matrix representation of the local coordinate system
        defined by the plane. 
    
    Examples
    --------
    
    Creation of a plane object using an origin and two vectors defining the 
    plane.
    
    >>> origin = [1, 2, 3]
    >>> v = [0, 0, 2]
    >>> w = [1, 0, 0]
    
    >>> plane = Plane(origin, v, w)
    >>> print(plane)
    origin: [1 2 3]; vec: [0 0 2], [1 0 0]
    
    Get some properties.
    
    >>> print(plane.dim)
    3
    >>> print(plane.t.inv)
    [[0. 1. 0. 1.]
     [0. 0. 1. 2.]
     [1. 0. 0. 3.]
     [0. 0. 0. 1.]]
    >>> print(plane.t.eigen_values)
    [8. 2. 0.]

    Transformation of global coordinates to the plane coordinate system.

    >>> global_coords = [
    ...     plane.origin,
    ...     plane.origin + plane.vec[0, :],
    ...     -plane.vec[1, :],
    ...     plane.vec[0, :] - 3,
    ...     plane.origin + 2 * plane.vec[0, :] + 3 * plane.vec[1, :]
    ... ]
    >>> print(np.array(global_coords))
    [[ 1  2  3]
     [ 1  2  5]
     [-1  0  0]
     [-3 -3 -1]
     [ 4  2  7]]
    
    >>> local_coords = plane.t.to_local(global_coords)
    >>> print(local_coords)
    [[ 0.  0.  0.]
     [ 2.  0.  0.]
     [-3. -2. -2.]
     [-4. -4. -5.]
     [ 4.  3.  0.]]
    >>> print(plane.t.to_global(local_coords))
    [[ 1.  2.  3.]
     [ 1.  2.  5.]
     [-1.  0.  0.]
     [-3. -3. -1.]
     [ 4.  2.  7.]]
    
    Calculation of the distance of the global points to the plane.

    >>> print(plane.distance(global_coords))
    [0. 0. 2. 5. 0.]
    
    Creation of the spectial case of a line in a two dimensional space.
    
    >>> plane = Plane([1, 2], [3, 4])
    >>> print(plane)
    origin: [1 2]; vec: [3 4]
    
    >>> print(plane.t.inv)
    [[ 0.6 -0.8  1. ]
     [ 0.8  0.6  2. ]
     [ 0.   0.   1. ]]
    >>> print(plane.t.eigen_values)
    [50.  0.]
    >>> print(plane.dim)
    2
    
    Transformation of global coordinates to the plane coordinate system.
    
    >>> global_coords = [
    ...     plane.origin,
    ...     plane.origin + plane.vec[0, :],
    ...     plane.vec[0, :] - 3,
    ...     plane.origin + 2 * plane.vec[0, :]
    ... ]
    >>> print(np.array(global_coords))
    [[ 1  2]
     [ 4  6]
     [ 0  1]
     [ 7 10]]
    
    >>> local_coords = plane.t.to_local(global_coords)
    >>> print(local_coords)
    [[ 0.   0. ]
     [ 5.   0. ]
     [-1.4  0.2]
     [10.   0. ]]
    >>> print(plane.t.to_global(local_coords))
    [[ 1.  2.]
     [ 4.  6.]
     [ 0.  1.]
     [ 7. 10.]]
    
    Calculation of the distance of the global points to the plane.

    >>> print(plane.distance(global_coords))
    [0.  0.  0.2 0. ]
    
    """
    
    def __init__(self, origin, *vec):
        # validated by setter
        self.origin = origin
        self.vec = vec
        
    @property
    def dim(self):
        return len(self.origin)

    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self, origin):
        self._origin = assertion.ensure_numvector(origin, min_length=2)
        self._clear_cache()

    @property
    def vec(self):
        return self._vec

    @vec.setter
    def vec(self, vecs):
        k = (self.dim - 1)
        if not len(vecs) == k:
            raise ValueError("%i vectors required, got %i" % (k, len(vecs)))
        self._vec = assertion.ensure_numarray(vecs)
        self._clear_cache()

    def _clear_cache(self):
        if hasattr(self, '_t'):
            del self._t
        
    @property
    def t(self):
        if not hasattr(self, '_t'):
            coords = np.vstack((self.vec, -self.vec)) + self.origin
            self._t = transformation.PCA(coords)
        return self._t


    def __mul__(self, s):
        """Scale the plane vectors by multiplication.

        Parameters
        ----------
        s : array_like(Number, shape=(2))
            Scale factors.

        Returns
        -------
        Plane
            Scaled plane.

        Examples
        --------

        Three dimensional case.

        >>> plane = Plane((1, 2, 3), (5, 9, 2), (2, 3, 4))
        >>> print(plane * [2, 5])
        origin: [1 2 3]; vec: [10 18  4], [10 15 20]
        
        Two dimensional case (line).
        
        >>> plane = Plane((1, 2), (5, 9))
        >>> print(plane * 2)
        origin: [1 2]; vec: [10 18]
        
        """
        if self.dim == 2:
            s = [s]
        s = assertion.ensure_numvector(s, length=self.dim-1)
        vec = (self.vec.T * s).T
        return Plane(self.origin, *vec)

    def __div__(self, s):
        """Scale a plane by division.

        Parameters
        ----------
        s : array_like(Number, shape=(2))
            Scale factors.

        Returns
        -------
        Plane
            Scaled plane.

        Examples
        --------

        Three dimensional case.

        >>> plane = Plane((1, 2, 3), (5, 9, 2), (2, 3, 4))
        >>> print(plane / [5, 2])
        origin: [1 2 3]; vec: [1.  1.8 0.4], [1.  1.5 2. ]

        Two dimensional case (line).

        >>> plane = Plane((1, 2), (5, 9))
        >>> print(plane / 2)
        origin: [1 2]; vec: [2.5 4.5]

        """
        return self.__truediv__(s)

    def __truediv__(self, s):
        if self.dim == 2:
            s = [s]
        s = assertion.ensure_numvector(s, length=self.dim-1)
        vec = (self.vec.T / s).T
        return Plane(self.origin, *vec)
            

    def distance(self, global_coords):
        """Calculate the distance between points and the plane.

        Parameters
        ----------
        global_coords : array_like(Number, shape=(n, self))
            Represents `n` data points.

        Returns
        -------
        np.ndarray(Number, shape=(n))
            Distances of the points to the plane.

        """
        local_coords = self.t.to_local(global_coords)
        return distance.norm(local_coords[:, (self.dim-1):self.dim])
    

    def __call__(self, k):
        """Convert a relative position in plane vectors direction to a global
        coordinate.

        Parameters
        ----------
        k : array_like(Number, shape=(2)) or array_like(Number, shape=(n, 2))
            Relative location of a point or points in plane vector direction.

        Returns
        -------
        np.ndarray(Number, shape=(n, self.dim))
            Global coordinates.

        See Also
        --------
        Plane.k

        Examples
        --------
        
        Three dimensional case.

        >>> plane = Plane((1, 2, 3), (1, 0, 2), (0, 3, 0))
        >>> print(plane([1, 2]))
        [2 8 5]
        >>> print(plane([(0, 0), (1, 0), (0, 1), (-2, 3)]))
        [[ 1  2  3]
         [ 2  2  5]
         [ 1  5  3]
         [-1 11 -1]]
        
        Two dimensional case (line).

        >>> plane = Plane((1, 2), (5, 9))
        >>> print(plane(0.5))
        [3.5 6.5]
        >>> print(plane([1, 2]))
        [[ 6 11]
         [11 20]]

        """
        if self.dim == 2:
            k = assertion.ensure_numarray([k]).T
        else:
            k = assertion.ensure_numarray(k)
            
        if len(k.shape) == 1:
            if not len(k) == (self.dim - 1):
                m = "'k' needs to have %i values, got %i"
                raise ValueError(m % ((self.dim - 1), len(k)))
            vec = (self.vec.T * k).sum(1)
        elif len(k.shape) == 2:
            vec = np.array([(self.vec.T * l).sum(1) for l in k])
        else:
            raise ValueError("'k' needs to have a shape of (2) or (n, 2)")
        coords = self.origin + vec
        return coords

    def __str__(self):
        vec_str = ', '.join(str(vec) for vec in self.vec)
        return "origin: %s; vec: %s" % (str(self.origin), vec_str)
    
    
def vector_surface_intersection(vec, surface, eps=0.001, max_iter=20):
    """Approximates the intersection point between of a `k` dimensional vector 
    and a `k` dimensional surface iteratively.

    Parameters
    ----------
    vec : Vector
        Vector to calculate the intersection point for.
    surface : callable
        Surface model. The model needs to recieve coordinates as an
        argument and needs to return the distance to the surface.

    Returns
    -------
    coord : np.array_like(Number, shape=(k))
        Approximate intersection point between the vector and the surface.

    Examples
    --------

    >>> from pyoints import surface, interpolate

    Create a callable surface object and a vector.

    >>> surface = surface.Surface(
    ...         [(0, 0, 0), (0, 2, 0), (2, 1, 4)],
    ...         method=interpolate.LinearInterpolator
    ...     )
    >>> vec = Vector((1, 1, -1), (0, 0, 1))
    
    Calculate the intersection point.
    
    >>> print(vector_surface_intersection(vec, surface))
    [1. 1. 2.]

    """
    if not isinstance(vec, Vector):
        raise TypeError("'vec' needs to be an instance of Vector")
    if not hasattr(surface, '__call__'):
        raise ValueError("'surface' is not callable")
    if not (assertion.isnumeric(eps) and eps > 0):
        raise ValueError("'eps' needs to be a number greater zero")
    if not (isinstance(max_iter, int) and max_iter > 0):
        raise TypeError("'max_iter' needs to be an integer greater zero")

    coord = np.copy(vec.target)

    for i in range(max_iter):

        h0 = surface([coord])

        # check residual
        if np.abs(h0 - coord[-1]) < eps:
            break

        # set new coordinate
        coord[-1] = h0
        k = vec.k([coord])[0]
        coord = vec(k)

    return coord


def vector_plane_intersection(vec, plane):
    """Calculates the intersection point of a `k` dimensional vector and a 
    `k` dimensional plane.

    Parameters
    ----------
    vec : Vector
        Vector to calculate the intersection point for.
    plane : Plane
        Plane to calculate the intersection point for.

    Returns
    -------
    coord : np.array_like(Number, shape=(k))
        Intersection point of the vector and the plane.

    Notes
    -----
    The algorithm solves the linear equation system:
    `plane.origin - vec.origin | k \* vec.vec - s \* plane.vec`

    Examples
    --------

    Create a plane and a vector in three dimensional space and calculate the 
    intersection point.

    >>> vec = Vector((1, 1, -1), (0, 0, 3))
    >>> plane = Plane((-1, 2, 5), (0, 2, 0), (1, 2, 2))
        
    >>> print(vector_plane_intersection(vec, plane))
    [1. 1. 9.]
    
    Create a plane and a vector in two dimensional space and calculate the 
    intersection point.

    >>> vec = Vector((1, 1), (1, 2))
    >>> plane = Plane((-1, 5), (0, 1))
        
    >>> print(vector_plane_intersection(vec, plane))
    [-1. -3.]

    """
    if not isinstance(vec, Vector):
        raise TypeError("'vec' needs to be an instance of Vector")
    if not isinstance(plane, Plane):
        raise TypeError("'plane' needs to be an instance of Plane")

    # plane.origin + s * plane.vec ==  vec.origin + k * vec.vec
    # plane.origin - vec.origin == k * vec.vec - s * plane.vec
    b = plane.origin - vec.origin
    A = np.vstack([vec.vec, -plane.vec]).T

    res = np.linalg.lstsq(A, b, rcond=None)

    p1 = vec(res[0][0])
    p2 = plane(res[0][1:])

    # check solution
    if np.allclose(p1, p2):
        return p1
    else:
        return None

