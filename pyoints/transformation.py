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
"""Multidimensional transformation matrices and coordinate transformations.
"""

import cv2
import warnings
import numpy as np
import itertools as it

from numpy.linalg import eigh
from numpy import (
    identity,
    dot,
)

from . import (
    distance,
)
from .assertion import (
    ensure_coords,
    ensure_numarray,
    ensure_numvector,
    ensure_tmatrix,
    isnumeric,
)


def transform(coords, T, inverse=False, precise=False):
    """Performs a linear transformation to coordinates using a transformation
    matrix.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k)) or array_like(Number, shape=(k))
        Represents `n` data points of `k` dimensions.
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix.
    precise : optional, bool
        Indicates whether or not to calculate the transformaton (at the expend
        of increased computation time) extra precise.

    Returns
    -------
    coords : np.ndarray(Number, shape=(n, k)) or np.ndarray(Number, shape=(k))
        Transformed coordinates.

    Examples
    --------

    Transform coordinates forth and back again.

    >>> coords = [(0, 0), (1, 2), (2, 3), (0, 3)]
    >>> T = [(1, 0, 5), (0, 1, 3), (0, 0, 1)]

    >>> tcoords = transform(coords, T)
    >>> print(tcoords)
    [[5. 3.]
     [6. 5.]
     [7. 6.]
     [5. 6.]]
    >>> print(transform(tcoords, T, inverse=True))
    [[0. 0.]
     [1. 2.]
     [2. 3.]
     [0. 3.]]

    """
    T = ensure_tmatrix(T)
    coords = ensure_numarray(coords)

    if inverse:
        try:
            T = np.linalg.inv(T)
        except np.linalg.LinAlgError as e:
            warnings.warn(str(e))
            T = np.linalg.pinv(T)

    T = np.asarray(T)
    if len(coords) == 0 or coords.shape[0] == 0:
        raise ValueError("can not transform empty array")
    elif len(coords.shape) == 1 and coords.shape[0] == T.shape[0] - 1:
        # single point
        tcoords = np.dot(np.append(coords, 1), T.T)[0: -1]
    elif len(coords.shape) == 2 and coords.shape[1] == T.shape[0] - 1:
        # multiple points
        if precise:
            # precise, but slow
            #tcoords = (homogenious(coords) @ T.T)[:, 0:-1]
            tcoords = np.dot(homogenious(coords), T.T)[:, 0:-1]
        else:
            # fast, but not very precise
            tcoords = cv2.transform(
                np.expand_dims(coords.astype(np.float64), axis=0),
                T
            )[0][:, 0:-1]
    else:
        raise ValueError("dimensions do not match")

    return tcoords.view(coords.__class__)


def homogenious(coords, value=1):
    """Create homogeneous coordinates by adding an additional dimension.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    value : optional, Number
        Desired values of additional dimension.

    Returns
    -------
    hcoords : np.ndarray(Number, shape=(n, k+1))
        Homogeneous coordinates.

    """
    coords = ensure_coords(coords)
    if not isnumeric(value):
        raise ValueError("'value' needs to be numeric")

    if len(coords.shape) == 1:
        H = np.append(coords, value)
    else:
        N, dim = coords.shape
        H = np.empty((N, dim + 1))
        H[:, :-1] = coords
        H[:, -1] = value
    return H


def matrix(t=None, r=None, s=None, order='srt'):
    """Creates a transformation matrix based on translation, rotation and scale
    coefficients.

    Parameters
    ----------
    t,r,s : optional, np.ndarray(Number, shape=(k))
        Transformation coefficients for translation `t`, rotation `r` and
        scale `s`.
    order : optional, string
        Order to compute the matrix multiplications. For example, an `order` of
        `trs` means to first translate `t`, then rotate `r`, and finally scale
        `s`.

    Returns
    -------
    LocalSystem(shape=(k+1, k+1))
        Transformation matrix according to arguments `t`, `r`, `s` and order
        `order`.

    See Also
    --------
    LocalSystem, t_matrix, r_matrix, s_matrix

    """
    shape = (0, 0)
    keys = ('t', 'r', 's')
    orders = [''.join(o) for o in it.permutations(keys)]

    if not isinstance(order, str):
        raise TypeError("'order' needs to be a string")
    if order not in orders:
        raise ValueError("order '%s' unknown" % order)

    if t is None and r is None and s is None:
        raise ValueError("at least one attribute neets to be provided")

    # create the matrices
    matrices = {}
    if t is not None:
        matrices['t'] = t_matrix(t)
        shape = matrices['t'].shape
    if r is not None:
        matrices['r'] = r_matrix(r)
        shape = matrices['r'].shape
    if s is not None:
        matrices['s'] = s_matrix(s)
        shape = matrices['s'].shape

    # double check the shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    assert shape[0] > 1

    # fill missing matrices with identity matrices
    for key in keys:
        if key not in matrices.keys():
            matrices[key] = i_matrix(shape[0] - 1)

    # check dimensions
    for M in matrices.values():
        if not M.shape == shape:
            raise ValueError("matrix dimensions differ")

    # create translation matrix according to order
    M = i_matrix(shape[0] - 1)
    for key in list(order):
        M = matrices[key] @ M

    return LocalSystem(M)


def i_matrix(dim):
    """Creates an identity transformation matrix.

    Parameters
    ----------
    dim : positive int
        Desired dimension of the transformation matrix. At least a value of
        two is reqired.

    Returns
    -------
    LocalSystem(int, shape=(dim+1, dim+1))
        Identity transformation matrix.

    See Also
    --------
    LocalSystem

    Examples
    --------

    >>> print(i_matrix(3))
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    """
    if not (isinstance(dim, int) and dim >= 2):
        raise ValueError("'dim' needs to be an integer greater one")
    I_m = np.identity(dim + 1)
    return LocalSystem(I_m)


def t_matrix(t):
    """Creates a translation matrix.

    Parameters
    ----------
    t : array_like(Number, shape=(k))
        Translation coefficients for each coordinate dimension. At least two
        coefficients are required.

    Returns
    -------
    LocalSystem(shape=(k+1, k+1))
        Translation matrix.

    See Also
    --------
    LocalSystem

    Examples
    --------

    >>> print(t_matrix([1, 2, 3]))
    [[1. 0. 0. 1.]
     [0. 1. 0. 2.]
     [0. 0. 1. 3.]
     [0. 0. 0. 1.]]

    """
    t = ensure_numvector(t, min_length=2)
    T_m = np.identity(len(t) + 1)
    T_m[:-1, -1] = t
    return LocalSystem(T_m)


def s_matrix(s):
    """Creates a scaling matrix.

    Parameters
    ----------
    s : array_like(Number, shape=(k))
        Scaling coefficients for each coordinate dimension. At least two
        coefficients are required.

    Returns
    -------
    LocalSystem(shape=(k+1, k+1))
        Scaling matrix.

    See Also
    --------
    LocalSystem

    Examples
    --------

    >>> print(s_matrix([0.5, 1, -3]))
    [[ 0.5  0.   0.   0. ]
     [ 0.   1.   0.   0. ]
     [ 0.   0.  -3.   0. ]
     [ 0.   0.   0.   1. ]]

    """
    if not (hasattr(s, '__len__') and len(s) >= 2):
        raise ValueError("'s' needs have a length greater one")
    dim = len(s)
    S_m = np.identity(dim + 1)
    diag = np.append(s, 1)
    np.fill_diagonal(S_m, diag)
    return LocalSystem(S_m)


def r_matrix(a, order='xyz'):
    """Creates a rotation matrix.

    Parameters
    ----------
    r : Number or array_like(Number, shape=(k))
        Rotation coefficients for each coordinate dimension. At least two
        coefficients are required.
    order : optional, String
        For at least three axes, `order` specifies the order of rotations. For
        example, an order of `zxy` means first to translate according to z axis,
        then rotate according to x-axis, and finally rotate according to
        y-axis.

    Returns
    -------
    LocalSystem(shape=(k+1, k+1))
        Rotation matrix.

    See Also
    --------
    LocalSystem

    Notes
    -----
    Currently only two and tree dimensions are supported yet.

    Examples
    --------

    Two dimensional case.

    >>> R = r_matrix(np.pi/4)
    >>> print(np.round(R, 3))
    [[ 0.707 -0.707  0.   ]
     [ 0.707  0.707  0.   ]
     [ 0.     0.     1.   ]]

    Three dimensional case.

    >>> R = r_matrix([np.pi/2, 0, np.pi/4])
    >>> print(np.round(R, 3))
    [[ 0.707 -0.     0.707  0.   ]
     [ 0.707  0.    -0.707  0.   ]
     [ 0.     1.     0.     0.   ]
     [ 0.     0.     0.     1.   ]]

    """
    if isnumeric(a):
        a = [a]
    else:
        a = ensure_numvector(a)

    keys = ('x', 'y', 'z')
    orders = [''.join(o) for o in it.permutations(keys)]

    if not isinstance(order, str):
        raise TypeError("'order' needs to be a string")
    if order not in orders:
        raise ValueError("order '%s' unknown" % order)

    R_dict = {}
    dim = len(a)
    if dim == 1:
        R_m = np.array([
            [np.cos(a[0]), -np.sin(a[0]), 0],
            [np.sin(a[0]), np.cos(a[0]), 0],
            [0, 0, 1]
        ])
    elif dim == 2:
        raise ValueError('rotation in 2D requires one angle only')
    elif dim == 3:
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(a[0]), -np.sin(a[0]), 0],
            [0, np.sin(a[0]), np.cos(a[0]), 0],
            [0, 0, 0, 1],
        ])
        Ry = np.array([
            [np.cos(a[1]), 0, np.sin(a[1]), 0],
            [0, 1, 0, 0],
            [-np.sin(a[1]), 0, np.cos(a[1]), 0],
            [0, 0, 0, 1],
        ])
        Rz = np.array([
            [np.cos(a[2]), -np.sin(a[2]), 0, 0],
            [np.sin(a[2]), np.cos(a[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        R_dict['x'] = Rx
        R_dict['y'] = Ry
        R_dict['z'] = Rz

        R_m = i_matrix(dim)
        for key in list(order):
            R_m = R_dict[key] @ R_m
    else:
        raise ValueError(
            '%i-dimensional rotations are not supported yet' % dim)

    return LocalSystem(R_m)


def q_matrix(q):
    """Creates an rotation matrix based on quaternion values.
    
    Parameters
    ----------
    q : array_like(Number, shape=(4))
        Quaternion parameters (w, x, y, z).
    
    Returns
    -------
    LocalSystem(int, shape=(4, 4))
        Rotation matrix derived from quaternions.
        
    Examples
    --------
    
    >>> T = q_matrix([0.7071, 0.7071, 0, 0])
    >>> print(np.round(T, 2))
    [[ 1.  0.  0.  0.]
     [ 0.  0. -1.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  0.  1.]]
        
    >>> t, r, s, det = decomposition(T)
    >>> print(r)
    [ 1.57077715 -0.          0.        ]
    >>> print(t)
    [0. 0. 0.]
    
    
    """
    q = ensure_numvector(q, length=4)
    if not np.isclose(distance.snorm(q), 1, rtol=0.001):
        raise ValueError("Invalid quaternion. Square sum should be one.")
        
    yaw = np.arctan2(
            2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    roll = np.arctan2(
            2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return r_matrix([yaw, pitch, roll])


def add_dim(T):
    """Adds a dimension to a transformation matrix.

    Parameters
    ----------
    T : LocalSystem(Number, shape=(k+1, k+1))
        Transformation matrix of `k` coordinate dimensions.

    Returns
    -------
    np.ndarray(float, shape=(k+2, k+2))
        Transformation matrix with an additional dimension.

    Examples
    --------

    Two dimensional case.

    >>> T = matrix(t=[2, 3 ], s=[0.5, 2])
    >>> print(np.round(T, 3))
    [[0.5 0.  2. ]
     [0.  2.  3. ]
     [0.  0.  1. ]]
    >>> T = add_dim(T)
    >>> print(np.round(T, 3))
    [[0.5 0.  0.  2. ]
     [0.  2.  0.  3. ]
     [0.  0.  1.  0. ]
     [0.  0.  0.  1. ]]

    """
    T = ensure_tmatrix(T)
    M = np.eye(len(T) + 1)
    M[:-2, :-2] = T[:-1, :-1]
    M[:-2, -1] = T[:-1, -1].T
    return LocalSystem(M)


def decomposition(T, ignore_warnings=False):
    """Determines some characteristic parameters of a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix of `k` coordinate dimensions.

    Returns
    -------
    t,r,s : optional, np.ndarray(Number, shape=(k))
        Transformation coefficients for translation `t`, rotation `r` and
        scale `s`.
    det : float
        Determinant `det` indicates a distorsion of `T`.

    Examples
    --------

    >>> T = matrix(t=[2, 3], r=0.2, s=[0.5, 2])
    >>> t, r, s, det = decomposition(T)
    >>> print(t)
    [2. 3.]
    >>> print(r)
    0.2
    >>> print(s)
    [0.5 2. ]
    >>> print(det)
    1.0

    See Also
    --------
    matrix

    Notes
    -----
    Idea taken from [1], [2] and [3].

    References
    -----
    [1] https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
    [2] https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix/13165#13165
    [3] https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    """
    T = ensure_tmatrix(T)
    dim = T.shape[0] - 1

    # translation
    t = np.asarray(T)[:-1, -1]

    # scale
    s = distance.norm(np.asarray(T.T))[:-1]

    # rotation
    R = T[:-1, :-1] / s
    if dim == 2:
        r1 = np.arctan2(R[1, 0], R[1, 1])
        r2 = np.arctan2(-R[0, 1], R[0, 0])
        if not ignore_warnings and not np.isclose(r1, r2):
            warnings.warn("Rotation angles seem to differ.")
        r = (r1 + r2) * 0.5
    elif dim == 3:
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            r_x = np.arctan2(R[2, 1], R[2, 2])
            r_y = np.arctan2(-R[2, 0], sy)
            r_z = np.arctan2(R[1, 0], R[0, 0])
        else:
            r_x = np.arctan2(-R[1, 2], R[1, 1])
            r_y = np.arctan2(-R[2, 0], sy)
            r_z = 0
        r = np.array([r_x, r_y, r_z])
    else:
        raise ValueError('Only %s dimensions are not supported jet' % dim)

    # determinant
    det = np.linalg.det(T)

    return t, r, s, det


def matrix_from_gdal(t):
    """Creates a transformation matrix using a gdal geotransfom array.

    Parameters
    ----------
    t : array_like(Number, shape=(6))
        Gdal geotransform array.

    Returns
    -------
    LocalSystem(Number, shape=(3, 3))
        Matrix representation of the gdal geotransform array.

    See Also
    --------
    matrix_to_gdal, LocalSystem

    Examples
    --------

    >>> T = matrix_from_gdal([-28493, 9, 2, 4255884, -2, -9.0])
    >>> print(T.astype(int))
    [[      9       2  -28493]
     [     -2      -9 4255884]
     [      0       0       1]]

    """
    t = ensure_numvector(t, min_length=6, max_length=6)

    T = np.array(np.zeros((3, 3), dtype=np.float))
    T[0, 2] = t[0]
    T[0, 0] = t[1]
    T[0, 1] = t[2]
    T[1, 2] = t[3]
    T[1, 0] = t[4]
    T[1, 1] = t[5]
    T[2, 2] = 1
    return LocalSystem(T)


def matrix_to_gdal(T):
    """Creates gdal geotransfom array based on a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(3, 3))
        Matrix representation of the gdal geotransform array.

    Returns
    -------
    t : array_like(Number, shape=(6))
        Gdal geotransform array.

    See Also
    --------
    matrix_from_gdal

    Examples
    --------

    >>> T = np.array([(9, -2, -28493), (2, -9, 4255884), (0, 0, 1)])
    >>> t = matrix_to_gdal(T)
    >>> print(t)
    (-28493.0, 9.0, -2.0, 4255884.0, 2.0, -9.0)

    """
    T = ensure_tmatrix(T).astype(np.float32)
    if not T.shape[0] == 3:
        raise ValueError('transformation matrix of shape (3, 3) required')
    t = (T[0, 2], T[0, 0], T[0, 1], T[1, 2], T[1, 0], T[1, 1])
    return t


class LocalSystem(np.ndarray, object):
    """Defines a local coordinate system based on a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix of `k` coordinate dimensions.

    Attributes
    ----------
    dim : positive int
        Number of coordinate dimensions.
    origin : np.ndarray(Number, shape=(k))
        Global origin of the local coordinate system.

    Examples
    --------

    >>> L = LocalSystem([(2, -0.5, 2), (0.5, 1, 3), (0, 0, 1)])
    >>> print(L)
    [[ 2.  -0.5  2. ]
     [ 0.5  1.   3. ]
     [ 0.   0.   1. ]]
    >>> print(L.dim)
    2
    >>> print(np.round(L.origin, 3))
    [2. 3.]

    See Also
    --------
    matrix

    """
    def __new__(cls, T):
        return ensure_tmatrix(T).view(cls)

    @property
    def dim(self):
        return len(self) - 1

    @property
    def det(self):
        return np.linalg.det(self)

    @property
    def inv(self):
        return np.linalg.inv(self).view(self.__class__)

    @property
    def origin(self):
        return self[:self.dim, self.dim]

    @origin.setter
    def origin(self, origin):
        origin = ensure_numvector(origin, length=self.dim)
        self[:self.dim, self.dim] = origin

    def decomposition(self):
        """Determinates some characteristic parameters of the transformation
        matrix.

        Returns
        -------
        tuple
            Decomposition values.

        See Also
        --------
        decomposition

        """
        return decomposition(self)

    def reflect(self, axis=0):
        """Reflects a specific coordinate axis.

        Parameters
        ----------
        axis : positive int
            Coordinate axis to reflect.

        Examples
        --------

        Reflect a two dimensional transformation matrix.

        >>> L = LocalSystem(matrix(t=[1, 2], s=[0.5, 2]))
        >>> print(L)
        [[0.5 0.  1. ]
         [0.  2.  2. ]
         [0.  0.  1. ]]
        >>> print(L.to_local([2, 3]))
        [2. 8.]

        >>> L.reflect()
        >>> print(L)
        [[-0.5 -0.  -1. ]
         [ 0.   2.   2. ]
         [ 0.   0.   1. ]]
        >>> print(L.to_local([2, 3]))
        [-2.  8.]

        Reflect a three dimensional transformation matrix.

        >>> L = LocalSystem(matrix(t=[1, 2, 3], s=[0.5, -2, 1]))
        >>> print(L)
        [[ 0.5  0.   0.   1. ]
         [ 0.  -2.   0.   2. ]
         [ 0.   0.   1.   3. ]
         [ 0.   0.   0.   1. ]]
        >>> print(L.to_local([1, 2, 3]))
        [ 1.5 -2.   6. ]

        >>> L.reflect(axis=1)
        >>> print(L)
        [[ 0.5  0.   0.   1. ]
         [ 0.   2.   0.  -2. ]
         [ 0.   0.   1.   3. ]
         [ 0.   0.   0.   1. ]]
        >>> print(L.to_local([1, 2, 3]))
        [1.5 2.  6. ]

        """
        R = np.eye(self.dim + 1)
        R[axis, axis] = -1
        self[:, :] = np.linalg.inv(np.linalg.inv(self) @ R)

    def to_local(self, global_coords):
        """Transforms global coordinates into local coordinates.

        Parameters
        ----------
        global_coords : array_like(Number, shape=(n, k))
            Represents `n` points of `k` dimensions in the global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(n, k))
            Local coordinates.

        Examples
        --------

        >>> T = matrix(t=[2, 3], s=[0.5, 10])
        >>> lcoords = T.to_local([(0, 0), (0, 1), (1, 0), (-1, -1)])
        >>> print(np.round(lcoords, 2))
        [[ 2.   3. ]
         [ 2.  13. ]
         [ 2.5  3. ]
         [ 1.5 -7. ]]

        """
        return transform(global_coords, self)

    def to_global(self, local_coords):
        """Transforms local coordinates into global coordinates.

        Parameters
        ----------
        local_coords : array_like(Number, shape=(n, k))
            Represents `n` points of `k` dimensions in the local coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(n, k))
            Global coordinates.

        Examples
        --------

        >>> T = matrix(t=[2, 3], s=[0.5, 10])
        >>> gcoords = T.to_global([(2, 3), (2, 13), (2.5, 3), (1.5, -7)])
        >>> print(np.round(gcoords, 2))
        [[ 0.  0.]
         [ 0.  1.]
         [ 1.  0.]
         [-1. -1.]]

        """
        return transform(local_coords, self, inverse=True)

    def explained_variance(self, global_coords):
        """Calculate the variance of global coordinates explained by the
        coordinate axes.

        Parameters
        ----------
        global_coords : array_like(Number, shape=(n, k))
            Represents 'n' points of `k` dimensions in the global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(self.dim))
            Total amount of explained variance by a specific coordinate axis.

        Examples
        --------

        >>> T = s_matrix([1, 2])
        >>> print(T)
        [[1. 0. 0.]
         [0. 2. 0.]
         [0. 0. 1.]]
        >>> e = T.explained_variance([(2, 1), (0, 0), (1, 1), (2, 3)])
        >>> print(np.round(e, 3))
        [0.688 4.75 ]

        See Also
        --------
        LocalSystem.explained_variance_ratio

        """
        global_coords = self.to_local(global_coords)
        return np.var(global_coords, axis=0)

    def explained_variance_ratio(self, global_coords):
        """Calculate the relative variance of global coordinates explained by
        the coordinate axes.

        Parameters
        ----------
        global_coords : array_like(Number, shape=(n, k))
            Represents 'n' points of `k` dimensions in the global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(self.dim))
            Relative amount of variance explained by each coordinate axis.

        Examples
        --------

        >>> T = s_matrix([1, 2])
        >>> print(T)
        [[1. 0. 0.]
         [0. 2. 0.]
         [0. 0. 1.]]
        >>> e = T.explained_variance_ratio([(2, 1), (0, 0), (1, 1), (2, 3)])
        >>> print(np.round(e, 3))
        [0.126 0.874]

        See Also
        --------
        LocalSystem.explained_variance

        """
        var = self.explained_variance(global_coords)
        return var / var.sum()


def eigen(coords):
    """Fit eigenvectors to coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions to fit the eigenvectors
        to.

    Returns
    -------
    eigen_vectors : np.ndarray(Number, shape=(k, k))
        Columnwise normalized eigenvectors in descending order of eigenvalue.
    eigen_values : np.ndarray(Number, shape=(k))
        Eigenvalues in descending order of magnitude.

    Notes
    -----
    Implementation idea taken from [1].

    References
    ----------
    [1] https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python

    See Also
    --------
    PCA

    Examples
    --------

    >>> coords = [(0, 0), (3, 4)]
    >>> eigen_vectors, eigen_values = eigen(coords)
    >>> print(eigen_vectors)
    [[ 0.6 -0.8]
     [ 0.8  0.6]]
    >>> print(eigen_values)
    [12.5  0. ]

    """
    coords = ensure_coords(coords)
    cCoords = coords - coords.mean(0)

    # calculate Eigenvectors and Eigenvalues
    cov_matrix = dot(cCoords.T, cCoords)
    # cov_matrix = cCoords.T @ cCoords  # fastest solution found
    # cov_matrix is a Hermitian matrix
    eigen_values, eigen_vectors = eigh(cov_matrix)
    # eigen_values in descending order ==> reverse

    return eigen_vectors[:, ::-1], eigen_values[::-1]


class PCA(LocalSystem):
    """Principal Component Analysis (PCA).

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions. These coordinates are
        used to fit a PCA.

    Attributes
    ----------
    eigen_values : np.ndarray(Number, shape=(k))
        Characteristic Eigenvalues of the PCA.

    Notes
    -----
    Implementation idea taken from [1].

    References
    ----------
    [1] https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python

    Examples
    --------

    >>> coords = [(0, 0), (3, 4)]
    >>> T = PCA(coords)
    >>> print(T)
    [[ 0.6  0.8 -2.5]
     [-0.8  0.6  0. ]
     [ 0.   0.   1. ]]
    >>> print(np.linalg.inv(T))
    [[ 0.6 -0.8  1.5]
     [ 0.8  0.6  2. ]
     [ 0.   0.   1. ]]
    >>> print(transform(coords, T))
    [[-2.5  0. ]
     [ 2.5  0. ]]

    """
    def __new__(cls, coords):
        # Do not edit anything!!!
        eigen_vectors, eigen_values = eigen(coords)
        center = np.mean(coords, axis=0)
        dim = len(center)

        T = LocalSystem(identity(dim + 1)).view(cls)
        T[:dim, :dim] = eigen_vectors.T
        T = T @ t_matrix(-center)  # do not edit!

        T._eigen_values = eigen_values

        return T

    @property
    def eigen_values(self):
        return self._eigen_values

    def pc(self, k):
        """Select a specific principal component.

        Parameters
        ----------
        k : positive int
            `k`-th principal component to return.

        Returns
        -------
        np.ndarray(Number, shape=(self.dim))
            `k`-th principal component.

        Examples
        --------

        >>> coords = [(-3, -4), (0, 0), (12, 16)]
        >>> T = PCA(coords)
        >>> print(T)
        [[ 0.6  0.8 -5. ]
         [-0.8  0.6  0. ]
         [ 0.   0.   1. ]]

        >>> print(T.pc(1))
        [0.6 0.8]
        >>> print(T.pc(2))
        [-0.8  0.6]

        """
        if not (k >= 1 and k <= self.dim):
            raise ValueError("%-'th principal component not available")

        return self[k - 1, :self.dim]
