"""Multidimensional transformation matrices and coordinate transformations.
"""

from numbers import Number
import numpy as np
import itertools as it
import warnings

from . import (
    distance,
    assertion,
)


def transform(coords, T, inverse=False):
    """Performs a linear transformation to coordinates using a transformation
    matrix.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents n data points of `k` dimensions.
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix.

    Returns
    -------
    coords : np.ndarray(Number, shape=(n, k))
        Transformed coordinates.

    """
    T = assertion.ensure_tmatrix(T)

    if inverse:
        try:
            T = np.linalg.inv(T)
        except np.linalg.LinAlgError, e:
            warnings.warn(e.message)
            T = np.linalg.pinv(T)

    T = np.asarray(T)

    H = homogenious(coords)
    HT = np.dot(H, T.T)
    if len(H.shape) == 1:
        return HT[0:-1]
    else:
        return HT[:, 0:-1]


def homogenious(coords, value=1):
    """Create homogenious coordinates by adding an additional dimension.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents n data points of `k` dimensions.
    value : optional, Number
        Values to set.

    Returns
    -------
    hcoords : np.ndarray(Number, shape=(n, k+1))
        Homogenious coordinates.

    """
    coords = assertion.ensure_coords(coords)
    if not isinstance(value, Number):
        raise ValueError("'value' needs to be numeric")

    if len(coords.shape) == 1:
        H = np.append(coords, value)
    else:
        N, dim = coords.shape
        H = np.empty((N, dim + 1))
        H[:, :-1] = coords
        H[:, -1] = value
    return H


def matrix(t=None, r=None, s=None, order='trs'):
    """Creates a transformation matrix based on translation, rotation and scale
    coefficients.

    Parameters
    ----------
    t, r, s : optional, np.ndarray(Number, shape=(k))
        Transformation coefficients for translation `t`, rotation `r` and
        scale `s`.
    order : optional, string
        Order to compute the matrix multiplications. A `order` of 'trs' means
        first translate `t`, then rotate `r` and finally scale `s`.

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
        raise ValueError("order needs to be of type string")
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

    # doble check the shape
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
        M = M * matrices[key]

    return LocalSystem(M)


def i_matrix(dim):
    """Creates a identity transformation matrix.

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

    >>> print i_matrix(3)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    """
    if not (isinstance(dim, int) and dim >= 2):
        raise ValueError("'dim' needs to be an integer greater one")
    I = np.matrix(np.identity(dim + 1))
    return LocalSystem(I)


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

    >>> print t_matrix([1, 2, 3])
    [[1. 0. 0. 1.]
     [0. 1. 0. 2.]
     [0. 0. 1. 3.]
     [0. 0. 0. 1.]]

    """
    if not (hasattr(t, '__len__') and len(t) >= 2):
        raise ValueError("'t' needs have a length greater one")
    dim = len(t)
    T = np.identity(dim + 1)
    T[0:dim, dim] = t
    return LocalSystem(T)


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

    >>> print s_matrix([0.5, 1, -3])
    [[ 0.5  0.   0.   0. ]
     [ 0.   1.   0.   0. ]
     [ 0.   0.  -3.   0. ]
     [ 0.   0.   0.   1. ]]

    """
    if not (hasattr(s, '__len__') and len(s) >= 2):
        raise ValueError("'s' needs have a length greater one")
    dim = len(s)
    S = np.identity(dim + 1)
    diag = np.append(s, 1)
    np.fill_diagonal(S, diag)
    return LocalSystem(S)


def r_matrix(a):
    """Creates a rotation matrix.

    Parameters
    ----------
    r : Number or array_like(Number, shape=(k))
        Rotation coefficients for each coordinate dimension. At least two
        coefficients are required.

    Returns
    -------
    LocalSystem(shape=(k+1, k+1))
        Rotation matrix.

    See Also
    --------
    LocalSystem

    Notes
    -----
    Currently only two and tree coordinate dimensions are supported yet.

    Examples
    --------

    Two dimensional case.

    >>> R = r_matrix(np.pi/4)
    >>> print np.round(R, 3)
    [[ 0.707 -0.707  0.   ]
     [ 0.707  0.707  0.   ]
     [ 0.     0.     1.   ]]

    Three dimensional case.

    >>> R = r_matrix([np.pi/2, 0, np.pi/4])
    >>> print np.round(R, 3)
    [[ 0.707 -0.     0.707  0.   ]
     [ 0.707  0.    -0.707  0.   ]
     [ 0.     1.     0.     0.   ]
     [ 0.     0.     0.     1.   ]]

    """
    if isinstance(a, (float, int)):
        R = np.matrix([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])
    elif hasattr(a, '__getitem__'):
        a = assertion.ensure_numvector(a)
        if len(a) == 2:
            raise ValueError('rotation in 2D requires one angle only')
        elif len(a) == 3:
            Rx = np.matrix([
                [1, 0, 0, 0],
                [0, np.cos(a[0]), -np.sin(a[0]), 0],
                [0, np.sin(a[0]), np.cos(a[0]), 0],
                [0, 0, 0, 1],
            ])
            Ry = np.matrix([
                [np.cos(a[1]), 0, np.sin(a[1]), 0],
                [0, 1, 0, 0],
                [-np.sin(a[1]), 0, np.cos(a[1]), 0],
                [0, 0, 0, 1],
            ])
            Rz = np.matrix([
                [np.cos(a[2]), -np.sin(a[2]), 0, 0],
                [np.sin(a[2]), np.cos(a[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            R = Rz * Ry * Rx
        else:
            raise ValueError(
                '%i-dimensional rotations are not supported yet' %
                len(a))
    else:
        raise ValueError("'r' needs be numeric or an iterable of numbers")

    return LocalSystem(R)


def add_dim(T):
    """Adds a dimension to a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix of `k` coordinate dimensions.

    Returns
    -------
    np.matrix(float, shape=(k+2, k+2))
        Transformation matrix with an additional dimension.


    Examples
    --------

    Two dimensional case.

    >>> T = matrix(t=[2, 3 ], s=[0.5, 2])
    >>> print np.round(T, 3)
    [[0.5 0.  2. ]
     [0.  2.  3. ]
     [0.  0.  1. ]]
    >>> T = add_dim(T)
    >>> print np.round(T, 3)
    [[0.5 0.  0.  2. ]
     [0.  2.  0.  3. ]
     [0.  0.  1.  0. ]
     [0.  0.  0.  1. ]]

    """
    T = assertion.ensure_tmatrix(T)
    M = np.eye(len(T)+1)
    M[:-2, :-2] = T[:-1, :-1]
    M[:-2, -1] = T[:-1, -1].T
    return M


def decomposition(T):
    """Determinates most important parameters of a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix of `k` coordinate dimensions.

    Returns
    -------
    t, r, s : optional, np.ndarray(Number, shape=(k))
        Transformation coefficients for translation `t`, rotation `r` and
        scale `s`.
    det : float
        Derterminant `det` indicates a distorsion of `T`, if not `det` == 1.

    Examples
    --------

    >>> T = matrix(t=[2, 3], r=0.2, s=[0.5, 2])
    >>> t, r, s, det = decomposition(T)
    >>> print t
    [2. 3.]
    >>> print r
    0.2
    >>> print s
    [0.5 2. ]
    >>> print det
    1.0

    See Also
    --------
    matrix

    References
    ----------
    https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
    https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix/13165#13165

    """

    T = assertion.ensure_tmatrix(T)
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
        assert np.isclose(r1, r2), 'Rotation angles seem to differ.'
        r = (r1 + r2) * 0.5
    elif dim == 3:
        r_x = np.arctan(R[2, 1] / R[2, 2])
        r_y = -np.arcsin(R[2, 0])
        r_z = np.arctan(R[1, 0] / R[0, 0])
        r = np.array([r_x, r_y, r_z])
    else:
        raise ValueError('Only %s dimensions are not supported jet' % dim)

    # determinant
    det = np.linalg.det(T)

    return t, r, s, det


class LocalSystem(np.matrix, object):
    """Defines a local coordinate system based on a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix of `k` coordinate dimensions.

    Attributes
    ----------
    dim : positive int
        Number of coordinate dimensions.
    components : np.ndarray(Number, shape=(k+1, k+1))
        Components of the local coordinate system. Each component represents
        a the direction vector of the coordinate axis in the global coordinate
        system.
    origin :
        Global origin of the local coordinate system.

    Examples
    --------

    >>> T = assertion.ensure_tmatrix([(2, -0.5, 2), (0.5, 1, 3), (0, 0, 1)])
    >>> print  T
    [[ 2.  -0.5  2. ]
     [ 0.5  1.   3. ]
     [ 0.   0.   1. ]]
    >>> L = LocalSystem(T)
    >>> L.dim
    2
    >>> print np.round(L.components, 3)
    [[ 2.  -0.5]
     [ 0.5  1. ]]
    >>> print np.round(L.origin, 3)
    [2. 3.]

    See Also
    --------
    matrix

    """

    def __new__(cls, T):
        return assertion.ensure_tmatrix(T).view(cls)

    @property
    def dim(self):
        return len(self) - 1

    # TODO notwendig?
    @property
    def components(self):
        # TODO here might be a sign error
        return np.asarray(self[:self.dim, :self.dim])

    @property
    def origin(self):
        return np.asarray(self[:self.dim, self.dim]).T[0]

    def to_local(self, gcoords):
        """Transforms global coordinates into local coordinates.

        Parameters
        ----------
        gcoords : array_like(Number, shape=(n, k))
            Represents n data points of `k` dimensions in the global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(n, k))
            Local coordinates.

        Examples
        --------

        >>> T = matrix(t=[2, 3], s=[0.5, 10])
        >>> lcoords = T.to_local([(0, 0), (0, 1), (1, 0), (-1, -1)])
        >>> print lcoords
        [[ 2.   3. ]
         [ 2.  13. ]
         [ 2.5  3. ]
         [ 1.5 -7. ]]

        """
        return transform(gcoords, self)

    def to_global(self, lcoords):
        """Transforms local coordinates into global coordinates.

        Parameters
        ----------
        lcoords : array_like(Number, shape=(n, k))
            Represents n data points of `k` dimensions in the local coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(n, k))
            Global coordinates.

        Examples
        --------

        >>> T = matrix(t=[2, 3], s=[0.5, 10])
        >>> lcoords = T.to_global([(2, 3), (2, 13), (2.5, 3), (1.5, -7)])
        >>> print lcoords
        [[ 0.  0.]
         [ 0.  1.]
         [ 1.  0.]
         [-1. -1.]]

        """
        return transform(lcoords, self, inverse=True)

    # TODO auslagern?
    def explained_variance(self, gcoords):
        """Get explained variance of global coordinates.

        Parameters
        ----------
        gcoords : array_like(Number, shape=(n, k))
            Represents n data points of `k` dimensions in the global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(self.dim))
            Total amount of explained variance by a specific coordinate axis.

        Examples
        --------

        >>> T = s_matrix([1, 2])
        >>> print T
        [[1. 0. 0.]
         [0. 2. 0.]
         [0. 0. 1.]]
        >>> e = T.explained_variance([(2, 1), (0, 0), (1, 1), (2, 3)])
        >>> print np.round(e, 3)
        [0.688 4.75 ]

        See Also
        --------
        LocalSystem.explained_variance_ratio

        """
        lcoords = self.to_local(gcoords)
        return np.var(lcoords, axis=0)

    # TODO auslagern?
    def explained_variance_ratio(self, gcoords):
        """Get relative amount of explained variance of global coordinates.

        Parameters
        ----------
        gcoords : array_like(Number, shape=(n, k))
            Represents n data points of `k` dimensions in the global coordinate
            system.

        Returns
        -------
        np.ndarray(Number, shape=(self.dim))
            Relative amount of explained variance by a specific coordinate
            axis.

        Examples
        --------

        >>> T = s_matrix([1, 2])
        >>> print T
        [[1. 0. 0.]
         [0. 2. 0.]
         [0. 0. 1.]]
        >>> e = T.explained_variance_ratio([(2, 1), (0, 0), (1, 1), (2, 3)])
        >>> print np.round(e, 3)
        [0.126 0.874]

        See Also
        --------
        LocalSystem.explained_variance

        """
        var = self.explained_variance(gcoords)
        return var / var.sum()
