"""Functions to ensure the properties of frequently used data structures.
"""

import numpy as np
from numbers import Number

from . import nptools


def ensure_dim(check_dim, dim=None, min_dim=2, max_dim=np.inf):
    """Ensure a dimension value to be in a specific range.

    Parameters
    ----------
    dim, min_dim, max_dim : optional, positive int
        Minimum and maximum allowed dimensions. If `dim` is provided the
        `check_dim` has to be exactly `dim`. If not, the `check_dim` must be in
        range `[min_dim, max_dim]`.

    Returns
    -------
    int
        Dimension value with ensured properties.

    Raises
    ------
    ValueError

    """
    check_dim = int(check_dim)
    if dim is not None:
        if not check_dim == dim:
            m = "%i dimensions required" % dim
            raise ValueError(m)
    else:
        if check_dim < min_dim:
            m = "at least %i dimensions required" % min_dim
            raise ValueError(m)
        if check_dim > max_dim:
            m = "at most %i dimensions required" % max_dim
            raise ValueError(m)
    return check_dim


def ensure_shape(shape, dim=None, min_dim=1, max_dim=np.inf):
    """Ensures the properties of an array shape.

    Parameters
    ----------
    shape : array_like(int, shape=(k))
        Shape of `k` dimensions to validate.

    Returns
    -------
    np.ndarray(int, shape=(k))
        Shape with ensured properties

    Raises
    ------
    ValueError, TypeError

    """
    if not nptools.isarray(shape):
        raise TypeError("'shape' needs to an array like object")
    shape = np.array(shape)
    if not nptools.isnumeric(shape, dtypes=[np.int32, np.int64]):
        raise ValueError("'shape' needs to have integer values")
    if not len(shape.shape) == 1:
        raise ValueError("'shape' needs to be a vector")
    if dim is not None:
        if not shape.shape[0] == dim:
            raise ValueError("'shape' requires a length of %i" % dim)
    else:
        if not (shape.shape[0] >= min_dim and shape.shape[0] <= max_dim):
            m = "length of 'shape' needs to be in range [%i, %i]"
            raise ValueError(m % (min_dim, max_dim))
    return shape


def ensure_length(check_length, length=None, min_length=0, max_length=np.inf):
    """Ensure a length value to be in a specific range.

    Parameters
    ----------
    check_length : int
        Length value to check.
    length, min_length, max_length : optional, positive int
        Minimum and maximum allowed length. If `length` is provided
        `check_length` has to be exactly `length`. If not, the `check_length`
        must be in range `[min_length, max_length]`.

    Returns
    -------
    int
        Length value with ensured properties.

    Raises
    ------
    ValueError

    """
    if not isinstance(check_length, int):
        raise TypeError("'check_length' needs to be an integer")
    if length is not None:
        if not check_length == length:
            m = "length %i required" % length
            raise ValueError(m)
    else:
        if check_length < min_length:
            m = "length of at least %i required" % min_length
            raise ValueError(m)
        if check_length > max_length:
            m = "length of at most %i required" % max_length
            raise ValueError(m)
    return check_length


def isnumeric(value, min_th=-np.inf, max_th=np.inf):
    """Checks if a value is numeric.

    Parameters
    ----------
    value : Number
        Value to validate.
    min_th, max_th : optional, Number
        Minimum and maximum value allowed range.

    Returns
    -------
    bool
        Indicates whether or not the value is numeric.

    """
    return isinstance(value, Number) and value >= min_th and value <= max_th


def iscoord(coord):
    """Checks if a value is associated with a coordinate.

    Parameters
    ----------
    coord : array_like
        Value associated with a coordinate.

    Returns
    -------
    bool
        Indicates whether or not the value is a coordinate.

    """
    return (hasattr(coord, '__len__') and
            len(coord) > 0 and
            not hasattr(coord[0], '__len__'))


def ensure_numarray(arr):
    """Ensures the properties of an numeric numpy ndarray.

    Parameters
    ----------
    arr : array_like(Number)
        Array like numeric object.

    Returns
    -------
    np.ndarray(Number)
        Array with guaranteed properties.

    Raises
    ------
    TypeError, ValueError

    Examples
    --------

    >>> print(ensure_numarray([0,1,2]))
    [0 1 2]
    >>> print(ensure_numarray((-4,-5)))
    [-4 -5]

    """
    if not nptools.isarray(arr):
        raise TypeError("'arr' needs to an array like object")
    arr = np.array(arr)
    if not nptools.isnumeric(arr):
        raise ValueError("array 'arr' needs to be numeric")
    return arr


def ensure_numvector(v, length=None, min_length=1, max_length=np.inf):
    """Ensures the properties of a numeric vector.

    Parameters
    ----------
    v : array_like(Number, shape=(k))
        Vector of length `n`.
    length, min_length, max_length : optional, positive int
        See `ensure_length`

    Returns
    -------
    v : np.ndarray(Number, shape=(n))
        Vector with guaranteed properties.

    Examples
    --------

    Valid vector.

    >>> v = (3, 2, 4, 4)
    >>> v = ensure_numvector(v)
    >>> print(v)
    [3 2 4 4]

    Vector of insufficient length.

    >>> try:
    ...     ensure_numvector(v, length=5)
    ... except ValueError as e:
    ...     print(e)
    length 5 required

    Raises
    ------
    TypeError, ValueError

    """
    v = ensure_numarray(v)
    if not len(v.shape) == 1:
        raise TypeError("one dimensional vector required")
    ensure_length(len(v), length, min_length, max_length)
    return v


def ensure_indices(v, min_value=0, max_value=np.inf):
    """Ensures an index array to be in a specific range.

    Parameters
    ----------
    v : array_like(int, shape=(n))
        Arrray of indices to check.
    min_value : optional, int
        Minumum allowed value of `v`.
    min_value : optional, int
        Maximum allowed value of `v`.

    Returns
    -------
    np.ndarray(int, shape=(n))
        Array of indices.

    Raises
    ------
    TypeError, ValueError

    """
    v = ensure_numvector(v)
    if v.dtype.kind not in ('i', 'u'):
        raise ValueError('integer array required')
    if not v.max() <= max_value:
        m = "index %i out of range [%i, %i]" % (v.max(), min_value, max_value)
        raise ValueError(m)
    if not v.min() >= min_value:
        m = "index %i out of range [%i, %i]" % (v.min(), min_value, max_value)
        raise ValueError(m)
    return v


def ensure_coords(coords, by_col=False, dim=None, min_dim=2, max_dim=np.inf):
    """Ensures all required properties of a coordinate like array.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.
    by_col : optional, bool
        Defines weather or not the coordinates are provided column by column
        instead of row by row.
    dim, min_dim, max_dim : optional, positive int
        See `ensure_dim`

    Returns
    -------
    coords : np.ndarray(Number, shape=(n, k))
        Coordinates with guaranteed properties.

    Raises
    ------
    TypeError, ValueError

    Examples
    --------

    Coordinates provided row by row.

    >>> coords = ensure_coords([(3, 2), (2, 4), (-1, 2), (9, 3)])
    >>> print(isinstance(coords, np.ndarray))
    True
    >>> print(coords)
    [[ 3  2]
     [ 2  4]
     [-1  2]
     [ 9  3]]

    Coordinates provided column by column.

    >>> coords = ensure_coords([(3, 2, -1, 9), (2, 4, 2, 3)], by_col=True)
    >>> print(coords)
    [[ 3  2]
     [ 2  4]
     [-1  2]
     [ 9  3]]

    See Also
    --------
    ensure_polar

    """
    coords = ensure_numarray(coords)
    if by_col:
        coords = coords.T
    if not len(coords.shape) == 2:
        m = "malformed shape of 'coords', got '%s'" % str(coords.shape)
        raise ValueError(m)
    ensure_dim(coords.shape[1], dim, min_dim, max_dim)
    return coords


def ensure_polar(pcoords, by_col=False, dim=None, min_dim=2, max_dim=np.inf):
    """Ensures the properties of polar coordinates.

    Parameters
    ----------
    pcoords : array_like(Number, shape=(n,k))
        Represents `n` data points of `k` dimensions in a polar coordinate
        system.
    by_col : optional, bool
        Defines weather or not the coordinates are provided column by column
        instead of row by row.
    dim, min_dim, max_dim : optional, positive int
        See `ensure_dim`

    Raises
    ------
    TypeError, ValueError

    Returns
    -------
    pcoords : np.ndarray(Number, shape=(n,k))
        Polar coordinates with guaranteed properties.

    See Also
    --------
    ensure_coords

    """
    pcoords = ensure_coords(
        pcoords,
        by_col=by_col,
        dim=dim,
        min_dim=min_dim,
        max_dim=max_dim
    )
    if not np.all(pcoords[:, 0] >= 0):
        raise ValueError("malformed polar radii")
    return pcoords


def ensure_tmatrix(T, dim=None, min_dim=2, max_dim=np.inf):
    """Ensures the properties of transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1,k+1))
        Transformation matrix.
    dim, min_dim, max_dim : optional, positive int
        See `ensure_dim`

    Returns
    -------
    T : np.matrix(Number, shape=(k+1,k+1))
        Transformation matrix with guaranteed properties.

    Raises
    ------
    TypeError, ValueError

    See Also
    --------
    transformation.matrix

    """
    if not nptools.isarray(T):
        raise ValueError("transformation matrix is not an array")
    if not isinstance(T, np.matrix):
        T = np.matrix(T)

    if not nptools.isnumeric(T):
        raise ValueError("'T' needs to be numeric")
    if not len(T.shape) == 2:
        raise ValueError("malformed shape of transformation matrix")
    if not T.shape[0] == T.shape[1]:
        raise ValueError("transformation matrix is not a square matrix")
    ensure_dim(T.shape[0] - 1, dim, min_dim, max_dim)

    return T
