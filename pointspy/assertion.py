"""Functions to ensure the properties of frequently used data structures.
"""

import numpy as np
from numbers import Number

from . import nptools


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

    Raises
    ------
    ValueError

    """
    return isinstance(value, Number) and value >= min_th and value <= max_th


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
    ValueError

    Examples
    --------

    >>> print ensure_numarray([0,1,2])
    [0 1 2]
    >>> print ensure_numarray((-4,-5))
    [-4 -5]

    """
    if not nptools.isarray(arr):
        raise ValueError("'arr' needs to an array like object")
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
        Minimum and maximum allowed length of the vector. If `length` is
        provided `n` has to be exactly `length`.

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
    vector of length 5 required

    Raises
    ------
    ValueError

    """
    v = ensure_numarray(v)
    if not len(v.shape) == 1:
        raise ValueError("one dimensional vector required")
    if length is not None:
        if not len(v) == length:
            raise ValueError("vector of length %i required" % length)
    else:
        if len(v) < min_length:
            raise ValueError("vector of length >= %i required" % min_length)
        if len(v) > max_length:
            raise ValueError("vector of length <= %i required" % max_length)
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
        Allowed coordinate dimensions. If `dim` is provided the dimension must
        match exactly. If not, the coordinate dimension must be in range
        `[min_dim, max_dim]`.

    Returns
    -------
    coords : np.ndarray(Number, shape=(n,k))
        Coordinates with guaranteed properties.

    Raises
    ------
    ValueError

    Examples
    --------

    Coordinates provided row by row.

    >>> coords = ensure_coords([(3,2),(2,4),(-1,2),(9,3)])
    >>> print type(coords)
    <type 'numpy.ndarray'>
    >>> print coords
    [[ 3  2]
     [ 2  4]
     [-1  2]
     [ 9  3]]

    Coordinates provided column by column.

    >>> coords = ensure_coords([(3,2,-1,9),(2,4,2,3)],by_col=True)
    >>> print coords
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
    if dim is not None:
        if not coords.shape[1] == dim:
            m = "%i coordinate dimensions required" % dim
            raise ValueError(m)
    else:
        if coords.shape[1] < min_dim:
            m = "at least %i coordinate dimensions required" % min_dim
            raise ValueError(m)
        if coords.shape[1] > max_dim:
            m = "at most %i coordinate dimensions required" % max_dim
            raise ValueError(m)
    return coords


def ensure_polar(pcoords, by_col=False):
    """Ensures the properties of polar coordinates.

    Parameters
    ----------
    pcoords : array_like(Number, shape=(n,k))
        Represents `n` data points of `k` dimensions in a polar coordinate
        system.
    by_col : optional, bool
        Defines weather or not the coordinates are provided column by column
        instead of row by row.

    Raises
    ------
    ValueError

    Returns
    -------
    pcoords : np.ndarray(Number, shape=(n,k))
        Polar coordinates with guaranteed properties.

    See Also
    --------
    ensure_coords

    """
    pcoords = ensure_coords(pcoords, by_col=by_col)
    if not np.all(pcoords[:, 0] >= 0):
        raise ValueError("malformed polar radii")
    return pcoords


def ensure_tmatrix(T, min_dim=2, max_dim=np.inf):
    """Ensures the properties of transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1,k+1))
        Transformation matrix.

    Returns
    -------
    T : np.matrix(Number, shape=(k+1,k+1))
        Transformation matrix with guaranteed properties.

    Raises
    ------
    ValueError

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
    if not T.shape[0] > min_dim:
        raise ValueError("at least %s coordinate dimensions needed" % min_dim)
    if not T.shape[0] <= max_dim + 1:
        raise ValueError("at most %s coordinate dimensions needed" % max_dim)

    return T
