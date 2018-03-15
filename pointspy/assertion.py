"""Functions to ensure the properties of frequently used data structures.
"""

import numpy as np


def _isnumeric(nparray):
     return np.issubdtype(nparray.dtype.type,np.int64) or np.issubdtype(nparray.dtype.type,np.float64)

def ensure_coords(coords, by_col=False):
    """Ensures all required properties of a coordinate like array.

    Parameters
    ----------
    coords : array_like(Number, shape=(n,k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.
    by_col : optional, bool
        Defines weather or not the coordinates are provided column by column
        instead of row by row.

    Returns
    -------
    coords : np.ndarray(Number, shape=(n,k))
        Coordinates with guaranteed properties.

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
    if not hasattr(coords, '__len__'):
        raise ValueError("coords has no length")

    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if by_col:
        coords = coords.T
    if not len(coords.shape) == 2:
        raise ValueError("malformed shape of 'coords', got '%s'"%str(coords.shape))
    if not coords.shape[1] > 1:
        raise ValueError("at least two coordinate dimensions needed")
    if not _isnumeric(coords):
        raise ValueError("numeric values of 'coords' reqired")

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


def ensure_tmatrix(T):
    """Ensures the properties of transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1,k+1))
        Transformation matrix.

    Returns
    -------
    T : np.matrix(Number, shape=(k+1,k+1))
        Transformation matrix with guaranteed properties.

    See Also
    --------
    transformation.matrix

    """

    if not hasattr(T, '__len__'):
        raise ValueError("transformation matrix has no length")
    if not isinstance(T, np.matrix):
        T = np.matrix(T)

    if not len(T.shape) == 2:
        raise ValueError("malformed shape of transformation matrix")
    if not T.shape[0] == T.shape[1]:
        raise ValueError("transformation matrix is not a square matrix")
    if not T.shape[0] > 2:
        raise ValueError("at least two coordinate dimensions needed")

    return T


def ensure_vector(v):
    """Ensures the properties of a vector.

    Parameters
    ----------
    v : array_like(Number, shape=(k))
        Vector of `k` dimensions.

    Returns
    -------
    v : np.ndarray(Number, shape=(k))
        Vector with guaranteed properties.

    """
    if not hasattr(v, '__getitem__'):
        raise ValueError("'v' needs to an array like object")
    v = np.array(v)
    if not len(v.shape) == 1:
        raise ValueError("malformed shape of vector 'v'")
    if not _isnumeric(v):
        raise ValueError("vector 'v' is not numeric")
    return v
