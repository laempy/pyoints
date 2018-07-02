"""Distance metrics.
"""

import numpy as np

from . import assertion


def norm(coords):
    """Normalization of coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Represents `n` points or a single point of `k` dimensions.

    Returns
    -------
    array_like(shape=(n, ))
        Normed values.

    See Also
    --------
    snorm

    """
    return np.sqrt(snorm(coords))


def snorm(coords):
    """Squared normalization of coordinates.

    Parameters
    ----------
    coords: array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Represents `n` points or a single point of `k` dimensions.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Squared normed values.

    See Also
    --------
    norm

    """
    coords = assertion.ensure_numarray(coords)
    if len(coords.shape) == 1:
        res = (coords * coords).sum()
    else:
        res = (coords * coords).sum(1)
    return res


def dist(p, coords):
    """Calculates the distances between points.

    Parameters
    ----------
    p : array_like(Number, shape=(n, k)) or array_like(Number, shape=(k))
        Represents `n` points or a single point of `k` dimensions.
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Normed values.

    See Also
    --------
    sdist

    """
    return np.sqrt(sdist(p, coords))


def sdist(p, coords):
    """Calculates the squared distances between points.

    Parameters
    ----------
    p : array_like(Number, shape=(n, k)) or array_like(Number, shape=(k))
        Represents `n` points or a single point of `k` dimensions.
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Squared distances between the points.

    See Also
    --------
    dist

    """
    p = assertion.ensure_numarray(p)
    coords = assertion.ensure_coords(coords)
    if len(p.shape) == 1:
        if not len(p) == coords.shape[1]:
            raise ValueError('Dimensions do not match!')
    else:
        if not p.shape == coords.shape:
            m = "Dimensions %s and %s do not match"
            raise ValueError(m % (str(p.shape), str(coords.shape)))

    return snorm(coords - p)


def rmse(A, B):
    """Calculates the Root Mean Squared Error of corresponding data sets.

    Parameters
    ----------
    A, B : array_like(Number, shape=(n, k))
        Represent `n` points or a single point of `k` dimensions.

    Returns
    -------
    Number
        Root Mean Squared Error.

    """
    return np.sqrt(np.mean(sdist(A, B)))


def idw(dists, p=2):
    """Calculates the weights for Inverse Distance Weighting method.

    Parameters
    ----------
    dists : Number or array_like(Number, shape=(n))
        Represent `n` distance values.
    p : optional, Number
        Weighting power.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Weights according to Inverse Distance Weighting.

    """
    return 1.0 / (1 + dists)**p
