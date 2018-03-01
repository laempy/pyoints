import numpy as np

# TODO module description

def norm(coords):
    # TODO
    """ Normalization of coordinates.
    
    Parameters
    ----------
    coords: (n,k), `array_like`
        Represents n data points of k dimensions.
    
    Returns
    -------
    norm: (n), `array_like`
        TODO 
    """
    return np.sqrt(snorm(coords))

def snorm(coords):
    # TODO
    """ Normalization of coordinates.
    
    Parameters
    ----------
    coords: (n,k), `array_like`
        Represents n data points of k dimensions.
    
    Returns
    -------
    snorm: (n), `array_like`
        TODO 
    """
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if len(coords.shape) == 1:
        res = (coords * coords).sum()
    else:
        res = (coords * coords).sum(1)
    return res

def dist(p, coords):
    return np.sqrt(sdist(p, coords))


def sdist(p, coords):
    if not isinstance(p, np.ndarray):
        p = np.array(p)
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if len(p.shape) == 1:
        assert len(p) == coords.shape[1], 'Dimensions do not match!'
    else:
        assert p.shape[1] == coords.shape[1], 'Dimensions do not match!'
    return snorm(coords - p)

# Quality


def rmse(A, B):
    return np.sqrt(np.mean(sdist(A,B)))


def idw(dists, p=2):
    # inverse distance weights
    return 1.0 / (1 + dists)**p
