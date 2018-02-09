import numpy as np


def norm(coords, squared=False):
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if len(coords.shape) == 1:
        res = (coords * coords).sum()
    else:
        res = (coords * coords).sum(1)
    if not squared:
        res = np.sqrt(res)
    return res


def dist(p, coords, squared=True):
    if not isinstance(p, np.ndarray):
        p = np.array(p)
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if len(p.shape) == 1:
        assert len(p) == coords.shape[1], 'Dimensions do not match!'
    else:
        assert p.shape[1] == coords.shape[1], 'Dimensions do not match!'
    return norm(coords - p, squared=squared)

# Quality


def rmse(A, B):
    return np.sqrt(np.mean(norm(A - B, squared=True)))


def IDW(dists, p=2):
    # inverse distance weights
    return 1.0 / (1 + dists)**p
