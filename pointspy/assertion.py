import numpy as np

def ensure_coords(coords, by_row=False):
    if not hasattr(coords, '__len__'):
        raise ValueError("coords has no length")

    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if by_row:
        coords = coords.T

    if not len(coords.shape) == 2:
        raise ValueError("malformed shape of coordinates")
    if not coords.shape[1] > 1:
        raise ValueError("at least two coordinate dimensions needed")

    return coords


def ensure_polar(pcoords, by_row=False):
    pcoords = ensure_coords(pcoords, by_row=by_row)
    if not np.all(pcoords[:, 0] >= 0):
        raise ValueError("malformed polar radii")
    return pcoords


def ensure_tmatrix(T):

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
