import numpy as np

def ensure_coords(coords, by_row=False):

    assert hasattr(coords, '__len__')

    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if by_row:
        coords = coords.T

    assert len(coords.shape) == 2
    assert coords.shape[1] > 0
    return coords


def ensure_polar(pcoords, by_row=False):
    pcoords = ensure_coords(pcoords, by_row=by_row)
    assert np.all(pcoords[:,0] >= 0)
    return pcoords


def ensure_tmatrix(T):

    assert hasattr(T, '__len__')
    if not isinstance(T, np.matrix):
        T = np.matrix(T)

    assert len(T.shape) == 2
    assert T.shape[0] == T.shape[1]
    assert T.shape[0] > 2

    return T
