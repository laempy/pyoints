import numpy as np
import distance


def coords2polar(coords):
    dim = coords.shape[1]
    d = distance.norm(coords)
    if dim == 2:
        x = coords[:, 0]
        y = coords[:, 1]
        d = distance.norm2D(coords)
        a = np.arctan2(y, x)
        return np.array((d, a)).T
    elif dim == 3:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        omega = np.arccos(z / d)
        phi = np.arctan2(y, x)
        return np.array((d, phi, omega)).T
    else:
        raise ValueError('%i dimensions are not supported yet.' % dim)


def polar2coords(polar):
    dim = polar.shape[1]
    d = polar[:, 0]
    if dim == 2:
        a = polar[:, 1]
        x = d * np.cos(a)
        y = d * np.sin(a)
        return np.array((x, y)).T
    elif dim == 3:
        phi = polar[:, 1]
        omega = polar[:, 2]
        x = d * np.sin(omega) * np.cos(phi)
        y = d * np.sin(omega) * np.sin(phi)
        z = d * np.cos(omega)
        return np.array((x, y, z)).T
    else:
        raise ValueError('%i dimensions are not supported yet.' % dim)
