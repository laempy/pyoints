"""Provides functions to handle polar coordinates.
"""

import numpy as np
import distance

from . import assertion


def coords2polar(coords):
    """Converts euclidian coordinates to polar coordinates.

    Parameters
    ----------
    coords : (n,k), array_like
        Represents `n` data points of `k` dimensions in a euclidian coordinate
        system.

    Returns
    -------
    pcoords : (n,k), array_like
        Represents `n` data points of `k` dimensions in a polar coordinate
        system. First column represents the distance to origin of the
        coordinate system. All other columns represent the angles.

    Examples
    --------

    2D coordinates.

    >>> coords = [(0,0),(0,1),(1,0),(1,1),(-1,1),(2,-5)]
    >>> pcoords = coords2polar(coords)
    >>> print np.round(pcoords,3)
    [[ 0.     0.   ]
     [ 1.     1.571]
     [ 1.     0.   ]
     [ 1.414  0.785]
     [ 1.414  2.356]
     [ 5.385 -1.19 ]]

    3D coordinates.

    >>> coords = [(0,0,0),(1,1,0),(-1,-1,-1),(2,-5,9)]
    >>> pcoords = coords2polar(coords)
    >>> print np.round(pcoords,3)
    [[ 0.     0.     0.   ]
     [ 1.414  0.785  1.571]
     [ 1.732 -2.356  2.186]
     [10.488 -1.19   0.539]]

    """
    coords = assertion.ensure_coords(coords)

    dim = coords.shape[1]
    d = distance.norm(coords)
    if dim == 2:
        x = coords[:, 0]
        y = coords[:, 1]
        a = np.arctan2(y, x)
        return assertion.ensure_polar([d, a], by_row=True)
    elif dim == 3:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # avoid nan
        omega = np.zeros(len(d))
        mask = d > 0
        omega[mask] = np.arccos(z[mask] / d[mask])

        phi = np.arctan2(y, x)
        return assertion.ensure_polar([d, phi, omega], by_row=True)
    else:
        raise ValueError('%i dimensions are not supported yet.' % dim)


def polar2coords(pcoords):
    """Converts polar coordinates to euclidian coordinates.

    Parameters
    ----------
    pcoords : (n,k), array_like
        Represents `n` data points of `k` dimensions in a polar coordinate
        system. First column represents the distance to origin of the
        coordinate system. All other columns represent the angles.

    Returns
    -------
    coords : (n,k), array_like
        Represents `n` data points of `k` dimensions in a euclidian coordinate
        system.

    Examples
    --------

    2D coordinates.

    >>> pcoords = [(0,0),(3,0),(3,np.pi),(4,-0.5*np.pi),(1,0.5)]
    >>> coords = polar2coords(pcoords)
    >>> print np.round(coords,3)
    [[ 0.     0.   ]
     [ 3.     0.   ]
     [-3.     0.   ]
     [ 0.    -4.   ]
     [ 0.878  0.479]]

    2D coordinates.

    >>> pcoords = [(0,0,0),(2,0,0),(4,0,np.pi),(4,0.5*np.pi,0.5)]
    >>> coords = polar2coords(pcoords)
    >>> print np.round(coords,3)
    [[ 0.     0.     0.   ]
     [ 0.     0.     2.   ]
     [ 0.     0.    -4.   ]
     [ 0.     1.918  3.51 ]]

    """
    pcoords = assertion.ensure_polar(pcoords)

    dim = pcoords.shape[1]
    d = pcoords[:, 0]
    if dim == 2:
        a = pcoords[:, 1]
        x = d * np.cos(a)
        y = d * np.sin(a)
        return assertion.ensure_coords([x, y], by_row=True)
    elif dim == 3:
        phi = pcoords[:, 1]
        omega = pcoords[:, 2]
        x = d * np.sin(omega) * np.cos(phi)
        y = d * np.sin(omega) * np.sin(phi)
        z = d * np.cos(omega)
        return assertion.ensure_coords([x, y, z], by_row=True)
    else:
        raise ValueError('%i dimensions are not supported yet.' % dim)
