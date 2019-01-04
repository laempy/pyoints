# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University,
# lamprecht@uni-trier.de
#
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""Handling of polar coordinates.
"""

import numpy as np

from .import (
    distance,
    assertion,
)
from .misc import print_rounded


def coords_to_polar(coords):
    """Converts Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.

    Returns
    -------
    pcoords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a polar coordinate
        system. First column represents the distance to the origin of the
        coordinate system. All other columns represent the corresponding
        axes angles.

    See Also
    --------
    polar_to_coords

    Examples
    --------

    2D coordinates.

    >>> coords = [(0, 0), (0, 1), (1, 0), (1, 1), (-1, 1), (2, -5)]
    >>> pcoords = coords_to_polar(coords)
    >>> print_rounded(pcoords, 3)
    [[ 0.     0.   ]
     [ 1.     1.571]
     [ 1.     0.   ]
     [ 1.414  0.785]
     [ 1.414  2.356]
     [ 5.385 -1.19 ]]

    3D coordinates.

    >>> coords = [(0, 0, 0), (1, 1, 0), (-1, -1, -1), (2, -5, 9)]
    >>> pcoords = coords_to_polar(coords)
    >>> print_rounded(pcoords, 3)
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
        return assertion.ensure_polar([d, a], by_col=True)
    elif dim == 3:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # avoid nan
        omega = np.zeros(len(d))
        mask = d > 0
        omega[mask] = np.arccos(z[mask] / d[mask])

        phi = np.arctan2(y, x)
        return assertion.ensure_polar([d, phi, omega], by_col=True)
    else:
        raise ValueError('%i dimensions are not supported yet.' % dim)


def polar_to_coords(pcoords):
    """Converts polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    pcoords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a polar coordinate
        system. First column represents the distance to the origin of the
        coordinate system. All other columns represent the corresponding
        axes angles.

    Returns
    -------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.

    See Also
    --------
    coords_to_polar

    Examples
    --------

    2D coordinates.

    >>> pcoords = [(0, 0), (3, 0), (3, np.pi), (4, -0.5*np.pi), (1, 0.5)]
    >>> coords = polar_to_coords(pcoords)
    >>> print_rounded(coords, 3)
    [[ 0.     0.   ]
     [ 3.     0.   ]
     [-3.     0.   ]
     [ 0.    -4.   ]
     [ 0.878  0.479]]

    3D coordinates.

    >>> pcoords = [(0, 0, 0), (2, 0, 0),(4, 0, np.pi), (4, 0.5*np.pi, 0.5)]
    >>> coords = polar_to_coords(pcoords)
    >>> print_rounded(coords, 3)
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
        return assertion.ensure_coords([x, y], by_col=True)
    elif dim == 3:
        phi = pcoords[:, 1]
        omega = pcoords[:, 2]
        x = d * np.sin(omega) * np.cos(phi)
        y = d * np.sin(omega) * np.sin(phi)
        z = d * np.cos(omega)
        return assertion.ensure_coords([x, y, z], by_col=True)
    else:
        raise ValueError('%i dimensions are not supported yet' % dim)
