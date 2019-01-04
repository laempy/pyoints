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
"""Coordinate Reference Systems and two dimensional geograpic coordinate
transformations.
"""

import pyproj
import numpy as np
from osgeo import osr

from . import (
    assertion,
)
from .misc import print_rounded


# Global proj4 definitions
WGS84 = '+proj=latlong +datum=WGS84 +to +proj=latlong +datum=WGS84 +units=m ' \
    '+no_defs'


class Proj:
    """Wrapper class for different commonly coordinate reference system
    formats.

    Parameters
    ----------
    proj4 : optional, str
        Coordinate reference system definition in Proj4 format. WGS84, if None
        or empty string.

    Attributes
    ----------
    proj4 : str
        Projection in Proj4 format.
    wkt : str
        Projection in Well Known Text format.
    pyproj : `pyproj.Proj`
        Projection as `pyproj.Proj` instance.

    Examples
    --------

    Create from EPSG code.

    >>> proj = Proj.from_epsg(4326)
    >>> print('AUTHORITY["EPSG","4326"]' in proj.wkt)
    True
    >>> print('WGS84' in proj.proj4)
    True

    Create from Proj4 string.

    >>> proj = Proj.from_proj4('+proj=longlat +datum=WGS84 +no_defs')
    >>> print(proj.proj4)
    +proj=longlat +datum=WGS84 +no_defs
    >>> print('AUTHORITY["EPSG","4326"]' in proj.wkt)
    True

    Create from Well Known Text.

    >>> proj = Proj.from_wkt(proj.wkt)
    >>> print('WGS84' in proj.proj4)
    True

    """

    def __init__(self, proj4=WGS84):
        if proj4 is None or not isinstance(proj4, str) or proj4 is '':
            raise ValueError("proj4 not defined")
        self._proj4 = proj4.rstrip()

    @property
    def proj4(self):
        return self._proj4

    @property
    def wkt(self):
        sr = osr.SpatialReference()
        sr.ImportFromProj4(self.proj4)
        return sr.ExportToWkt()

    @property
    def pyproj(self):
        return pyproj.Proj(self.proj4)

    def __str__(self):
        return 'proj4: %s' % str(self.proj4)

    @classmethod
    def from_proj4(cls, proj4):
        """Create `Proj` object from Proj4 format.

        Parameters
        ----------
        proj4 : str
            Coordinate projection definition in Proj4 format.

        """
        return Proj(proj4)

    @classmethod
    def from_wkt(cls, wkt):
        """Create `Proj` object from Well Known Text.

        Parameters
        ----------
        wkt : str
            Coordinate projection definition in Well Known Text format.

        """
        if not isinstance(wkt, str):
            raise TypeError("'wkt' needs to be a string")
        proj4 = osr.SpatialReference(wkt=wkt).ExportToProj4()
        if proj4 is '':
            raise ValueError("WKT unknown")
        return Proj.from_proj4(proj4)

    @classmethod
    def from_epsg(cls, epsg):
        """Create `Proj` object from EPSG code.

        Parameters
        ----------
        epsg : int
            Coordinate projection definition in EPSG format.

        """
        if not isinstance(epsg, int):
            raise TypeError("'epsg' needs to be an integer")
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        proj4 = sr.ExportToProj4()
        if proj4 is '':
            raise ValueError("epsg code '%i' unknown" % epsg)
        return Proj.from_proj4(proj4)


class GeoTransform:
    """Provides a coordinate transformation between different spatial reference
    systems.

    Parameters
    ----------
    from_proj,to_proj : `Proj`
        Define the coordinate transformation from the origin projection system
        `from_proj` to the target projection system `to_proj`.

    Examples
    --------

    Transform coordinates.

    >>> wgs84 = Proj.from_epsg(4326)
    >>> gk2 = Proj.from_epsg(31466)
    >>> coords = [
    ...     (6.842, 49.971),
    ...     (6.847, 49.969),
    ...     (6.902, 49.991),
    ...     (6.922, 50.101)
    ... ]
    >>> geoTransfrom = GeoTransform(wgs84, gk2)

    >>> tCoords = geoTransfrom(coords)
    >>> print_rounded(tCoords, 3)
    [[2560446.801 5537522.386]
     [2560808.009 5537303.984]
     [2564724.211 5539797.116]
     [2566007.32  5552049.646]]

    Reverse transformation.

    >>> print_rounded(geoTransfrom(tCoords, reverse=True), 3)
    [[ 6.842 49.971]
     [ 6.847 49.969]
     [ 6.902 49.991]
     [ 6.922 50.101]]

    """

    def __init__(self, from_proj, to_proj):
        self.from_proj = from_proj
        self.to_proj = to_proj

    @property
    def from_proj(self):
        return self._from_proj

    @from_proj.setter
    def from_proj(self, proj):
        if not isinstance(proj, Proj):
            raise TypeError("isinstance of 'Proj' required")
        self._from_proj = proj

    @property
    def to_proj(self):
        return self._to_proj

    @to_proj.setter
    def to_proj(self, proj):
        if not isinstance(proj, Proj):
            raise TypeError("isinstance of 'Proj' required")
        self._to_proj = proj

    def __call__(self, coords, reverse=False):
        """Applies the coordinate transformation.

        Parameters
        ----------
        coords : array_like(Number, shape=(n, k))
            Represents `n` points of `k` dimensions to transform.
        reverse : optional, bool
            Indicates whether or not to apply the inverse transformation.

        Returns
        -------
        np.ndarray(Number, shape=(n, k))
            Transformed coordinates.

        """
        coords = assertion.ensure_numarray(coords)
        if not isinstance(reverse, bool):
            raise TypeError("'reverse' needs to be boolean")

        if reverse:
            from_proj = self._to_proj
            to_proj = self._from_proj
        else:
            from_proj = self._from_proj
            to_proj = self._to_proj

        # get x and y coordinates
        if len(coords.shape) == 1:
            x, y = coords[0:2]
        elif len(coords.shape) == 2:
            coords = assertion.ensure_coords(coords)
            x = coords[:, 0]
            y = coords[:, 1]
        else:
            raise ValueError('malformed coordinate dimensions')

        # coordinate projection
        t_xy = np.array(pyproj.transform(
            from_proj.pyproj,
            to_proj.pyproj,
            x,
            y
        )).T

        # set new coordinates
        tCoords = np.copy(coords)
        if len(coords.shape) == 1:
            tCoords[0:2] = t_xy
        else:
            tCoords[:, 0:2] = t_xy

        return tCoords
