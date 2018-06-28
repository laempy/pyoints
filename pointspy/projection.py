"""Coordinate Reference Systems and two dimensional geograpic coordinate
transformations.
"""

import pyproj
import numpy as np
from osgeo import osr

from . import assertion


# Global proj4 definitions
WGS84 = '+proj=latlong +datum=WGS84 +to +proj=latlong +datum=WGS84 +units=m ' \
    '+no_defs'


class Proj():
    """Wrapper class for different coordinate reference system definitions.

    Paramerters
    -----------
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
        Projection as `pyproj.Proj` object.

    Examples
    --------

    Create from EPSG code.

    >>> proj = Proj.from_epsg(4326)
    >>> print proj.wkt
    GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]
    >>> print proj.proj4
    +proj=longlat +datum=WGS84 +no_defs

    Create from Proj4 string.

    >>> proj = Proj.from_proj4('+proj=longlat +datum=WGS84 +no_defs')
    >>> print proj.proj4
    +proj=longlat +datum=WGS84 +no_defs
    >>> print proj.wkt
    GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]

    Create from Well Known Text.

    >>> proj = Proj.from_wkt(proj.wkt)
    >>> print proj.proj4
    +proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs

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
        """`Proj` object from Proj4 format.

        Parameters
        ----------
        proj4 : str
            Coordinate projection definition in Proj4 format.

        """
        return Proj(proj4)

    @classmethod
    def from_wkt(cls, wkt):
        """`Proj` object from Well Known Text.

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
        """`Proj` object from EPSG code.

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
    """Provides a coordinate transformation.

    Parameters
    ----------
    fromProj, toProj : `Proj`
        Define the coordinate transformation form the origin projection system
        `fromProj` to the target projection system `toProj`.

    Examples
    --------

    Transform coordinates.

    >>> import numpy as np
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
    >>> print np.round(tCoords, 3)
    [[2560446.801 5537522.386]
     [2560808.009 5537303.984]
     [2564724.211 5539797.116]
     [2566007.32  5552049.646]]

    Reverse transformation.

    >>> print np.round( geoTransfrom(tCoords, reverse=True) ,3)
    [[ 6.842 49.971]
     [ 6.847 49.969]
     [ 6.902 49.991]
     [ 6.922 50.101]]

    """

    def __init__(self, fromProj, toProj):
        if not isinstance(fromProj, Proj) or not isinstance(toProj, Proj):
            raise TypeError("objects of type 'Proj' required")
        self._fromProj = fromProj
        self._toProj = toProj

    def __call__(self, coords, reverse=False):
        coords = assertion.ensure_numarray(coords)
        if not isinstance(reverse, bool):
            raise TypeError("'reverse' needs to be boolean")

        if reverse:
            fromProj = self._toProj
            toProj = self._fromProj
        else:
            fromProj = self._fromProj
            toProj = self._toProj

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
            fromProj.pyproj,
            toProj.pyproj,
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
