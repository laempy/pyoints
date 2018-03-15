import numpy as np
from pyproj import Proj as pyProj
from pyproj import transform as CoordinateTransformation
from osgeo import osr

from . import assertion


# Global proj4 definitions
WGS84 = '+proj=latlong +datum=WGS84 +to +proj=latlong +datum=WGS84 +units=m +no_defs'
# TODO add some default projections

# TODO assertions


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
    GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]
    >>> print proj.proj4
    +proj=longlat +datum=WGS84 +no_defs

    Create from Proj4 string.

    >>> proj = Proj.from_proj4('+proj=longlat +datum=WGS84 +no_defs')
    >>> print proj.proj4
    +proj=longlat +datum=WGS84 +no_defs
    >>> print proj.wkt
    GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]

    Create from Well Known Text.

    >>> proj = Proj.from_wkt(proj.wkt)
    >>> print proj.proj4
    +proj=longlat +datum=WGS84 +no_defs

    """

    def __init__(self, proj4=WGS84):
        assert proj4 is not None
        assert isinstance(proj4, str)
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
        return pyProj(self.proj4)

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
        assert isinstance(wkt, str)
        proj4 = osr.SpatialReference(wkt=wkt).ExportToProj4()
        assert proj4 is not '', 'WKT unknown'
        return Proj.from_proj4(proj4)

    @classmethod
    def from_epsg(cls, epsg):
        """`Proj` object from EPSG code.

        Parameters
        ----------
        epsg : int
            Coordinate projection definition in EPSG format.

        """
        assert isinstance(epsg, int)
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        proj4 = sr.ExportToProj4()
        assert proj4 is not '', 'epsg code "%i" unknown' % epsg
        return Proj.from_proj4(proj4)


class CoordinateTransform:
    """Provides a coordinate transformation.

    Parameters
    ----------
    fromProj, toProj : `Proj`
        Define the coordinate transformation form the origin projection system
        `fromProj` to the target projection system `toProj`.

    Examples
    --------

    Transform coordinates.

    >>> wgs84 = Proj.from_epsg(4326)
    >>> gk2 = Proj.from_epsg(31466)
    >>> coords = [(6.842,49.971),(6.847,49.969),(6.902,49.991),(6.922,50.101)]
    >>> coordTransfrom = CoordinateTransform(wgs84,gk2)
    >>> tCoords = coordTransfrom(coords)
    >>> print np.round( tCoords,3)
    [[2560446.801 5537522.386]
     [2560808.009 5537303.984]
     [2564724.211 5539797.116]
     [2566007.32  5552049.646]]

    Reverse transformation.

    >>> print np.round( coordTransfrom(tCoords,reverse=True) ,3)
    [[ 6.842 49.971]
     [ 6.847 49.969]
     [ 6.902 49.991]
     [ 6.922 50.101]]

    """

    def __init__(self, fromProj, toProj):
        assert isinstance(fromProj, Proj)
        assert isinstance(toProj, Proj)
        self._fromProj = fromProj
        self._toProj = toProj

    def __call__(self, coords, reverse=False):
        coords = assertion.ensure_coords(coords)
        assert isinstance(reverse, bool)

        if reverse:
            fromProj = self._toProj
            toProj = self._fromProj
        else:
            fromProj = self._fromProj
            toProj = self._toProj

        tCoords = CoordinateTransformation(
            fromProj.pyproj,
            toProj.pyproj,
            coords[:, 0],
            coords[:, 1]
        )
        return assertion.ensure_coords(tCoords, by_row=True)
