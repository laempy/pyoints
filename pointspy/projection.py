import numpy as np
from pyproj import Proj as pyProj
from pyproj import transform as CoordinateTransformation
from osgeo import osr


class Proj():

    def __init__(self, proj4=None):
        if proj4 is None or proj4 is '':
            proj4 = '+proj=latlong +datum=WGS84 +to +proj=latlong +datum=WGS84 +units=m +no_defs'
        self._proj4 = proj4

    @property
    def proj4(self):
        return self._proj4

    @property
    def wkt(self):
        sr = osr.SpatialReference()
        sr.ImportFromProj4(self.proj4)
        return sr.ExportToWkt()

    @property
    def pyProj(self):
        return pyProj(self.proj4)

    def __str__(self):
        return 'proj4: %s' % str(self.proj4)


def projFromProj4(proj4):
    return Proj(proj4)


def projFromWtk(wkt):
    proj4 = osr.SpatialReference(wkt=wkt).ExportToProj4()
    return Proj(proj4)


def projFromEPSG(epsg):
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    proj4 = sr.ExportToProj4()
    return Proj(proj4)


class CoordinateTransform:

    def __init__(self, fromProj, toProj):
        self._fromProj = fromProj
        self._toProj = toProj

    def __call__(self, coords):
        return np.vstack(CoordinateTransformation(
            self._fromProj.pyProj, self._toProj.pyProj, coords[:, 0], coords[:, 1])).T

    def transformExtent(self, ext):
        dim = len(ext) / 2
        coords = np.array([ext[0:2], ext[dim:(dim + 2)]])
        self(coords).flatten()
