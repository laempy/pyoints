import numpy as np
from pyproj import Proj as pyProj
from pyproj import transform as CoordinateTransformation
from osgeo import osr


# Global proj4 definitions
WGS84 = '+proj=latlong +datum=WGS84 +to +proj=latlong +datum=WGS84 +units=m +no_defs'
#TODO add some default projections

#TODO assertions

class Proj():
    """Wrapper class for different projection definitions.
    
    Paramerters
    -----------
    proj4 : optional, str
        Coordinate projection definition in Proj4 format. WGS84, if None or 
        empty string.
        
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
    
    TODO
    
    """

    def __init__(self, proj4=None):
        if proj4 is None or proj4 is '':
            proj4 = WGS84
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
    def pyproj(self):
        return pyProj(self.proj4)

    def __str__(self):
        return 'proj4: %s' % str(self.proj4)


def proj_from_proj4(proj4):
    """Creates a `Proj` object.
    
    Parameters
    ----------
    proj4 : str
        Coordinate projection definition in Proj4 format. 
        
    Returns
    -------
    `Proj`
    
    Examples
    --------
    
    TODO
    
    """
    return Proj(proj4)


def proj_from_wtk(wkt):
    """Creates a `Proj` object.
    
    Parameters
    ----------
    wkt : str
        Coordinate projection definition in Well Known Text format. 
        
    Returns
    -------
    `Proj`
    
    Examples
    --------
    
    TODO    
    
    """
    proj4 = osr.SpatialReference(wkt=wkt).ExportToProj4()
    return Proj(proj4)


def proj_from_epsg(epsg):
    """Creates a `Proj` object.
    
    Parameters
    ----------
    epsg : int
        Coordinate projection definition in EPSG format. 
        
    Returns
    -------
    `Proj`
    
    Examples
    --------
    
    TODO
    
    """
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    proj4 = sr.ExportToProj4()
    return Proj(proj4)



class CoordinateTransform:
    """Provides a coordinate transformation.
    
    Parameters
    ----------
    fromProj, toProj : `Proj`
        Define the coordinate transformation form the origin projection system 
        `fromProj` to the target projection system `toProj`.
        
    Examples
    --------
    
    TODO
    
    """
    def __init__(self, fromProj, toProj):
        self._fromProj = fromProj
        self._toProj = toProj

    def __call__(self, coords):
        tCoords = CoordinateTransformation(
            self._fromProj.pyproj,
            self._toProj.pyproj, 
            coords[:, 0],
            coords[:, 1]
        )
        return np.vstack(tCoords).T
