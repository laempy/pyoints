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
import os
import numpy as np
import datetime
from osgeo import (
    gdal,
    osr,
)
from .BaseGeoHandler import GeoFile
from .dtype_converters import numpy_to_gdal_dtype
from ..extent import Extent
from .. import (
    assertion,
    grid,
    nptools,
    projection,
    transformation,
)



class RasterReader(GeoFile):
    """Reads image files.

    Parameters
    ----------
    infile : String
        Raster file to be read.
    proj : optional, Proj
        Spatial reference system. Usually just provided, if the spatial
        reference has not be set yet.
    date : datetime.date
        Date of capture.

    See Also
    --------
    GeoFile

    """

    def __init__(self, infile, proj=None, date=None):
        GeoFile.__init__(self, infile)

        # Read header
        gdalRaster = gdal.Open(self.file, gdal.GA_ReadOnly)

        if proj is None:
            wkt = gdalRaster.GetProjection()
            if wkt is not '':
                self.proj = projection.Proj.from_proj4(
                    osr.SpatialReference(wkt=wkt).ExportToProj4())
            else:
                raise ValueError("no projection found")
        else:
            self.proj = proj

        self.t = transformation.matrix_from_gdal(gdalRaster.GetGeoTransform())

        self._shape = (gdalRaster.RasterYSize, gdalRaster.RasterXSize)
        self._num_bands = gdalRaster.RasterCount

        self._corners = grid.transform_to_corners(self.t, self._shape)
        self._extent = Extent(self._corners)

        # try to read date
        if date is None:
            date = gdalRaster.GetMetadataItem('ACQUISITIONDATETIME')
            if date is None:
                date = gdalRaster.GetMetadataItem('TIFFTAG_DATETIME')
            if date is not None:
                year, month, day = date.split(' ')[0].split(':')
                self.date = datetime.date(int(year), int(month), int(day))
        else:
            self.date = date

        del gdalRaster

    @property
    def num_bands(self):
        return self._num_bands

    @property
    def corners(self):
        return self._corners

    @property
    def extent(self):
        return self._extent

    def load(self, extent=None):

        gdalRaster = gdal.Open(self.file, gdal.GA_ReadOnly)

        T = self.t
        shape = (gdalRaster.RasterYSize, gdalRaster.RasterXSize)
        corner = (0, 0)

        if extent is not None:
            T, corner, shape = grid.extentinfo(T, extent)

        attr = np.recarray(
            shape, dtype=[
                ('bands', int, gdalRaster.RasterCount)])
        attr.bands[:] = np.swapaxes(
            gdalRaster.ReadAsArray(corner[1], corner[0], shape[1], shape[0]).T,
            0,
            1
        )
        raster = grid.Grid(self.proj, attr, T)

        del gdalRaster

        return raster


def writeTif(image, outfile, T=None, proj=None, no_data=np.nan):
    """Writes an image to disc.

    Parameters
    ----------
    image : np.ndarray(Number, shape=(rows, cols, k))
        Image to save
    outfile : String
        File to save the raster to.
    no_data : optional, object
        No data value to be used.
    T : optional, array_like(Number, shape=(3, 3))
        Projection matrix to be used.
    proj : Proj
        Projection to be used.
    no_data : optional, object
        Desired no data value.

    Raises
    ------
    IOError
    
    See Also
    --------
    writeRaster

    """
    # validate input
    if not os.access(os.path.dirname(outfile), os.W_OK):
        raise IOError('File %s is not writable' % outfile)
    if not isinstance(image, np.ndarray):
        m = "'image' needs to be an instance of 'np.ndarray', got %s" 
        raise TypeError(m % type(image))     
    if not len(image.shape) in (2, 3):
        raise ValueError("'image' has an unexpected shape for a raster" )
    if not nptools.isnumeric(image):
        raise ValueError("'image' needs to be numeric")
        
    bands = image.astype(nptools.minimum_numeric_dtype(image))
    
    num_bands = 1 if len(bands.shape) == 2 else bands.shape[2]
        
    driver = gdal.GetDriverByName('GTiff')
    gdalRaster = driver.Create(
        outfile,
        bands.shape[1],
        bands.shape[0],
        num_bands,
        numpy_to_gdal_dtype(bands.dtype)
    )
    
    # SetProjection
    if proj is not None:
        if not isinstance(proj, projection.Proj):
            raise ValueError("'proj' needs to be an instance of Proj")
        gdalRaster.SetProjection(proj.wkt)
    
    # SetGeoTransform
    if T is not None:
        T = assertion.ensure_tmatrix(T, dim=2)
        t = transformation.matrix_to_gdal(T)      
        gdalRaster.SetGeoTransform(t)
    
    # set bands
    if num_bands == 1:
        band = gdalRaster.GetRasterBand(1)
        band.SetNoDataValue(no_data)
        band.WriteArray(bands)
        band.FlushCache()
    else:
        for i in range(num_bands):
            band = gdalRaster.GetRasterBand(i + 1)
            band.SetNoDataValue(no_data)
            band.WriteArray(bands[:, :, i])
            band.FlushCache()
            band = None
            del band

    gdalRaster.FlushCache()
    gdalRaster = None
    del gdalRaster
    

def writeRaster(raster, outfile, field='bands', no_data=np.nan):
    """Writes a a Grid to disc.

    Parameters
    ----------
    raster : Grid(shape=(cols, rows))
        A two dimensional Grid to be stored with of `cols` columns and `rows`
        rows.
    outfile : String
        File to save the raster to.
    field : optional, str
        Field considered as raster bands.
    no_data : optional, object
        Desired no data value.

    Raises
    ------
    IOError
    
    See Also
    --------
    writeTif

    """
    if not isinstance(raster, grid.Grid):
        m = "'geoRecords' needs to be of type 'Grid', got %s" % type(raster)
        raise TypeError(m)
    if not raster.dim == 2:
        raise ValueError("'geoRecords' needs to be two dimensional")
    
    if not isinstance(field, str):
        raise TypeError("'field' needs to be a string")
    if not hasattr(raster, field):
        raise ValueError("'geoRecords' needs to have a field '%s'" % field)
    image = raster[field]
    
    writeTif(image, outfile, T=raster.t, proj=raster.proj, no_data=no_data)