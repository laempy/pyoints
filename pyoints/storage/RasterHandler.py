# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
# END OF LICENSE NOTE
import os
import numpy as np
import datetime
from affine import Affine

from osgeo import (
    gdal,
    osr,
)
from .BaseGeoHandler import GeoFile
from .. import (
    grid,
    nptools,
    projection,
    transformation,
    Extent,
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

        self.t = transformation.LocalSystem(np.matrix(
            Affine.from_gdal(*gdalRaster.GetGeoTransform())
        ).reshape(3, 3))

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


def writeRaster(raster, outfile, field='bands', noData=np.nan):
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

    Raises
    ------
    IOError

    """
    # validate input
    if not os.access(os.path.dirname(outfile), os.W_OK):
        raise IOError('File %s is not writable' % outfile)
    if not isinstance(raster, grid.Grid):
        raise TypeError("'geoRecords' needs to be of type 'Grid'")
    if not raster.dim == 2:
        raise ValueError("'geoRecords' needs to be two dimensional")
    if not isinstance(field, str):
        raise TypeError("'field' needs to be a string")
    if not hasattr(raster, field):
        raise ValueError("'geoRecords' needs to have a field '%s'" % field)
    bands = raster[field]
    if not nptools.isnumeric(bands):
        raise ValueError("'geoRecords[%s]' needs to be numeric" % field)

    if len(bands.shape) == 2:
        num_bands = 1
    else:
        num_bands = bands.shape[2]

    driver = gdal.GetDriverByName('GTiff')
    gdalRaster = driver.Create(
        outfile,
        raster.shape[1],
        raster.shape[0],
        num_bands,
        gdal.GDT_Float32
    )

    if num_bands == 1:
        band = gdalRaster.GetRasterBand(1)
        band.SetNoDataValue(noData)
        band.WriteArray(bands)
        band.FlushCache()
    else:
        for i in range(num_bands):
            band = gdalRaster.GetRasterBand(i + 1)
            band.SetNoDataValue(noData)
            band.WriteArray(bands[:, :, i])
            band.FlushCache()
            band = None
            del band

    gdalRaster.SetGeoTransform(transformation.matrix_to_gdal(raster.t))
    gdalRaster.SetProjection(raster.proj.wkt)

    gdalRaster.FlushCache()
    gdalRaster = None
    del gdalRaster
