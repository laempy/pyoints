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
    projection,
    Extent,
)


class RasterReader(GeoFile):

    def __init__(self, file, date=None, proj=None):

        GeoFile.__init__(self, file)

        # Read header
        gdalRaster = gdal.Open(self.file, gdal.GA_ReadOnly)

        self._proj = proj
        if proj is None:
            wkt = gdalRaster.GetProjection()
            if wkt is not '':
                self._proj = projection.Proj.from_proj4(
                    osr.SpatialReference(wkt=wkt).ExportToProj4())
        if self._proj is None:
            raise Exception('No projection found')

        self._T = np.matrix(
            Affine.from_gdal(
                *
                gdalRaster.GetGeoTransform())).reshape(
            3,
            3)
        self._shape = (gdalRaster.RasterYSize, gdalRaster.RasterXSize)
        self._numBands = gdalRaster.RasterCount

        self._corners = grid.transform2corners(self._T, self._shape)
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
    def numBands(self):
        return self._numBands

    @property
    def proj(self):
        return self._proj

    @property
    def corners(self):
        return self._corners

    @property
    def extent(self):
        return self._extent

    def load(self, extent=None):

        gdalRaster = gdal.Open(self.file, gdal.GA_ReadOnly)

        T = self._T
        shape = (gdalRaster.RasterYSize, gdalRaster.RasterXSize)
        corner = (0, 0)

        if extent is not None:
            T, corner, shape = grid.Grid.extentInfo(T, shape, extent)

        attr = np.recarray(
            shape, dtype=[
                ('bands', int, gdalRaster.RasterCount)])
        attr.bands[:] = np.swapaxes(
            gdalRaster.ReadAsArray(
                corner[1],
                corner[0],
                shape[1],
                shape[0]).T,
            0,
            1)
        raster = grid.Grid(self.proj, attr, T)

        del gdalRaster

        return raster

    def cleanCache(self):
        pass


def writeRaster(grid, filename, noData=np.nan):
    """Writes a a pointspy Grid to disc.

    Parameters
    ----------
    grid : Grid(shape=(cols, rows))
        A two dimensional pointspy Grid to be stored with of `cols`
        columns and `rows` rows.
    filename : String
        File to save the raster to.

    Examples
    --------
    TODO

    """
    # TODO test writable file

    driver = gdal.GetDriverByName('GTiff')

    if len(grid.bands.shape) == 2:
        numBands = 1
    else:
        numBands = grid.bands.shape[2]

    raster = driver.Create(
        filename,
        grid.shape[1],
        grid.shape[0],
        numBands,
        gdal.GDT_Float32
    )

    if numBands == 1:
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(noData)
        band.WriteArray(grid.bands)
        band.FlushCache()
    else:
        for i in range(numBands):
            band = raster.GetRasterBand(i + 1)
            band.SetNoDataValue(noData)
            band.WriteArray(grid.bands[:, :, i])
            band.FlushCache()

    raster.SetGeoTransform(grid.get_gdal_transform())
    raster.SetProjection(grid.proj.wkt)

    raster.FlushCache()
