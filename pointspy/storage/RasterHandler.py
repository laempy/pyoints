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
    transformation,
    Extent,
)


class RasterReader(GeoFile):
    """
    TODO: docstring
    """

    def __init__(self, filename, proj=None, date=None):
        GeoFile.__init__(self, filename)

        # Read header
        gdalRaster = gdal.Open(self.file, gdal.GA_ReadOnly)

        if proj is None:
            wkt = gdalRaster.GetProjection()
            if wkt is not '':
                self._proj = projection.Proj.from_proj4(
                    osr.SpatialReference(wkt=wkt).ExportToProj4())
        else:
            self.proj = proj

        self.t = np.matrix(
                Affine.from_gdal(*gdalRaster.GetGeoTransform())
            ).reshape(3, 3)
        self._shape = (gdalRaster.RasterYSize, gdalRaster.RasterXSize)
        self._num_bands = gdalRaster.RasterCount

        self._corners = grid.transform2corners(self.t, self._shape)
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
            T, corner, shape = grid.Grid.extentInfo(T, extent)

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
        num_bands = 1
    else:
        num_bands = grid.bands.shape[2]

    raster = driver.Create(
        filename,
        grid.shape[1],
        grid.shape[0],
        num_bands,
        gdal.GDT_Float32
    )

    if num_bands == 1:
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(noData)
        band.WriteArray(grid.bands)
        band.FlushCache()
    else:
        for i in range(num_bands):
            band = raster.GetRasterBand(i + 1)
            band.SetNoDataValue(noData)
            band.WriteArray(grid.bands[:, :, i])
            band.FlushCache()

    raster.SetGeoTransform(transformation.matrix_to_gdal(grid.t))
    raster.SetProjection(grid.proj.wkt)

    raster.FlushCache()
