import numpy as np

import liblas
import laspy

from .. georecords import GeoRecords
from .. extent import Extent
from .. import (
    projection,
    assertion,
)
from .BaseGeoHandler import GeoFile


class LasReader(GeoFile):

    def __init__(self, file, proj=None):
        GeoFile.__init__(self, file)

        lasFile = laspy.file.File(self.file, mode='r')

        self._proj = proj
        if proj is None:
            # headerReader=liblas.file.File(file,mode='r')
            # self._proj=projection.projFromProj4(headerReader.header.srs.get_proj4())
            for vlr in lasFile.header.vlrs:
                if vlr.record_id == 2112:
                    wtk = vlr.VLR_body
                    self._proj = projection.Proj.from_wkt(wtk)
                    break

        if self._proj is None:
            raise Exception('No projection found')

        self._extent = Extent((lasFile.header.min, lasFile.header.max))
        self._count = int(lasFile.header.point_records_count)

        lasFile.close()
        del lasFile

    def __len__(self):
        return self._count

    @property
    def proj(self):
        return self._proj

    @property
    def extent(self):
        return self._extent

    @property
    def corners(self):
        return Extent(self.extent[[0, 1, 3, 4]]).corners()

    def load(self, extent=None):

        lasFile = laspy.file.File(self.file, mode='r')

        scale = np.array(lasFile.header.scale, dtype=np.float64)
        offset = np.array(lasFile.header.offset, dtype=np.float64)
        lasFields = [dim.name for dim in lasFile.point_format]

        points = lasFile.points.view(np.recarray).point

        # filter by extent (before scaling)
        if extent is not None:
            ext = Extent(extent)
            iext = Extent(np.array([
                        (ext.min_corner - offset) / scale,
                        (ext.max_corner - offset) / scale
                     ], dtype='i4'))
            sIds = iext.intersects([points.X, points.Y, points.Z])
            points = points[sIds]

        # much faster than accessing lasFile.x
        coords = np.empty((len(points), 3), dtype=np.float64)
        coords[:, 0] = points.X * scale[0] + offset[0]
        coords[:, 1] = points.Y * scale[1] + offset[1]
        coords[:, 2] = points.Z * scale[2] + offset[2]

        # grep data
        dataDict = {'coords': coords}
        if 'intensity' in lasFields:
            values = points.intensity
            if np.any(values):
                dataDict['intensity'] = values  # .copy()
        if 'raw_classification' in lasFields:
            values = points.raw_classification
            if np.any(values):
                dataDict['classification'] = values  # .copy()
        if 'user_data' in lasFields:
            values = points.user_data
            if np.any(values):
                dataDict['user_data'] = values  # .copy()
        if 'gps_time' in lasFields:
            values = points.gps_time
            if np.any(values):
                dataDict['gps_time'] = values  # .copy()
        if 'flag_byte' in lasFields:
            values = points.flag_byte
            if np.any(values):
                flag_byte = values[sIds]  # .copy()
                dataDict['num_returns'] = flag_byte / 8
                dataDict['return_num'] = flag_byte % 8
        if 'pt_src_id' in lasFields:
            values = points.pt_src_id
            if np.any(values):
                dataDict['pt_src_id'] = values  # .copy()
        if 'red' in lasFields and 'green' in lasFields and 'blue' in lasFields:
            red = points.red
            green = points.green
            blue = points.blue
            if np.any(red) or np.any(green) or np.any(blue):
                values = np.vstack([red, green, blue]).T  # .copy()
                dataDict['rgb'] = values

        dtypes = []
        for key in dataDict:
            dtypes.append(LasRecords.FIELDS[key])

        # Create recarray
        data = np.recarray((len(points),), dtype=dtypes)
        for key in dataDict:
            data[key] = dataDict[key]

        if len(points) == 0:
            return data.view(LasRecords)

        # Close File
        lasFile.close()
        del lasFile

        return LasRecords(self.proj, data)

    def cleanCache(self):
        pass


def writeLas(geoRecords, las_file, precision=[5, 5, 5]):

    # Create File
    header = laspy.header.Header()
    header.file_sig = 'LASF'
    #header.format = 1.4
    header.data_format_id = 3
    lasFile = laspy.file.File(las_file, mode='w', header=header)

    # Set projection
    # TODO ohne liblas ==>
    # https://github.com/laspy/laspy/blob/master/laspy/header.py
    headerReader = liblas.file.File(las_file, mode='r')
    liblasHeader = headerReader.header
    headerReader.close()
    del headerReader

    srs = liblas.srs.SRS()
    srs.set_proj4(geoRecords.proj.proj4)
    liblasHeader.srs = srs

    headerWriter = liblas.file.File(las_file, mode='w', header=liblasHeader)
    headerWriter.close()
    del headerWriter

    # Set values
    offset = geoRecords.extent().min_corner

    # TODO scale ueberarbeiten
    scale = np.repeat(10.0, 3)**-np.array(precision)
    lasFile.header.scale = scale
    lasFile.header.offset = offset

    lasFile.x = geoRecords.coords[:, 0]
    lasFile.y = geoRecords.coords[:, 1]
    lasFile.z = geoRecords.coords[:, 2]

    # Add attributes
    fields = geoRecords.dtype.names
    if 'intensity' in fields:
        lasFile.intensity = geoRecords.intensity
    if 'classification' in fields:
        lasFile.raw_classification = geoRecords.classification
    if 'user_data' in fields:
        lasFile.user_data = geoRecords.user_data
    if 'return_num' in fields and 'num_returns' in fields:
        lasFile.set_flag_byte(
            geoRecords.return_num +
            geoRecords.num_returns *
            8)
    if 'gps_time' in fields:
        lasFile.gps_time = geoRecords.gps_time
    if 'pt_src_id' in fields:
        lasFile.pt_src_id = geoRecords.pt_src_id
    if 'rgb' in fields:
        lasFile.red = geoRecords.rgb[:, 0]
        lasFile.green = geoRecords.rgb[:, 1]
        lasFile.blue = geoRecords.rgb[:, 2]

    # Close file
    lasFile.header.update_min_max()
    lasFile.close()
    del lasFile

    return LasReader(las_file)


class LasRecords(GeoRecords):

    FIELDS = {
        'coords': ('coords', float, 3),
        'intensity': ('intensity', int),
        'classification': ('classification', int),
        'user_data': ('user_data', np.uint8),
        'gps_time': ('gps_time', float),
        'num_returns': ('num_returns', np.uint8),
        'return_num': ('return_num', np.uint8),
        'pt_src_id': ('pt_src_id', float),
        'rgb': ('rgb', np.uint8, 3),
    }

    def activate(self, field):
        return self.add_field(self.FIELDS[field])

    def grd(self):
        return self.classes(2, 11)

    def veg(self):
        return self.classes(3, 4, 5, 20)

    def classes(self, *classes):
        mask = np.in1d(self.classification, classes)
        return self[mask]

    @property
    def last_return(self):
        return self.return_num == self.num_returns

    @property
    def first_return(self):
        return self.return_num == 1

    @property
    def only_return(self):
        return self.num_returns == 1


def updateLasHeader(las_file, offset=None, translate=None, precision=None):
    lasFile = laspy.file.File(las_file, mode='rw')

    if precision is not None:
        precision = assertion.ensure_numvector(precision)
        if not len(precision) == 3:
            raise ValueError('"precision" has to have a length of 3')
        scale = np.repeat(10.0, 3)**-np.array(precision)
        lasFile.header.scale = scale

    if offset is not None:
        offset = assertion.ensure_numvector(offset)
        if not len(offset) == 3:
            raise ValueError('"offset" has to have a length of 3')
        lasFile.header.offset = offset

    if translate is not None:
        translate = assertion.ensure_numvector(translate)
        if not len(translate) == 3:
            raise ValueError('"translate" has to have a length of 3')
        lasFile.header.offset += translate

    #lasFile.header.update_min_max()
    lasFile.close()
    del lasFile


