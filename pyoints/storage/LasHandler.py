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
"""Handling of .las-files.
"""

import os
import sys
import numpy as np

import laspy
try:
    # use liblas to provide full spatial reference support (workaround)
    import liblas
except ModuleNotFoundError:
    pass

from ..extent import Extent
from ..georecords import (
    GeoRecords,
    LasRecords,
)
from .. import (
    transformation,
    projection,
    assertion,
    nptools,
)

from .BaseGeoHandler import GeoFile
from .dtype_converters import numpy_to_laspy_dtype


SUPPORTED_FORMATS = [0, 1, 2, 3, 4, 5]


class LasReader(GeoFile):
    """Class to read .las-files.

    Parameters
    ----------
    infile : String
        Las file to be read.
    proj : optional, Proj
        Spatial reference system. Usually provided only, if the spatial
        reference of the file has not be set yet.

    See Also
    --------
    GeoFile

    """

    def __init__(self, infile, proj=None):
        GeoFile.__init__(self, infile)

        lasFile = laspy.file.File(self.file, mode='r')

        # try to read projection from file
        if proj is None:
            if 'liblas' in sys.modules:
                reader = liblas.file.File(self.file, mode='r')
                wkt = reader.header.srs.get_wkt().decode('utf-8')
                if not wkt == '':
                    proj = projection.Proj.from_wkt(wkt)
                reader.close()
            else:
                for vlr in lasFile.header.vlrs:
                    if vlr.record_id == 2112:
                        # Spatial reference in well known text format
                        wkt = str(vlr.VLR_body.decode('utf-8'))
                        proj = projection.Proj.from_wkt(wkt)
                    elif vlr.record_id == 34735:
                        # GeoTIFF GeoKeyDirectoryTag
                        # not supported yet
                        pass
                    elif vlr.record_id == 34736:
                        # GeoTIFF GeoDoubleParamsTag
                        # not supported yet
                        pass
                    elif vlr.record_id == 34737:
                        # GeoTIFF GeoAsciiParamsTag
                        # not supported yet
                        pass

        self.proj = proj
        self.date = lasFile.header.date

        self.t = transformation.t_matrix(lasFile.header.offset)
        self._extent = Extent((lasFile.header.min, lasFile.header.max))
        self._count = int(lasFile.header.point_records_count)

        lasFile.close()
        del lasFile

    def __len__(self):
        return self._count

    @property
    def extent(self):
        return self._extent

    @property
    def corners(self):
        return Extent(self.extent[[0, 1, 3, 4]]).corners

    def load(self, extent=None):

        lasFile = laspy.file.File(self.file, mode='r')

        # check point formats
        if lasFile.header.data_format_id not in SUPPORTED_FORMATS:
            m = "Only point formats %s supported yet, got %"
            raise ValueError(
                m %
                (SUPPORTED_FORMATS, lasFile.header.data_format_id))

        date = lasFile.header.date
        scale = np.array(lasFile.header.scale, dtype=np.float64)
        offset = np.array(lasFile.header.offset, dtype=np.float64)
        las_fields = [
            str(dim.name.encode().decode()) for dim in lasFile.point_format
        ]  # ugly workaround to get actual strings

        points = lasFile.points['point'].copy().view(np.recarray)

        # Close File
        lasFile.close()
        del lasFile

        # filter by extent (before scaling)
        if extent is not None:
            ext = Extent(extent)
            iext = Extent([
                (ext.min_corner - offset) / scale,
                (ext.max_corner - offset) / scale
            ])
            sids = iext.intersection(
                    np.vstack([points.X, points.Y, points.Z]).T)
            points = points[sids]

        # much faster than accessing lasFile.x
        coords = np.empty((len(points), 3), dtype=np.float64)
        coords[:, 0] = points.X * scale[0] + offset[0]
        coords[:, 1] = points.Y * scale[1] + offset[1]
        coords[:, 2] = points.Z * scale[2] + offset[2]

        # grep data
        omit = ['X', 'Y', 'Z']
        dtypes = []
        dataDict = {'coords': coords}
        for name in las_fields:
            if name == 'flag_byte':
                values = points.flag_byte
                if np.any(values):
                    dataDict['num_returns'] = values // 8  # last three bits
                    dataDict['return_num'] = values % 8  # first three bits
            elif name == 'raw_classification':
                values = points.raw_classification
                if np.any(values):
                    dataDict['classification'] = values % 32  # bits 0 to 4
                    values = values // 32
                if np.any(values):
                    dataDict['synthetic'] = values % 2  # bit 5
                    values = values // 2
                if np.any(values):
                    dataDict['keypoint'] = values % 2  # bit 6
                    values = values // 2
                if np.any(values):
                    dataDict['withheld'] = values  # bit 7

            elif name not in omit:
                values = points[name]
                if np.any(values):
                    dataDict[name] = values

        # collect dtypes
        available_dtypes = LasRecords.available_fields()
        for name in dataDict.keys():
            for descr in available_dtypes:
                if descr[0] == name:
                    dtypes.append(descr)

        # create recarray
        data = nptools.recarray(dataDict, dtype=dtypes)

        if len(points) == 0:
            t = np.eye(4)
        else:
            t = transformation.t_matrix(offset)
        return LasRecords(self.proj, data, T=t, date=date)


def writeLas(geoRecords, outfile, point_format=3):
    """ Write a LAS file to disc.

    Parameters
    ----------
    geoRecords : GeoRecords
        Points to store to disk.
    outfile : String
        Desired output file.
    point_format : optional, positive int
        Desired LAS point format. See LAS specification for details.

    """
    # validate input
    if not isinstance(geoRecords, GeoRecords):
        raise TypeError("'geoRecords' needs to be of type 'GeoRecords'")
    if not os.access(os.path.dirname(outfile), os.W_OK):
        raise IOError('File %s is not writable' % outfile)

    if point_format not in SUPPORTED_FORMATS:
        raise ValueError("'point_format' %s not supported" % str(point_format))

    records = geoRecords.records()

    # Create file header
    header = laspy.header.Header(file_version=1.3, point_format=point_format)
    header.file_sig = 'LASF'

    # Open file in write mode
    lasFile = laspy.file.File(outfile, mode='w', header=header)

    # create VLR records
    vlrs = []
    if 'liblas' in sys.modules:
        # use liblas to create spatial reference
        srs = liblas.srs.SRS()
        srs.set_wkt(str.encode(geoRecords.proj.wkt))
        for i in range(srs.vlr_count()):
            vlr = laspy.header.VLR(
                user_id="LASF_Projection",
                record_id=srs.GetVLR(i).recordid,
                VLR_body=srs.GetVLR(i).data,
                description="OGC Coordinate System GeoTIFF"
            )
            vlr.parse_data()
            vlrs.append(vlr)
    else:
        # create wkt record only
        vlr = laspy.header.VLR(
            user_id="LASF_Projection",
            record_id=2112,
            VLR_body=str.encode(geoRecords.proj.wkt),
            description="OGC Coordinate System WKT"
        )
        vlrs.append(vlr)

    # set VLRs
    lasFile.header.set_vlrs(vlrs)

    if point_format > 5:
        lasFile.header.wkt = 1

    dim = min(geoRecords.dim, 3)

    # find optimal offset and scale scale to achieve highest precision
    offset = np.zeros(3)
    scale = np.ones(3)

    offset[:dim] = geoRecords.t.origin

    max_values = np.abs(records.extent().corners - offset[:dim]).max(0)
    max_digits = 2**28  # long int
    scale[:dim] = max_values / max_digits
    scale[np.isclose(scale, 0)] = 1 / max_digits

    lasFile.date = geoRecords.date
    lasFile.header.scale = scale.copy()
    lasFile.header.offset = offset.copy()

    # get default fields
    las_fields = [field.name for field in lasFile.point_format]
    field_names = records.dtype.names

    # Fields to omit
    omit = []
    omit.extend(las_fields)
    omit.extend(np.dtype(LasRecords.CUSTOM_FIELDS).names)

    # create user defined fields
    for name in field_names:
        if name not in omit:
            dtype = records.dtype[name]
            type_id = numpy_to_laspy_dtype(dtype)
            if type_id is None:
                omit.append(name)
            else:
                lasFile.define_new_dimension(name, type_id, '')

    # initialize
    flag_byte = np.zeros(len(records), dtype=np.uint)
    raw_classification = np.zeros(len(records), dtype=np.uint8)

    # set fields
    for name in field_names:
        if name == 'coords':
            lasFile.set_x_scaled(records.coords[:, 0])
            lasFile.set_y_scaled(records.coords[:, 1])
            if records.dim > 2:
                lasFile.set_z_scaled(records.coords[:, 2])
        elif name == 'classification':  # classification bits 0 to 4
            raw_classification += records.classification
        elif name == 'synthetic':  # classification bit 5
            raw_classification += records.synthetic.astype(np.uint8) * 32
        elif name == 'keypoint':  # classification bit 6
            raw_classification += records.keypoint.astype(np.uint8) * 64
        elif name == 'withheld':  # classification bit 7
            raw_classification += records.withheld.astype(np.uint8) * 128
        elif name == 'return_num':
            flag_byte = flag_byte + records.return_num
        elif name == 'num_returns':
            flag_byte = flag_byte + records.num_returns * 8
        elif name == 'intensity':
            lasFile.set_intensity(records.intensity)
        elif name == 'user_data':
            lasFile.set_user_data(records.user_data)
        elif name == 'red':
            lasFile.set_red(records.red)
        elif name == 'green':
            lasFile.set_green(records.green)
        elif name == 'blue':
            lasFile.set_blue(records.blue)
        elif name == 'nir':
            lasFile.set_blue(records.nir)
        elif name == 'pt_src_id':
            lasFile.set_pt_src_id(records.pt_src_id)
        elif name == 'gps_time':
            lasFile.set_gps_time(records.gps_time)
        elif name not in omit:
            lasFile._writer.set_dimension(name, records[name])

    if point_format > 5:
        lasFile.set_classification(raw_classification)
    else:
        lasFile.set_raw_classification(raw_classification)

    # close file
    lasFile.header.update_min_max()
    lasFile.close()
    del lasFile


def _updateLasHeader(las_file, offset=None, translate=None, precision=None):
    # experimental
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
        lasFile.header.offset = lasFile.header.offset + translate

    lasFile.header.update_min_max()
    lasFile.close()
    del lasFile


def _createTypeTestLas(outfile):
    # experimental

    # Create file header
    header = laspy.header.Header()
    header.file_sig = 'LASF'
    header.format = 1.2
    header.data_format_id = 3

    # Open file in write mode
    lasFile = laspy.file.File(outfile, mode='w', header=header)

    lasFile.header.scale = [1, 1, 1]
    lasFile.header.offset = [0, 0, 0]

    names = []
    for type_id in range(1, 31):
        name = 'field_%i' % type_id
        if type_id not in [8, 18, 28]:
            lasFile.define_new_dimension(name, type_id, '')
            names.append(name)

    k = 10
    lasFile.x = np.random.rand(k)
    lasFile.y = np.random.rand(k)
    lasFile.z = np.random.rand(k)

    lasFile.header.update_min_max()
    lasFile.close()
    del lasFile
