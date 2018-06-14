import os
import numpy as np
import laspy

from .. georecords import GeoRecords
from .. extent import Extent
from .. import (
    transformation,
    projection,
    assertion,
    nptools,
)
from .BaseGeoHandler import GeoFile


class LasReader(GeoFile):
    """Class to read .las-files.

    Parameters
    ----------
    infile : String
        Las file to be read.

    See Also
    --------
    GeoFile

    """
    def __init__(self, infile, proj=None):
        GeoFile.__init__(self, infile)

        lasFile = laspy.file.File(self.file, mode='r')

        if proj is None:
            for vlr in lasFile.header.vlrs:
                # read projection from WKT
                if vlr.record_id == 2112:
                    wkt = str(vlr.VLR_body.decode('utf-8'))
                    proj = projection.Proj.from_wkt(wkt)
                    break

        self.proj = proj

        self.t = transformation.t_matrix(lasFile.header.min)
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

        scale = np.array(lasFile.header.scale, dtype=np.float64)
        offset = np.array(lasFile.header.offset, dtype=np.float64)
        lasFields = [dim.name.encode('utf-8') for dim in lasFile.point_format]

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
        omit_fields = ('X', 'Y', 'Z')
        dtypes = []
        dataDict = {'coords': coords}
        for name in lasFields:
            if name == 'flag_byte':
                values = points.flag_byte
                if np.any(values):
                    dataDict['num_returns'] = values / 8
                    dataDict['return_num'] = values % 8
            elif name == 'raw_classification':
                values = points.raw_classification
                if np.any(values):
                    dataDict['classification'] = values
            elif name not in omit_fields:
                values = points[name]
                if np.any(values):
                    dataDict[name] = values

        # create recarray
        data = nptools.recarray(dataDict, dtype=dtypes)

        # Close File
        lasFile.close()
        del lasFile

        if len(points) == 0:
            return data.view(LasRecords)

        t = transformation.t_matrix(offset)
        return LasRecords(self.proj, data, T=t)


def writeLas(geoRecords, outfile):
    """ Write a LAS file to disc.

    Parameters
    ----------
    geoRecords : GeoRecords
        Points to store to disk.
    lasfile : String
        Desired output file.

    """
    # validate input
    if not isinstance(geoRecords, GeoRecords):
        raise ValueError('Type GeoRecords required')
    if not os.access(os.path.dirname(outfile), os.W_OK):
        raise IOError('File %s is not writable' % outfile)

    # Create file header
    header = laspy.header.Header()
    header.file_sig = 'LASF'
    header.format = 1.2
    header.data_format_id = 3

    # Open file in write mode
    lasFile = laspy.file.File(outfile, mode='w', header=header)

    # create projection VLR using WKT
    proj_vlr = laspy.header.VLR(
        user_id="LASF_Projection",
        record_id=2112,
        VLR_body=str.encode(geoRecords.proj.wkt),
        description="OGC Coordinate System WKT"
    )
    proj_vlr.parse_data()

    # set VLRs
    lasFile.header.set_vlrs([proj_vlr])

    # find optimal offset and scale scale to achieve highest precision
    offset = geoRecords.t.origin

    max_values = np.abs(geoRecords.extent().corners - offset).max(0)
    max_digits = 2**30  # long
    scale = max_values / max_digits

    lasFile.header.scale = scale
    lasFile.header.offset = offset

    # get default fields
    lasFields = [dim.name.encode('utf-8') for dim in lasFile.point_format]

    # create user defined fields
    for name in geoRecords.dtype.names:
        if name not in lasFields and name not in LasRecords.USER_DEFINED_FIELDS:
            dtype = geoRecords.dtype[name]
            # seee https://pythonhosted.org/laspy/tut_part_3.html
            if dtype.kind in np.typecodes['AllInteger']:
                type_id = 6  # long
            else:
                type_id = 10  # double
            lasFile.define_new_dimension(name, type_id, '')

    # set user defined fields
    for name in geoRecords.dtype.names:
        if name not in LasRecords.USER_DEFINED_FIELDS:
            lasFile._writer.set_dimension(name, geoRecords[name])

    # set coordinates
    lasFile.set_x_scaled(geoRecords.coords[:, 0])
    lasFile.set_y_scaled(geoRecords.coords[:, 1])
    lasFile.set_z_scaled(geoRecords.coords[:, 2])

    # set special fields
    fields = geoRecords.dtype.names
    if 'classification' in fields:
        print('set_c')
        lasFile.set_raw_classification(geoRecords.classification)
    if 'return_num' in fields and 'num_returns' in fields:
        flay_byte = geoRecords.return_num + geoRecords.num_returns * 8
        lasFile.set_flag_byte(flay_byte)

    # Close file
    lasFile.header.update_min_max()
    lasFile.close()
    del lasFile

    return LasReader(outfile)


class LasRecords(GeoRecords):
    """Data structure extending GeoRecords to provide an optimized API for LAS
    data.

    Properties
    ----------
    last_return : np.ndarray(bool)
        Array indicating if a point is a last return point.
    first_return : np.ndarray(bool)
        Array indicating if a point is a first return point.
    only_return : np.ndarray(bool)
        Array indicating if a point is the only returned point.

    See Also
    --------
    GeoRecords

    """
    USER_DEFINED_FIELDS = {
        'coords': ('coords', np.float, 3),
        'classification': ('classification', np.uint8),
        'num_returns': ('num_returns', np.uint8),
        'return_num': ('return_num', np.uint8),
    }

    @property
    def last_return(self):
        return self.return_num == self.num_returns

    @property
    def first_return(self):
        return self.return_num == 1

    @property
    def only_return(self):
        return self.num_returns == 1

    def activate(self, field):
        """Activates a desired field on demand.

        Parameters
        ----------
        field : String
            Name of the field to activate.

        """
        return self.add_fields([self.USER_DEFINED_FIELDS[field]])

    def grd(self):
        """Filter by points classified as ground.

        Returns
        -------
        LasRecords
            Filtered records.

        """
        return self.classes(2, 11)

    def veg(self):
        """Filter by points classified as vegetation.

        Returns
        -------
        LasRecords
            Filtered records.

        """
        return self.classes(3, 4, 5, 20)

    def classes(self, *classes):
        """Filter by classes.

        Parameters
        ----------
        *classes : int
            Classes to filter by.

        Returns
        -------
        LasRecords
            Filtered records.

        """
        mask = np.in1d(self.classification, classes)
        return self[mask]


# experimental
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
        lasFile.header.offset = lasFile.header.offset + translate

    #lasFile.header.update_min_max()
    lasFile.close()
    del lasFile


