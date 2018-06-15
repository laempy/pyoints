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

"""
# seee https://pythonhosted.org/laspy/tut_part_3.html
0 	Raw Extra Bytes 	Value of "options"
1 	unsigned char 	1 byte
2 	Char 	1 byte
3 	unsigned short 	2 bytes
4 	Short 	2 bytes
5 	unsigned long 	4 bytes
6 	Long 	4 bytes
7 	unsigned long long 	8 bytes
8 	long long 	8 bytes
9 	Float 	4 bytes
10 	Double 	8 bytes
11 	unsigned char[2] 	2 byte
12 	char[2] 	2 byte
13 	unsigned short[2] 	4 bytes
14 	short[2] 	4 bytes
15 	unsigned long[2] 	8 bytes
16 	long[2] 	8 bytes
17 	unsigned long long[2] 	16 bytes
18 	long long[2] 	16 bytes
19 	float[2] 	8 bytes
20 	double[2] 	16 bytes
21 	unsigned char[3] 	3 byte
22 	char[3] 	3 byte
23 	unsigned short[3] 	6 bytes
24 	short[3] 	6 bytes
25 	unsigned long[3] 	12 bytes
26 	long[3] 	12 bytes
27 	unsigned long long[3] 	24 bytes
28 	long long[3] 	24 bytes
29 	float[3] 	12 bytes
30 	double[3] 	24 bytes

]
"""

# Type conversion

LASPY_TYPE_MAP = [
    (1, ['|u1']),
    (2, ['|S1']),
    (3, ['<u2']),
    (4, ['<i2']),
    (5, ['<u4']),
    (6, ['<i4', '<i8']),
    (7, ['<u8']),
    (8, []),
    (9, ['<f4']),
    (10, ['<f8']),
    (11, ['|V2']),
    (12, ['|S2']),
]

LASPY_TO_NUMPY_TYPE = {}
for dim in range(1, 4):
    for key, t in LASPY_TYPE_MAP:
        if len(t)>0:
            type_id = key + (len(LASPY_TYPE_MAP)-1) * (dim-1)
            LASPY_TO_NUMPY_TYPE[type_id] = (t[0], dim)

NUMPY_TO_LASPY_TYPE = {}
for dim in range(1, 4):
    NUMPY_TO_LASPY_TYPE[dim] = {}
    for t, p in LASPY_TYPE_MAP:
        type_id = t + (len(LASPY_TYPE_MAP)-1) * (dim-1)
        for key in p:
            NUMPY_TO_LASPY_TYPE[dim][key] = type_id

def _dtype_to_laspy_type_id(dtype):
    if dtype.subdtype is None:
        dt = dtype
        type_name = dt.str
        type_dim = dt.shape[0] if len(dt.shape) > 0 else 1
    else:
        dt = dtype.subdtype
        type_name = dt[0].str
        type_dim = dt[1][0] if len(dt[1]) > 0 else 1
    return NUMPY_TO_LASPY_TYPE[type_dim][type_name]


def createTypeTestLas(outfile):

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
        las_fields = [
                str(dim.name.encode().decode()) for dim in lasFile.point_format
            ]  # ugly workaround to get actual strings

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
        omit_fields = ['X', 'Y', 'Z']
        dtypes = []
        dataDict = {'coords': coords}
        for name in las_fields:
            if name == 'flag_byte':
                values = points.flag_byte
                if np.any(values):
                    dataDict['num_returns'] = values // 8
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
    lasFields = [dim.name for dim in lasFile.point_format]

    # create user defined fields
    additional_fields = []
    omit_fields = lasFields + list(np.dtype(LasRecords.USER_DEFINED_FIELDS).names)
    for name in geoRecords.dtype.names:
        if name not in omit_fields:
            dtype = geoRecords.dtype[name]
            type_id = _dtype_to_laspy_type_id(dtype)
            lasFile.define_new_dimension(name, type_id, '')
            additional_fields.append(name)

    # set additional fields
    for name in additional_fields:
        lasFile._writer.set_dimension(name, geoRecords[name])

    field_names = geoRecords.dtype.names

    if 'return_num' in field_names or 'num_returns' in field_names:
        lasFile.flag_byte = np.zeros(len(geoRecords), dtype=np.uint)

    # set fields
    omit_fields = ['X', 'Y', 'Z']
    for name in field_names:
        if name == 'coords':
            lasFile.set_x_scaled(geoRecords.coords[:, 0])
            lasFile.set_y_scaled(geoRecords.coords[:, 1])
            lasFile.set_z_scaled(geoRecords.coords[:, 2])
        elif name == 'classification':
            lasFile.set_raw_classification(geoRecords.classification)
        elif name == 'return_num':
            lasFile.flag_byte += geoRecords.return_num
        elif name == 'num_returns':
            lasFile.flag_byte += geoRecords.num_returns * 8
        elif name not in omit_fields:
            lasFile._writer.set_dimension(name, geoRecords[name])

    # close file
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
    USER_DEFINED_FIELDS = [
        ('coords', np.float, 3),
        ('classification', np.uint8),
        ('num_returns', np.uint8),
        ('return_num', np.uint8),
    ]

    @property
    def last_return(self):
        return self.return_num == self.num_returns

    @property
    def first_return(self):
        return self.return_num == 1

    @property
    def only_return(self):
        return self.num_returns == 1

    def activate(self, field_name):
        """Activates a desired field on demand.

        Parameters
        ----------
        field_name : String
            Name of the field to activate.

        """
        for field in self.USER_DEFINED_FIELDS:
            if field[0] == field_name:
                return self.add_fields([field])
        raise ValueError('field "%s" not found' % field_name)

    def grd(self):
        """Filter by points classified as ground.

        Returns
        -------
        LasRecords
            Filtered records.

        """
        return self[self.class_indices(2, 11)]

    def veg(self):
        """Filter by points classified as vegetation.

        Returns
        -------
        LasRecords
            Filtered records.

        """
        return self[self.class_indices(3, 4, 5, 20)]

    def class_indices(self, *classes):
        """Filter by classes.

        Parameters
        ----------
        *classes : int
            Classes to filter by.

        Returns
        -------
        np.ndarray(int)
            Filtered record indices.

        """
        return np.where(np.in1d(self.classification, classes))[0]


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


