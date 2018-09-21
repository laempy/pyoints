"""Helper functions for data type conersion.
"""

import numpy as np
from osgeo import gdal

# Laspy
########

"""
# see https://pythonhosted.org/laspy/tut_part_3.html
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
"""

LASPY_TYPE_MAP = [
    (1, ['|u1']),
    (2, ['|S1']),
    (3, ['<u2']),
    (4, ['<i2']),
    (5, ['<u4']),
    (6, ['<i4', '<i8']),
    (7, ['<u8']),
    (8, []),  # error occurs
    (9, ['<f4']),
    (10, ['<f8']),
]

LASPY_TO_NUMPY_TYPE = {}
for dim in range(1, 4):
    for key, t in LASPY_TYPE_MAP:
        if len(t) > 0:
            type_id = key + len(LASPY_TYPE_MAP) * (dim - 1)
            LASPY_TO_NUMPY_TYPE[type_id] = (t[0], dim)

NUMPY_TO_LASPY_TYPE = {}
for dim in range(1, 4):
    NUMPY_TO_LASPY_TYPE[dim] = {}
    for t, p in LASPY_TYPE_MAP:
        type_id = t + len(LASPY_TYPE_MAP) * (dim - 1)
        for key in p:
            NUMPY_TO_LASPY_TYPE[dim][key] = type_id


def numpy_to_laspy_dtype(dtype):
    """Converts a numpy data type to a laspy data type.
    
    Parameters
    ----------
    dtype : np.dtype
        Numpy data type to convert.
        
    Returns
    -------
    int
        Laspy data type id.
    
    Examples
    --------
    
    >>> dtype = np.dtype(np.int32)
    >>> print(numpy_to_laspy_dtype(dtype))
    6
    
    """
    dtype = np.dtype(dtype)
    if dtype.subdtype is None:
        dt = dtype
        type_name = dt.str
        type_dim = dt.shape[0] if len(dt.shape) > 0 else 1
    else:
        dt = dtype.subdtype
        type_name = dt[0].str
        type_dim = dt[1][0] if len(dt[1]) > 0 else 1
    if type_dim not in NUMPY_TO_LASPY_TYPE.keys():
        return None
    if type_name not in NUMPY_TO_LASPY_TYPE[type_dim].keys():
        return None
    return NUMPY_TO_LASPY_TYPE[type_dim][type_name]


# GDAL
#######

NUMPY_TO_GDAL_TYPE = {
    '|u1' : gdal.GDT_Byte,
    '|i1' : gdal.GDT_Byte,
    '<u2' : gdal.GDT_UInt16,
    '<i2' : gdal.GDT_Int16,
    '<u4' : gdal.GDT_UInt32,
    '<i4' : gdal.GDT_Int32,
    '<u8' : gdal.GDT_Float32,
    '<i8' : gdal.GDT_Float32,
    '<f2' : gdal.GDT_Float32,
    '<f4' : gdal.GDT_Float32,
    '<f8' : gdal.GDT_Float64,
    '<c8' : gdal.GDT_CFloat32,
    '<c16' : gdal.GDT_CFloat64,
}

def numpy_to_gdal_dtype(dtype):
    """Converts a numpy data type to a gdal data type.
    
    Parameters
    ----------
    dtype : np.dtype
        Numpy data type to convert.
        
    Returns
    -------
    int
        Gdal data type id.
    
    Examples
    --------
    
    >>> dtype = np.dtype(np.int32)
    >>> print(numpy_to_gdal_dtype(dtype))
    5
    
    """
    dtype = np.dtype(dtype)
    key = dtype.str
    if key not in NUMPY_TO_GDAL_TYPE:
        raise ValueError("data type '%s' not found" % key)
    return NUMPY_TO_GDAL_TYPE[key]