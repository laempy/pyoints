# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:03:44 2018

@author: sebastian
"""
import os
from pointspy import (
    nptools,
    storage,
)

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# create file
data = nptools.recarray({
    'coords': [(0, 1.1), (1, 2), (3, 4), (1, 1), (0, 0)],
    'values': [3, 2, 4, 1, 5],
    'text': ['A', 'B', 'C', 'D', 'E']
})
print(data)
outfile = os.path.join(outpath, 'test.csv')
print('Save %s' % outfile)
storage.writeCsv(data, outfile)


# load file
# dtype = [('text', object), ('coords', float, 2), ('values', int)]
dtype = None
data = storage.loadCsv(outfile, header=True, dtype=dtype)
print(data)
print(data.dtype)
