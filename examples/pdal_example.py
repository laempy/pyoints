#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:15:24 2018

@author: sebastian
"""

import numpy as np
import json as js

from pointspy.misc import *
from pointspy import (
    storage,
    projection,
    transformation,
)

import pdal

filename = "/daten/Seafile/PANTHEON_Data/Campaign_May_2018/survey_03/las/Scan_024.las"

T = transformation.matrix(t=[3, 2, 5], s=[2, 0.5, 0.4], r=[0.2, 0.1, 0.3])

t = " ".join(T.flatten().astype(str).tolist()[0])
print t
print


def mad(ins, outs):
    outs['Z'] = ins['Z'] * 2

json = {
    "pipeline": [
        {
            "type":"readers.las",
            "filename":filename,
        },
        {
            "type":"filters.transformation",
            #"matrix":"0.4 -1  0  1  1  0  0  2  0  0  1  3  0  0  0  1",
            "matrix":t,
        },
    ]
}
tic()
pipeline = pdal.Pipeline(unicode(js.dumps(json)))
#pipeline.validate() # check if our JSON and options were good
pipeline.loglevel = 0 #really noisy
count = pipeline.execute()
arrays = pipeline.arrays

toc()

tic()
pipeline = pdal.Pipeline(unicode(js.dumps(json)))

#pipeline.validate() # check if our JSON and options were good
#pipeline.loglevel = 8 #really noisy
count = pipeline.execute()
arrays = pipeline.arrays

toc()
metadata = pipeline.metadata
log = pipeline.log

las = arrays[0].view(np.recarray)
print las.shape
print las

tic()
lasReader = storage.LasReader(filename, proj=projection.Proj())
las = lasReader.load()
toc()
print las.shape
print las

tic()
print transformation.transform(las.coords, T)
toc()


