# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Test to read and write .ply files."""

import os

from pointspy import (
    storage,
)

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Create GeoRecords from scratch
center = [332592.88, 5513244.80, 120]
geoRecords = storage.misc.create_random_GeoRecords(center=center, epsg=25832)


# save PLY file
outfile = os.path.join(outpath, 'test.ply')
print("save %s" % outfile)
storage.writePly(geoRecords, outfile)


# load PLY file
print("load %s" % outfile)
ply = storage.loadPly(outfile)
print(ply)
