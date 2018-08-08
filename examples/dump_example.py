# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Save and load dump files."""

import os

from pointspy import storage


outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Create GeoRecords from scratch
center = [332592.88, 5513244.80, 120]
geoRecords = storage.misc.create_random_GeoRecords(center=center, epsg=25832)
print(geoRecords.dtype.descr)

# save DUMP file
outfile = os.path.join(outpath, 'test.pydump')
print("save %s" % outfile)
storage.writeDump(geoRecords, outfile)

# load DUMP file
print("load %s" % outfile)
dumpReader = storage.DumpReader(outfile)
print(dumpReader.t.origin)
geoRecords = dumpReader.load()
print(geoRecords.dtype.descr)
