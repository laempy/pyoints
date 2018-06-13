# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:48:11 2018

@author: sebastian
"""

from pointspy import (
    storage,
    projection,
)

infile = '/home/sebastian/Schreibtisch/temp/pantheon/Scan_070.ply'
outfile = '/home/sebastian/Schreibtisch/temp/pantheon/Scan_070_out.ply'
proj = projection.Proj.from_epsg(26592)


ply = storage.loadPly(infile, proj)
print ply
print ply.dtype

storage.writePly(ply, outfile)