# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:51:38 2018

@author: sebastian
"""

import os
import glob
import numpy as np

from pointspy import (
    IndexKD,
    storage,
    projection,
    distance,
    transformation,
    filters,
)


infile = '/daten/Seafile/PANTHEON_Data/Campaign_May_2018/field18/Las/Scan_070.las'
outfile = '/home/sebastian/Schreibtisch/test.las'

wgs84 = projection.Proj.from_epsg(4326)
proj_rome = projection.Proj.from_epsg(26592)
project = projection.GeoTransform(wgs84, proj_rome)

t = project([(12.29821763, 42.27982770, 279.664)])[0, :]
T = transformation.t_matrix(t)

lasReader = storage.LasReader(infile, proj=proj_rome)
print lasReader.proj.proj4
las = lasReader.load()
las.t = T
las.coords = transformation.transform(las.coords, las.t)

#las = las.add_fields([('lol', float)])
las = las.add_fields([('new_awsom_field', 'i4'), ('z', float)])
las.new_awsom_field = np.arange(len(las))
las.z = las.coords[:, 2]
print las.extent()

print np.round(las.t, 2)

lasReader = storage.writeLas(las, outfile)
print lasReader.proj.proj4
las = lasReader.load()
print las.extent()
print np.round(las.t.astype(float), 2)

asd

# config
laspath = '/daten/Seafile/PANTHEON_Data/Campaign_May_2018/field18/Las/'
flspath = '/daten/Seafile/PANTHEON_Data/Campaign_May_2018/field18/Faro/'
outpath = '/daten/Seafile/PANTHEON_Data/Campaign_May_2018/field18/output/'


# projection system
wgs84 = projection.Proj.from_epsg(4326)
italy_rome = projection.Proj.from_epsg(26592)
project = projection.CoordinateTransform(wgs84, italy_rome)

t = project([(42.27982770, 12.29821763, 279.664)])[0, :]

# get files
lasfiles = glob.glob(os.path.join(laspath, '*.las'))
rtk = np.genfromtxt(os.path.join(laspath, 'RTK.csv'), delimiter=',')

rtk = project(rtk)
print np.round(rtk, 0)

# TODO get altitude from

np.savetxt(os.path.join(outpath, 'RTK_italy.csv'), rtk)

for pos, filename in zip(rtk, lasfiles):

    print filename
    lasReader = storage.LasReader(filename, proj=italy_rome)
    las = lasReader.load()

    flsfile = os.path.join(flspath, '%s.fls'% lasReader.file_name)
    flsReader = storage.FlsReader(flsfile, proj=italy_rome)
    print np.round(flsReader.t, 2)
    altitude = flsReader.t[2, 3]

    print 'max_dist'
    sdists = distance.sdist([0, 0, altitude], las.coords)
    print sdists
    max_dist = 15
    mask = sdists < (max_dist ** 2)
    las = las[mask]
    sdists = sdists[mask]

    if False:
        print 'filter'
        fIds = list(filters.ball(
            IndexKD(las.coords),
            0.002,
            min_pts = 1,
            order=np.argsort(sdists)
        ))
        las = las[fIds]

    t = (pos[0], pos[1], pos[2] - altitude)
    T = transformation.t_matrix(t)
    print T
    # TODO just one matrix
    #M = np.linalg.inv(las.t) * T
    #print M

    #coords = transformation.transform(las.coords, M)
    #print coords

    #las.coords = transformation.transform(las.coords, las.t, inverse=True)

    las.coords = transformation.transform(las.coords, T)
    print las.t
    print las.coords


    outfile = os.path.join(outpath, '%s.las' % lasReader.file_name)
    print outfile
    storage.writeLas(las, outfile)
    #asddas
adsda
exit(0)



infile = '/daten/Seafile/PANTHEON_Data/Campaign_May_2018/field18/Las/Scan_070_test.las'

#lasReader = storage.LasReader(infile)

offset = [50, -20, 300]
storage.updateLasHeader(infile, offset)

infile = '/daten/Seafile/PANTHEON_Data/Campaign_May_2018/field18/Las/Scan_069_test.las'

t = project([(42.27982770, 12.29821763, 279.664)])[0, :]
print(t)
storage.updateLasHeader(infile, translate=t)

# TODO origin?

infile = '/daten/Seafile/PANTHEON_Data/Campaign_May_2018/field18/Las/Scan_071_test.las'

t = project(np.array([[42.27983729, 12.29815610, 278.099]]))[0, :]
print(t)
storage.updateLasHeader(infile, translate=t)