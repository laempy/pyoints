# BEGIN OF LICENSE NOTE
# This file is part of Pointspy.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
#
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Some random funtions, which ease development.
"""

import time
import pkg_resources


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() -
                                       startTime_for_tictoc) + " seconds.")
        return time.time() - startTime_for_tictoc
    else:
        print("Toc: start time not set")
        return None


def list_licences(requirements_file):
    with open(requirements_file) as f:
        package_list = f.readlines()
    package_list = [pkgname.strip() for pkgname in package_list]
    for pkgname in package_list:
        try:
            pkg = pkg_resources.require(pkgname)[0]
        except BaseException:
            print("package '%s' not found" % pkgname)
            continue

        try:
            lines = pkg.get_metadata_lines('METADATA')
        except BaseException:
            lines = pkg.get_metadata_lines('PKG-INFO')

        for line in lines:
            if line.startswith('License:'):
                m = "%s : %s" % (str(pkg), line[9:])
                print(m)
