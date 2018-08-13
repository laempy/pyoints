# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University, 
# lamprecht@uni-trier.de
# 
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
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
