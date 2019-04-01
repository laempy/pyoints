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
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""Some random functions, which ease development.
"""

import time
import sys
import pkg_resources
import numpy as np
from numbers import Number


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


def sizeof_fmt(num, suffix='B'):
    """

    Notes
    -----
    Taken form [1]. Originally posted by [2].

    References
    ----------
    [1] jan-glx (2018), https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables,
    (acessed: 2018-08-16)
    [2] Fred Cirera, https://stackoverflow.com/a/1094933/1870254
    (acessed: 2018-08-16)

    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)

    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, (str, int, float)):
        pass
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif (hasattr(obj, '__iter__') and
            not isinstance(obj, (str, bytes, bytearray))):
        size += sum([get_size(i, seen) for i in obj])

    return size


def print_object_size(obj):
    """Get the size of cached objects.

    Parameters
    ----------
    obj : object
        Object to determine size from.

    """
    print(sizeof_fmt(get_size(obj)))


def print_rounded(values, decimals=2):
    """Prints rounded values.

    Parameters
    ----------
    values : Number or array_like(Number)
        Values to display.
    decimals : optional, int
        Number of decimals passed to 'numpy.round'.

    Notes
    -----
    Sets negative values close to zero to zero.

    """
    if isinstance(values, Number):
        rounded = np.round(values, decimals=decimals)
        if np.isclose(rounded, 0):
            rounded = rounded * 0
    elif isinstance(values, (np.ndarray, list, tuple)):
        rounded = np.round(values, decimals=decimals)
        rounded[np.isclose(rounded, 0)] = 0
    else:
        raise ValueError("Data type '%s' not supported" % type(values))

    if isinstance(values, tuple):
        rounded = tuple(rounded)

    np.set_printoptions(
        linewidth=75,
        precision=decimals,
        suppress=True,
        threshold=20,
        sign=' ',
        floatmode='unique',
        legacy='1.13'
    )
    print(rounded)
