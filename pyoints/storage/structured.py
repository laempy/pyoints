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
"""Reading and writing of common structured file types.
"""
import os
import json

from .. import assertion


def loadJson(infile):
    """Loads a JSON file from disc.

    Parameters
    ----------
    infile : str
        Input file.

    Returns
    -------
    dict
        File data.

    """
    if not os.path.isfile(infile):
        raise IOError('file "%s" not found' % infile)
    with open(infile, 'r') as f:
        js = json.load(f)
    return js


def writeJson(data, outfile):
    """Writes a JSON file to disk.

    Parameters
    ----------
    data : dict
        JSON compatible data to store.
    outfile : str
        Output JSON file.

    """
    data = assertion.ensure_json(data)
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4)
