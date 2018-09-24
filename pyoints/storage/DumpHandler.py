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
import codecs
import dill as pickle

from .BaseGeoHandler import GeoFile


class DumpReader(GeoFile):
    """Class to read GeoRecords from python dump files.

    See Also
    --------
    GeoFile

    """

    def __init__(self, filename):
        GeoFile.__init__(self, filename)

    @property
    def proj(self):
        return self.load().proj4

    @property
    def corners(self):
        return self.extent.corners

    @property
    def extent(self):
        return self.load().extent()

    @property
    def t(self):
        return self.load().t

    def load(self):
        if not hasattr(self, '_records'):
            self._data = loadDump(self.file)
        return self._data

    def clean_cache(self):
        del self._data


def loadDump(filename):
    """Loads a dump file.

    Parameters
    ----------
    filename : String
        Dump file to load.

    Returns
    -------
    object

    """
    with open(filename, 'rb') as f:
        return pickle.load(f, pickle.HIGHEST_PROTOCOL)


def writeDump(obj, filename):
    """Dump an object to a file.

    Parameters
    ----------
    obj : object
        Object to dump
    outfile : String
        File to dump object to.

    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def dumpstring_to_object(pickled):
    """Converts a dump string to an object.

    Parameters
    ----------
    string: string
        Dump string

    Returns
    -------
    object
        Loaded object.

    """
    if pickled is None:
        return None
    unpickled = pickle.loads(codecs.decode(pickled.encode(), "base64"))
    return unpickled


def dumpstring_from_object(data):
    """Converts an object to a dump string.

    Parameters
    ----------
    string: string
        Dump string

    Returns
    -------
    string
        Dump string.

    """
    dump = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    pickled = codecs.encode(dump, "base64").decode()
    return pickled
