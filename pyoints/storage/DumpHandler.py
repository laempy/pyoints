# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
# END OF LICENSE NOTE
import dill as pickle

from .BaseGeoHandler import GeoFile


class DumpReader(GeoFile):
    """Class to read GeoRecords form python dump files.

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
    """Dump a object to a file.

    Parameters
    ----------
    obj : object
        Object to dump
    outfile : String
        File to dump object to.

    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def dumpstring_to_object(string):
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
    if string is None:
        return None
    return pickle.loads(string.decode(encoding='base64', errors='strict'))


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
    return "%s" % pickle.dumps(
        data, pickle.HIGHEST_PROTOCOL).encode(
        encoding='base64', errors='strict')
