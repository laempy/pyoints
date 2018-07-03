import dill as pickle

from .BaseGeoHandler import GeoFile


class DumpReader(GeoFile):

    def __init__(self, filename):
        GeoFile.__init__(self, filename)

    @property
    def proj(self):
        return self.load().proj4

    @property
    def corners(self):
        ext = self.load().extent()
        return ext.corners()

    def load(self):
        if not hasattr(self, '_records'):
            self._data = loadDump(self.filename)
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
    with open(file, 'r') as f:
        return pickle.load(f)


def writeDump(obj, filename):
    """Dump a object.

    Parameters
    ----------
    obj : object
        Object to dump
    filename : String
        File to dump object to.

    """
    with open(filename, 'w') as f:
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
