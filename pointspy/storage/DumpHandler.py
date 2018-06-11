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
            print('load')
            self._records = loadDump(self.filename)
        return self._records

    def clean_cache(self):
        del self._records


def loadDump(filename):
    with open(file, 'r') as f:
        return pickle.load(f)


def writeDump(obj, file):
    with open(file, 'w') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def dumpstring2py(string):
    if string is None:
        return None
    return pickle.loads(string.decode(encoding='base64', errors='strict'))


def py2dumpstring(data):
    return "%s" % pickle.dumps(
        data, pickle.HIGHEST_PROTOCOL).encode(
        encoding='base64', errors='strict')
