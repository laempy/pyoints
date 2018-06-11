import os
from .. import (
    assertion,
    projection
)


class GeoFile:

    def __init__(self, filename, directory=False):
        if directory:
            if not os.path.isdir(filename):
                raise IOError('directory "%s" not found' % filename)
        elif not os.path.isfile(filename):
            raise IOError('file "%s" not found' % filename)

        self.file_name, self.extension = os.path.splitext(
            os.path.basename(filename))
        self.extension = self.extension[1:]
        self.path = os.path.dirname(filename)
        self.file = os.path.abspath(filename)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        t = assertion.ensure_tmatrix(t)
        self._t = t

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, proj):
        if not isinstance(proj, projection.Proj):
            raise ValueError("'proj' needs to be of type 'projection.Proj'")
        self._proj = proj

    @property
    def extent(self):
        raise NotImplementedError()

    @property
    def corners(self):
        raise NotImplementedError()

    @property
    def date(self):
        return None

    def __len__():
        raise NotImplementedError()

    def load(self, extent=None):
        raise NotImplementedError()

    def clean_cache(self):
        raise NotImplementedError()
