import os
from .. import (
    assertion,
    projection
)


class GeoFile:
    """Interface to read files containing spatial information.

    Parameters
    ----------
    infile : String
        File to be read. It should contain data to be interpreted as points of
        `k` dimensions.
    directory : bool
        Indicates if the file is a comosite of several files stored in a
        directory.

    Properties
    ----------
    t : np.matrix(Number, shape=(k+1, k+1))
        Transformation matrix to transform the `k`-dimensional points. Usually
        this matrix defines the origin of a local coordinate system.
    proj : Proj
        Coordinate projection system.
    extent : Extent(Number, shape=(2*k))
        Defines the spatial extent of the points.
    corners : np.ndarray(Number, shape=(2**k, k))
        Corners of the extent.
    date : Date
        File creation date.

    See Also
    --------
    Proj, Extent

    """

    def __init__(self, infile, directory=False):
        if directory:
            if not os.path.isdir(infile):
                raise IOError('directory "%s" not found' % infile)
        elif not os.path.isfile(infile):
            raise IOError('file "%s" not found' % infile)

        self.file_name, self.extension = os.path.splitext(
            os.path.basename(infile))
        self.extension = self.extension[1:]
        self.path = os.path.dirname(infile)
        self.file = os.path.abspath(infile)

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
        """Return the number of points.

        Returns
        -------
        positive int
            Number of objects within the file.

        """
        raise NotImplementedError()

    def load(self, extent=None):
        """Load data on demand.

        Parameters
        ----------
        extent : optional, array_like(Number, shape=(2*self.dim))
            Defines in which volume or area points shall be loaded.

        Returns
        -------
        GeoRecords
            Desired data of the file.
        """

        raise NotImplementedError()

    def clean_cache(self):
        """Cleans all cached data.
        """
        raise NotImplementedError()
