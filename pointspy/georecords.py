import numpy as np

from .indexkd import IndexKD
from .extent import Extent

from . import (
    transformation,
    projection,
    assertion,
    nptools,
)

# TODO module description
# TODO examples


class GeoRecords(np.recarray, object):
    """Abstraction class to ease handling of point sets as well as structured
    matrices of point like objects. This gives the oportunity to handle
    unstructured point sets the same way like rasters or voxels. The class also
    provides a IndexKD object on demand to speed up neigbourhod analyzes.

    Parameters
    ----------
    proj : projection.Proj
        Projection object to provide the coordinate reference system.
    npRecarray : np.recarray(shape=(n,))
        A numpy record array provides coordinates and attributes of `n` points.
        The field `coords` is required and represents coordinates of `k`
        dimensions.
    T : optional, array_like(Number, shape=(k+1, k+1))
        Linear transformation matrix to transform the coordinates. Set to a
        identity matrix if not given. This eases handling of structured point
        sets like rasters or voxels.

    Attributes
    ----------
    proj : projection.Proj
        Projection of the coordinates.
    t : np.matrix(Number, shape=(k+1, k+1))
        Linear transformation matrix to transform the coordinates.
    dim : positive int
        Number of coordinate dimensions of the `coords` field.
    count : positive int
        Number of objects within the data structure. E.g. number of points
        of a point cloud or number of cells of a raster.

    Examples
    --------

    >>> import pointspy.nptools
    >>> import pointspy.projection
    >>> coords = [(2, 3), (3, 2), (0, 1), (-1, 2.2), (9, 5)]
    >>> values = [1, 3, 4, 0, 6]
    >>> rec = nptools.recarray({'coords': coords, 'values': values})
    >>> geo = GeoRecords(projection.Proj(),rec)
    >>> print geo.values
    [1 3 4 0 6]
    >>> print geo.coords
    [[ 2  3]
     [ 3  2]
     [ 0  1]
     [-1  2]
     [ 9  5]]
    >>> new_coords = [(1, 2), (9, 2), (8, 2), (-7, 3), (7, 8)]
    >>> geo['coords'] = new_coords
    >>> print geo.coords
    [[ 1  2]
     [ 9  2]
     [ 8  2]
     [-7  3]
     [ 7  8]]

    """

    def __init__(self, proj, rec, T=None):
        self.proj = proj    # validated by setter
        if T is None:
            T = transformation.t_matrix(self.extent().min_corner)
        self.t = T    # validated by setter

    def __new__(cls, proj, rec, T=None):
        if not isinstance(rec, np.recarray):
            raise ValueError("'rec' needs to be of type 'np.recarray'")
        if 'coords' not in rec.dtype.names:
            raise ValueError("field 'coords' needed")
        if not len(rec.dtype['coords'].shape) == 1:
            raise ValueError("malformed coordinate shape")
        if not rec.dtype['coords'].shape[0] >= 2:
            raise ValueError("at least two coordinate dimensions needed")
        return rec.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._t = getattr(obj, '_t', None)
        self._proj = getattr(obj, '_proj', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __setitem__(self, key, value):
        # TODO single edit?
        # TODO write protection
        if key is 'coords':
            # clear cache, when setting new coords
            np.recarray.__setitem__(self, key, value)
            self._clear_cache()

    def _clear_cache(self):
        # Deletes cached data.
        if hasattr(self, '_indices'):
            del self._indices
        if hasattr(self, '_extents'):
            del self._extents

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        t = assertion.ensure_tmatrix(t)
        if not t.shape[0] == self.dim + 1:
            raise ValueError('dimensions do not fit')
        self._t = t
        self._clear_cache()

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, proj):
        if not isinstance(proj, projection.Proj):
            raise ValueError("'proj' needs to be of type 'projection.Proj'")
        self._proj = proj

    @property
    def dim(self):
        return self.dtype['coords'].shape[0]

    @property
    def count(self):
        return np.product(self.shape)

    @staticmethod
    def keys(shape):
        """Keys or indices of a numpy ndarray.

        Parameters
        ----------
        shape : array_like(int)
            Shape of desired output array.

        Returns
        -------
        np.ndarray(int, shape=shape)
            Array of indices with desired shape. Each entry provides a index
            tuple to access the array entries.

        Examples
        --------

        >>> print GeoRecords.keys((3,4))
        s

        """
        if isinstance(shape, int):
            return np.arange(shape)
        else:
            shape = assertion.ensure_numvector(shape)
            if not nptools.isnumeric(shape, dtypes=[int]):
                raise ValueError("'shape' values have to be integers")
            #return np.indices(shape)
            # TODO moveaxis?
            return np.moveaxis(np.indices(shape), 0, -1)


    def extent(self, dim=None):
        """Provides the spatial extent of the data structure.

        Parameters
        ----------
        dim: optional, positive int
            Define which coordinates to use. If not given all dimensions are
            used.

        Returns
        -------
        extent: `Extent`
            Spatial extent of the coordinates.
        """
        if dim is None:
            dim = self.dim
        elif not (isinstance(dim, int) and dim > 0 and dim <= self.dim):
            raise ValueError("'dim' needs to be an int in range(2,self.dim)")

        # use cache for performance reasons
        if not hasattr(self, '_extents'):
            self._extents = {}
        ext = self._extents.get(dim)
        if ext is None:
            ext = Extent(self.records().coords[:, :dim])
            self._extents[dim] = ext
        return ext

    def indexKD(self, dim=None):
        """Spatial index of the coordinates.

        Parameters
        ----------
        dim: optional, positive int
            Desired dimension of the spatial index. If None the all coordinate
            dimensions are used.

        Returns
        -------
        indexKD: `IndexKD`
            Spatial index of the coordinates with disired dimension.
        """
        if dim is None:
            dim = self.dim
        elif not (isinstance(dim, int) and dim > 0 and dim <= self.dim):
            raise ValueError("'dim' needs to be an int in range(1,self.dim)")

        # use cache for performance reasons
        if not hasattr(self, '_indices'):
            self._indices = {}
        indexKD = self._indices.get(dim)
        if indexKD is None:
            indexKD = IndexKD(self.records().coords[:, 0:dim])
            self._indices[dim] = indexKD
        return indexKD

    def ids(self):
        """Keys or indices of the data structure.

        Returns
        -------
        ids: `np.ndarray`
            Array of indices. Each entry provides a index tuple to recieve a
            data element wth `__getitem__`.
        """
        return self.__class__.keys(self.shape)

    def add_fields(self, dtypes, data=None):
        # TODO use nptools
        # TODO overwrite self?
        # TODO
        newDtypes = self.dtype.descr + dtypes

        # todo len nicht valide
        records = np.recarray(self.shape, dtype=newDtypes)
        for field in self.dtype.names:
            records[field] = self[field]
        if data is not None:
            for field, column in zip(dtypes, data):
                if column is not None:
                    records[field] = column
        return self.__class__(self.proj, records, T=self.T)

    def add_field(self, dtype, data=None):
        return self.add_fields([dtype], [data])

    def merge(self, geoRecords):
        # TODO
        data = nptools.merge((self, geoRecords))
        # TODO sich selbst uberschreiben
        return self.__class__(self.proj, data, T=self.T)

    def records(self):
        # TODO
        return self.reshape(self.count)

    def project(self, proj):
        # TODO
        coords = projection.project(self.coords, self.proj, proj)
        return self.set_coords(proj, coords)

    def apply(self, func, dtypes=[object]):
        """Applies or maps a function to each element of the data array.

        Parameters
        ----------
        func:  `function`
            Function to apply to each element of the data array.
        dtypes:  optional, `np.dtype`
            Data type description of the output array.

        Returns
        -------
        applyed: `np.ndarray`
            Array of the same shape as the data array.
        """

        return nptools.map(func, self, dtypes=dtypes)
