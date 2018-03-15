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


class GeoRecords(np.recarray, object):
    """Abstraction class to ease handling of point sets as well as structured
    matrices of point like objects. This gives the oportunity to handle
    unstructured point sets the same way like rasters or voxels. The class also
    provides a IndexKD object on demand to speed up neigbourhod analyzes.

    Parameters
    ----------
    proj : `Proj`
        Projection object to provide the coordinate reference system.
    npRecarray : np.recarray(shape=(n,))
        A numpy record array provides coordinates and attributes of `n` points.
        The field `coords` is required and represents coordinates of `k`
        dimensions.
    T : optional, array_like(shape=(k+1,k+1))
        Linear transformation matrix to transform the coordinates. Set to a
        identity matrix if not given. This eases handling of structured point
        sets like rasters or voxels.

    Attributes
    ----------
    proj : `Proj`
        Projection of the coordinates.
    t : array_like(shape=(k+1,k+1))
        Linear transformation matrix to transform the coordinates.
    dim : positive int
        Number of coordinate dimensions of the `coords` field.
    count : positive int
        Number number of objects in the data structure. E.g. number of points
        in the point cloud or number of cells in the raster.

    """

    def __init__(self, proj, rec, T=None):
        self.proj = proj
        if T is None:
            self._t = transformation.t_matrix(self.extent().min_corner)
        else:
            self._t = assertion.enure_tmatrix(T)
        self._clear_cache()

    def __new__(cls, proj, rec, T=None):
        if not isinstance(proj, projection.Proj):
            raise ValueError("'proj' needs to be of type 'projection.Proj'")
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
        self._clear_cache()

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def _clear_cache(self):
        # Deletes cached data.
        self._indices = {}

    @property
    def t(self):
        return self._t

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
        shape: `int or array_like`
            Shape of desired output array.

        Returns
        -------
        keys: `np.ndarray`
            Array of indices with desired shape. Each entry provides a index
            tuple to call `__getitem__`.
        """
        if not (isinstance(shape, int) or hasattr(shape, '__len__')):
            raise ValueError("'shape' has to be an integer or array like")

        if isinstance(shape, int):
            return np.arange(shape)
        else:
            # TODO moveaxis?
            return np.moveaxis(np.indices(shape), 0, -1)

    def extent(self, dim=-1):
        """Provides the spatial extent of the data structure.

        Parameters
        ----------
        dim: optional, int
            Define which coordinates to use. If not given all dimensions are
            used.

        Returns
        -------
        extent: `Extent`
            Spatial extent of the coordinates.
        """
        if not isinstance(dim, int):
            raise ValueError("'dim' needs to be of type int")
        if not abs(dim) <= self.dim:
            raise ValueError("'dim' inappropiate dimension selected")
        return Extent(self.records().coords[:, :dim])

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
        return self.setCoords(proj, coords)

    def set_coords(self, proj, coords):
        """Set new coordinates and projection.

        Parameters
        ----------
        proj: `Proj`
            Projection object provides the geograpic projection of the points.
        coords: `numpy.ndarray`
            New coordinates of the data structure.
        """
        assert isinstance(coords, np.ndarray)
        assert isinstance(proj, projection.Proj)
        assert self.coords.shape == coords.shape

        self['coords'] = coords
        self._proj = proj
        self._clear_cache()

    def indexKD(self, dim=None):
        """Spatial index of the coordinates.

        Parameters
        ----------
        dim: optional, `int`
            Desired dimension of the spatial index. If None the all coordinate
            dimensions are used.

        Returns
        -------
        indexKD: `IndexKD`
            Spatial index of the coordinates with disired dimension.
        """
        if dim is None:
            dim = self.dim
        indexKD = self._indices.get(dim)
        if indexKD is None:
            indexKD = IndexKD(self.records().coords[:, 0:dim])
            self._indices[dim] = indexKD
        return indexKD

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
