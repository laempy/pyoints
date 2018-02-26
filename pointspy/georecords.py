import numpy as np

from .indexkd import IndexKD
from .extent import Extent
from . import transformation 
from . import projection


# TODO module description

class GeoRecords(np.recarray, object):
    """Abstraction class ease handling of point sets as well as structured 
    matrices of point like objects. This gives the oportunity to handle
    unstructured point sets the same way like rasters or voxels. The class also
    provides a IndexKD object on demand to speed up neigbourhod analyzes.

    Parameters
    ----------
    proj: `Proj`
        Projection object provides the geograpic projection of the points.
    npRecarray: (n,), `numpy.recarray`
        A numpy record array provides coordinates and attributes of n points.
        Field "coords" representing coordinates with k dimensions is 
        strictly required.
    T: optional, (k+1,k+1), `array_like`
        Optional linear transformation matrix to transform the coordinates.
        This eases handling of structured point sets like rasters or voxels.
    """

    def __init__(self, proj, npRecarray, T=None):
        self._proj = proj
        if T is None:
            # set transformation matrix to minimum corner
            self._t = transformation.t_matrix(self.extent().min_corner)
        self._clear_cache()


    def __new__(cls, proj, npRecarray, T=None):
        assert isinstance(
            proj, projection.Proj), 'Wrong Projection definition!'
        assert hasattr(npRecarray, 'coords'), 'Field "coords" needed!'
        assert len(npRecarray.coords.shape)==2 and npRecarray.coords.shape>=2, 'At least two coordinate dimensions needed!'

        return npRecarray.view(cls)

    @property
    def proj(self):
        """Provides projection object.

        Returns
        -------
        proj: `Proj`
            Provides geograpic projection of point coordinates.
        """
        return self._proj

    @property
    def t(self):
        """Provides linear transformation matrix.

        Returns
        -------
        T: optional, (self.dim+1,self.dim+1), `array_like`
            Linear transformation matrix.
        """
        return self._t

    @property
    def dim(self):
        """Provides the number coordinate dimensions.

        Returns
        -------
        dim: positive `int`
            Number of coordinate dimensions of the "coords" field.
        """
        return self.dtype['coords'].shape[0]

    @property
    def count(self):
        """Provides the number of objects organized in the data structure.

        Returns
        -------
        count: positive `int`
            Number number of objects in the data structure. E.g. number of 
            points in the point cloud or number of cells in the raster.
        """
        return np.product(self.shape)

    def extent(self, dim=-1):
        """Provides the spatial extent of the data structure.

        Parameters
        ----------
        dim: optional, `int`
            Define which coordinates to use.

        Returns
        -------
        extent: `Extent`
            Spatial extent of the coordinates.
        """
        return Extent(self.records().coords[:,:dim])

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
        assert isinstance(shape, int) or hasattr(shape,'__len__')
        
        if isinstance(shape, int):
            return np.arange(shape)
        else:
            # TODO moveaxis?
            return np.moveaxis(np.indices(shape), 0, -1)

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

        #print np.lib.recfunctions.append_fields(self,B,B.dtype)
        # data=np.lib.recfunctions.rec_append_fields(self.flatten(),(name),B.flatten()).reshape(self.shape)
        #d=[self[field] for field in self.dtype.names]
        # data=np.lib.recfunctions.rec_append_fields(B,self.dtype.names,d)#.reshape(self.shape)
        # data=np.lib.recfunctions.merge_arrays([B,self],flatten=True,usemask=False,asrecarray=True)
        # TODO sich selbst uberschreiben
        return self.__class__(self.proj, records, T=self.T)

    def add_field(self, dtype, data=None):
        return self.add_fields([dtype], [data])

    def merge(self, geoRecords):
        # TODO
        data = npTools.merge((self, geoRecords))
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
        assert isinstance(coords,np.ndarray)
        assert isinstance(proj,Proj)
        assert self.coords.shape == coords.shape
        
        self.coords = coords
        self._proj = proj
        self._clear_cache()

    def _clear_cache(self):
        """Deletes all cached data.
        """
        self._indices = {}


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

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._t = getattr(obj, '_t', None)
        self._proj = getattr(obj, '_proj', None)
        self._clear_cache()

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

