import numpy as np
from collections import defaultdict

import nptools
import projection
from indexkd import IndexKD
from extent import Extent


class GeoRecords(np.recarray, object):

    def __init__(self, proj, npRecarray, T=None):
        self._proj = proj
        if T is None:
            self._T = np.eye(self.dim + 1, dtype=float)
            self._T[0:self.dim, self.dim] = self.extent().minCorner

    def __new__(cls, proj, npRecarray, T=None):
        # np.racarray needed
        assert isinstance(
            proj, projection.Proj), 'Wrong Projection definition!'
        assert hasattr(npRecarray, 'coords'), 'Field "coords" needed!'
        return npRecarray.view(cls)

    @property
    def proj(self):
        return self._proj

    @property
    def t(self):
        return self._T

    @property
    def dim(self):
        return self.dtype['coords'].shape[0]

    @property
    def count(self):
        return np.product(self.shape)

    def extent(self, dim=None):
        if dim is None:
            dim = self.dim
        return Extent(self.records().coords[:, 0:dim])

    @staticmethod
    def keys(shape):
        if isinstance(shape, int):
            return np.arange(shape)
        else:
            return np.moveaxis(np.indices(shape), 0, -1)

    def ids(self):
        return self.__class__.keys(self.shape)

    def addFields(self, dtypes, data=None):
        newDtypes = self.dtype.descr + dtypes

        records = np.recarray(len(self), dtype=newDtypes)
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
        return self.__class__(self.proj, records, T=self.T)

    def addField(self, dtype, data=None):
        return self.addFields([dtype], [data])

    def merge(self, geoRecords):
        # TODO
        data = npTools.merge((self, geoRecords))
        return GeoRecords(self.proj, data).view(self.__class__)

    def records(self):
        return self.reshape(self.count)

    def project(self, proj):
        coords = projection.project(self.coords, self.proj, proj)
        return self.setCoords(proj, coords)

    def setCoords(self, proj, coords):
        self.coords = coords
        self._proj = proj
        self.clearCache()

    def clearCache(self):
        if hasattr(self, '_indices'):
            del self._indices

    # Cached

    def indexKD(self, dim=None):
        if dim is None:
            dim = self.dim
        if not hasattr(self, '_indices'):
            self._indices = defaultdict(lambda: None)
        indexKD = self._indices[dim]
        if indexKD is None:
            indexKD = IndexKD(self.records().coords[:, 0:dim])
            self._indices[dim] = indexKD
        return indexKD

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._T = getattr(obj, '_T', None)
        self._proj = getattr(obj, '_proj', None)

    def __array_wrap__(self, out_arr, context=None):
        #print '__array_wrap__'
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def apply(self, func, dtypes=[object]):
        return npTools.map(func, self, dtypes=dtypes)
