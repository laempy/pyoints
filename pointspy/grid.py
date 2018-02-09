import numpy as np
import pandas as pd

import transformation
import nptools
import vector

from georecords import GeoRecords
from extent import Extent


def corners2Transform(corners, resolutions):

    sT = transformation.s_matrix(resolutions)
    tT = transformation.t_matrix(corners[0, :])

    dX = float(corners[0, 0] - corners[1, 0])
    dY = float(corners[0, 1] - corners[1, 1])

    alpha = np.tan(dY / dX)
    rT = transformation.r_matrix(alpha)
    #raise NotImplementedError()
    T = tT * sT * rT
    # T=tT*sT#*rT
    return T


def transform2corners(T, shape):
    ext = Extent(np.concatenate([np.zeros(len(shape)), shape]))
    return Grid.keys2coords(T, ext.corners())


class Grid(GeoRecords):

    def __init__(self, proj, npRecarray, T):
        self._proj = proj
        self._T = T

    def __new__(cls, proj, npRecarray, T):
        # origin: left upper corner
        dim = len(T) - 1
        shape = npRecarray.shape

        data = np.zeros(
            shape, dtype=[
                ('coords', float, dim)]).view(
            np.recarray)
        keys = cls.keys(data.shape)
        data['coords'] = cls.keys2coords(T, keys)

        # if isinstance(attributes,np.recarray):
        #    attr=attributes
        # else:
        #    attr=np.recarray(attributes.shape,dtype=[('values',attributes.dtype)])
        #    attr['values']=attributes
        data = npTools.fuse(data, npRecarray)
        grid = GeoRecords(proj, data).reshape(shape).view(cls)

        return grid

    @staticmethod
    def coords2keys(T, coords):
        localSystem = transformation.LocalSystem(T)
        dim = localSystem.dim
        s = np.product(coords[:, 0:dim].shape) / dim
        flatCoords = coords[:, 0:dim].view().reshape((s, dim))
        values = localSystem.toGlobal(flatCoords)
        # keys=np.array(np.round(values),dtype=int)[:,::-1]
        keys = np.array(np.floor(values), dtype=int)[:, ::-1]
        return keys.reshape(coords[:, 0:dim].shape)

    @staticmethod
    def keys2coords(T, keys):
        localSystem = transformation.LocalSystem(T)
        dim = localSystem.dim
        s = np.product(keys.shape) / dim
        flatKeys = keys.view().reshape((s, dim))[:, ::-1] + 0.5
        values = localSystem.toLocal(flatKeys)
        coords = np.array(values, dtype=float)
        return coords.reshape(keys.shape)

    @staticmethod
    def coords2coords(transform, coords):
        return Grid.keys2coords(transform, Grid.coords2keys(transform, coords))

    @staticmethod
    def extentInfo(transform, shape, extent):
        dim = len(extent) / 2

        cornerIndices = Grid.coords2keys(transform, extent.corners())

        shape = np.ptp(cornerIndices, axis=0) + 1

        # Minimum corner
        minCornerIdx = np.argmin(cornerIndices, axis=0)
        minCornerIndex = np.array([cornerIndices[idx, i]
                                   for i, idx in enumerate(minCornerIdx)], dtype=int)
        minCorner = Grid.keys2coords(
            transform, np.array([minCornerIndex - 0.5]))[0, :]

        # define transformation
        T = np.copy(transform)
        T[0:dim, dim] = minCorner

        return T, minCornerIndex, shape

    def windowByExtent(self, extent):
        T, cornerIndex, shape = self.extentInfo(
            self.transform, self.shape, extent)
        mask = self.keys(shape) + cornerIndex
        return self[zip(mask.T)].reshape(shape)

    def gdalGeoTransform(self):
        return (self.t[0, 2], self.t[0, 0], self.t[1, 0],
                self.t[1, 2], self.t[0, 1], self.t[1, 1])


def voxelize(geoRecords, T, dtypes=[('geoRecords', object)]):

    keys = Grid.coords2keys(T, geoRecords.records().coords)
    shape = tuple(keys.max(0) + 1)

    lookUp = np.vectorize(
        lambda key: list(),
        otypes=[list])(
        np.empty(
            shape,
            dtype=list))

    # Gruppieren der keys
    df = pd.DataFrame({'indices': keys2indices(keys, shape)})
    groupDict = df.groupby(by=df.indices).groups
    keys = indices2keys(groupDict.keys(), shape)

    lookUp[keys.T.tolist()] = groupDict.values()

    # Aggregate per cell
    records = geoRecords.records()
    cells = npTools.map(lambda ids: (records[ids],), lookUp, dtypes=dtypes)

    return Grid(geoRecords.proj, cells.T, T)


def keys2indices(keys, shape):
    w = np.concatenate(
        (np.array([np.product(shape[i:]) for i in range(len(shape))])[1:], [1]))
    return (keys * w).sum(1)


def indices2keys(indices, shape):
    keys = []
    w = np.concatenate(
        (np.array([np.product(shape[i:]) for i in range(len(shape))])[1:], [1]))
    for d in w:
        keys.append(indices / d)
        indices = indices % d
    keys = np.array(keys, dtype=int).T
    return keys
