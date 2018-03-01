import numpy as np
import pandas as pd

import transformation
import nptools

from .georecords import GeoRecords
from .extent import Extent


def corners2Transform(corners, resolutions):
    # TODO revise
    # TODO nur min und maxcorner
    # TODO: als k-dimensionale Funktion in transformation

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
    # TODO revise
    # TODO: als k-dimensionale Funktion in transformation
    ext = Extent(np.concatenate([np.zeros(len(shape)), shape]))
    return Grid.keys2coords(T, ext.corners())


class Grid(GeoRecords):
    """Grid class extends GeoRecords to ease handling of matrices, like rasters
    or voxels.

    Parameters
    ----------
    proj: `Proj`
        Projection object provides the geograpic projection of the grid.
    npRecarray: `numpy.recarray`
        Multidimensional array of objects. Element of the matrix represents a
        object with k coordinate dimension.
    T: (k+1,k+1), `array_like`
        The  linear transformation matrix to transform the coordinates.
        The translation represents the origin, the rotation the 
        orientation and the scale the pixel size of the matrix.
    """

    def __new__(cls, proj, npRecarray, T):      
        assert isinstance(npRecarray,np.recarray)
        assert len(np.Recarray.shape)>1
        assert isinstance(T,np.array)
        assert np.Recarray.shape[0] == np.Recarray.shape[1]
        
        if not 'coords' in npRecarray.dtype.names:
            keys = cls.keys(npRecarray.shape)
            coords = cls.keys2coords(T, keys)
            data = nptools.add_field(npRecarray,coords,'coords')
        grid = GeoRecords(proj, data, T=T).reshape(shape).view(cls)
        return grid

    @staticmethod
    def coords2keys(T, coords):
        """Transforms coordinates to matrix indices.

        Parameters
        ----------
        T: (k+1,k+1), `array_like`
            The  linear transformation matrix to transform the coordinates.
            The translation represents the origin, the rotation the 
            orientation and the scale the pixel size of the matrix.
        coords: `numpy.recarray`
            Coordinates with at least k dimensions to convert to indices.
        
        Returns
        -------
        keys: `numpy.recarray`
            Indices of the coordinates within the matrix.
        """
        localSystem = transformation.LocalSystem(T)
        dim = localSystem.dim
        # TODO vereinfachen (wie mit strukturierten koordinaten umgehen?
        s = np.product(coords[:, :dim].shape) / dim
        flatCoords = coords[:, :dim].view().reshape((s, dim))
        values = localSystem.to_global(flatCoords)
        
        #TODO Reihenfolge der Spalten?
        # keys=np.array(np.round(values),dtype=int)[:,::-1]
        keys = np.array(np.floor(values), dtype=int)[:, ::-1]
        return keys.reshape(coords[:, :dim].shape)

    @staticmethod
    def keys2coords(T, keys):
        localSystem = transformation.LocalSystem(T)
        dim = localSystem.dim
        s = np.product(keys.shape) / dim
        flatKeys = keys.view().reshape((s, dim))[:, ::-1] + 0.5
        values = localSystem.to_local(flatKeys)
        coords = np.array(values, dtype=float)
        return coords.reshape(keys.shape)

    @staticmethod
    def coords2coords(transform, coords):
        return Grid.keys2coords(transform, Grid.coords2keys(transform, coords))

    @staticmethod
    def extentinfo(transform, shape, extent):
        # notwendig?
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

    def get_window(self, extent):
        # TODO extentinfo notwendig?
        T, cornerIndex, shape = self.extentinfo(
            self.transform, self.shape, extent)
        mask = self.keys(shape) + cornerIndex
        return self[zip(mask.T)].reshape(shape)

    def get_gdal_transform(self):
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
    cells = nptools.map(lambda ids: (records[ids],), lookUp, dtypes=dtypes)

    return Grid(geoRecords.proj, cells.T, T)

fkeys2indices
def keys2indices(keys, shape):
    # TODO stimmt mit Georecords keys ueberein
    w = np.concatenate(
        (np.array([np.product(shape[i:]) for i in range(len(shape))])[1:], [1]))
    return (keys * w).sum(1)


def indices2keys(indices, shape):
    # TODO stimmt mit Georecords indices ueberein
    keys = []
    w = np.concatenate(
        (np.array([np.product(shape[i:]) for i in range(len(shape))])[1:], [1]))
    for d in w:
        keys.append(indices / d)
        indices = indices % d
    keys = np.array(keys, dtype=int).T
    return keys
