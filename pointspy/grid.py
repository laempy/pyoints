import numpy as np
import pandas as pd

from . import (
    assertion,
    transformation,
    registration,
    nptools,
    projection,
    GeoRecords,
    Extent,
)


def corners2transform(corners, scale=None):
    # TODO revise
    # TODO nur min und maxcorner
    # TODO: als k-dimensionale Funktion in transformation
    # TODO rename
    """Create a transformation matrix based on the corners of a raster.

    Parameters
    ----------
    corners : array_like(Number, shape=(2**k, k))
        Corners of a `k` dimensional grid in a `k` dimensional space.
    scale : optional, array_like(Number, shape=(k))
        Optional scale to define the pixel resolution of a raster.

    Examples
    --------

    Create some corners

    >>> T = transformation.matrix(t=[3, 5], s=[10, 20], r=np.pi/2)
    >>> coords = Extent([np.zeros(2), np.ones(2)]).corners()
    >>> corners = transformation.transform(coords, T)

    Create transformation matrix without scale.

    >>> M = corners2transform(corners)
    >>> print(np.round(M, 3))
    [[ 0. -1.  3.]
     [ 1.  0.  5.]
     [ 0.  0.  1.]]

    Create transformation matrix with a scale.

    >>> M = corners2transform(corners, [0.5, 2])
    >>> print(np.round(M, 3))
    [[ 0.  -2.   3. ]
     [ 0.5  0.   5. ]
     [ 0.   0.   1. ]]

    """

    corners = assertion.ensure_coords(corners)
    dim = corners.shape[1]
    pts = Extent([np.zeros(dim), np.ones(dim)]).corners()

    # find transformation matrix
    T = registration.find_transformation(corners, pts)

    # get translation, rotation and scale
    t, r, s, det = transformation.decomposition(T)

    return transformation.matrix(t=t, r=r, s=scale)
    #T = registration.find_transformation(pts, corners)

    #return T
    #return T
    #print np.round(T, 6)
    # TODO scaling missing
    T = registration.find_rototranslation(corners, pts)

    #print np.round(T, 6)
    return T
    #print transformation.decomposition(T)

    # TODO 3D

    #get_rototranslation()

    # TODO create matrix based on LGS
    #corners / corners.max(0)
    #np.linalg.solve()

    sT = transformation.s_matrix(scale)
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
    """Generates the corners of a grid based on a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix in a `k` dimensional space.
    shape : array_like(int, shape=(k))
        Desired shape of the grid


    Examples
    --------

    >>> T = transformation.matrix(t=[10, 20], s=[0.5, 2])
    >>> print(np.round(T, 3))
    [[ 0.5  0.  10. ]
     [ 0.   2.  20. ]
     [ 0.   0.   1. ]]

    >>> corners = transform2corners(T, (100, 200))
    >>> print(np.round(corners, 3))
    [[ 10.  20.]
     [ 60.  20.]
     [ 60. 420.]
     [ 10. 420.]]

    """
    ext = Extent([np.zeros(len(shape)), shape])
    return transformation.transform(ext.corners(), T)


class Grid(GeoRecords):
    """Grid class extends GeoRecords to ease handling of matrices, like rasters
    or voxels.

    Parameters
    ----------
    proj : pointspy.projection.Proj
        Projection object provides the geograpic projection of the grid.
    npRecarray : `numpy.recarray`
        Multidimensional array of objects. Element of the matrix represents a
        object with k coordinate dimension.
    T : array_like(Number, shape=(k+1, k+1))
        The  linear transformation matrix to transform the coordinates.
        The translation represents the origin, the rotation the
        orientation and the scale the pixel size of the matrix.

    TODO: Examples
    Examples
    --------

    """
    def __new__(cls, proj, npRecarray, T):

        if not isinstance(proj, projection.Proj):
            raise ValueError('"proj" needs to be an instance of Proj')
        if not isinstance(npRecarray, np.recarray):
            raise ValueError('numpy record array required')
        if not len(npRecarray.shape) > 1:
            raise ValueError('at two dimensions required')
        T = assertion.ensure_tmatrix(T)

        if 'coords' not in npRecarray.dtype.names:
            keys = cls.keys(npRecarray.shape)
            coords = cls.keys2coords(T, keys)
            data = nptools.add_fields(npRecarray, [('coords', float, coords.shape[-1])], data=coords)
        grid = GeoRecords(proj, data, T=T).reshape(npRecarray.shape).view(cls)
        return grid

    @staticmethod
    def coords2keys(T, coords):
        """Transforms coordinates to matrix indices.

        Parameters
        ----------
        T : array_like(Number, shape=(k+1,k+1))
            The  linear transformation matrix to transform the coordinates.
            The translation represents the origin, the rotation the
            orientation and the scale the pixel size of the matrix.
        coords : `numpy.recarray`
            Coordinates with at least `k` dimensions to convert to indices.

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

        # TODO Reihenfolge der Spalten?
        # keys=np.array(np.round(values),dtype=int)[:,::-1]
        keys = np.array(np.floor(values), dtype=int)[:, ::-1]
        return keys.reshape(coords[:, :dim].shape)

    @staticmethod
    def keys2coords(T, keys):
        # TODO check
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
        minCornerIndex = np.array(
            [cornerIndices[idx, i] for i, idx in enumerate(minCornerIdx)],
            dtype=int
        )
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
