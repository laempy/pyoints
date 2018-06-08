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
    >>> coords = Extent([np.zeros(2), np.ones(2)]).corners
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
    pts = Extent([np.zeros(dim), np.ones(dim)]).corners

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
    return transformation.transform(ext.corners, T)


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


    See Also
    --------
    GeoRecords

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
            dtype = [('coords', float, coords.shape[-1])]
            data = nptools.add_fields(npRecarray, dtype, data=coords)
        grid = GeoRecords(proj, data, T=T).reshape(npRecarray.shape).view(cls)
        return grid

    @staticmethod
    def coords2keys(T, coords):
        """Transforms coordinates to matrix indices.

        Parameters
        ----------
        T : array_like(Number, shape=(k+1,k+1))
            A linear transformation matrix to transform the coordinates.
            The translation represents the origin, the rotation the
            orientation and the scale the pixel size of a raster.
        coords : array_like(Number, shape=(n, k))
            Coordinates with at least `k` dimensions to convert to indices.

        Returns
        -------
        keys : np.ndarray(int, shape=(n, k))
            Indices of the coordinates within the grid.

        """
        T = transformation.LocalSystem(T)
        dim = T.dim
        # TODO vereinfachen (wie mit strukturierten koordinaten umgehen?
        s = np.product(coords[:, :dim].shape) / dim
        flat_coords = coords[:, :dim].view().reshape((s, dim))
        values = T.to_global(flat_coords)

        # TODO Reihenfolge der Spalten?
        # keys=np.array(np.round(values),dtype=int)[:,::-1]
        keys = np.array(np.floor(values), dtype=int)[:, ::-1]
        return keys.reshape(coords[:, :dim].shape)

    @staticmethod
    def keys2coords(T, keys):
        """Converts indices of raster cells to coordinates.

        Parameters
        ----------
        T : array_like(Number, shape=(k+1,k+1))
            The transformation matrix of a `k` dimensional raster.
        keys : array_like(int, shape=(n, k))
            Indices of `n` raster cells.

        Returns
        -------
        coords : array_like(Number, shape=(n, k))
            Desired coordinates of the raster cells.

        See Also
        --------
        Grid.coords2keys, Grid.coords2coords

        """
        keys = assertion.ensure_coords(keys)
        T = transformation.LocalSystem(T)
        if not keys.shape[1] == T.shape[1] - 1:
            raise ValueError('dimensions do not match')

        s = np.product(keys.shape) / T.dim
        flatKeys = keys.view().reshape((s, T.dim))[:, ::-1] + 0.5
        values = T.to_local(flatKeys)
        coords = np.array(values, dtype=float)
        return coords.reshape(keys.shape)

    @staticmethod
    def coords2coords(T, coords):
        """Aligns coordinates with a raster grid.

        Parameters
        ----------
        T : array_like(Number, shape=(k+1, k+1))
            The transformation matrix of a `k` dimensional raster.
        coords : array_like(Number, shape=(n, k))
            Coordinates to align with a raster grid.

        Returns
        -------
        coords : array_like(Number, shape=(n, k))
            Desired coordinates aligned with the grid.

        See Also
        --------
        Grid.coords2keys, Grid.keys2coords

        """
        return Grid.keys2coords(T, Grid.coords2keys(T, coords))

    @staticmethod
    def extentinfo(T, extent):
        """Recieve information on a raster subset with given boundaries.

        Parameters
        ----------
        T : array_like(Number, shape=(k+1,k+1))
            The transformation matrix of a `k` dimensional raster.
        extent : array_like(Number, shape=(2 * k))
            Desired extent.

        Returns
        -------
        T : array_like(Number, shape=(k+1,k+1))
            Extent of the original raster.
        origin_key : array_like(int, shape=(k))
            Key or index of the origin of the new transformation matrix.
        shape : np.array(int, shape=(k))
            Shape of the raster subset.

        Examples
        --------

        >>> T = transformation.matrix(t=[100, 200], s=[2, -2])
        >>> ext = Extent([150, 250, 200, 300])
        >>> M, min_corner_key, shape = Grid.extentinfo(T, ext)
        >>> print(M)
        [[  2.   0. 150.]
         [  0.  -2. 300.]
         [  0.   0.   1.]]
        >>> print(min_corner_key)
        [-50  25]
        >>> print(shape)
        [26 26]

        """
        # ensure extent
        extent = Extent(extent)
        T = assertion.ensure_tmatrix(T)

        dim = len(extent) / 2

        if not T.shape[0] - 1 == dim:
            raise ValueError('dimensions do not match')

        corner_keys = Grid.coords2keys(T, extent.corners)

        shape = np.ptp(corner_keys, axis=0) + 1

        # Minimum corner
        idx = np.argmin(corner_keys, axis=0)
        origin_key = corner_keys[idx, range(dim)]
        min_corner = Grid.keys2coords(T, [origin_key - 0.5])[0, :]

        # define transformation
        t = np.copy(T)
        t[:-1, dim] = min_corner

        return t, origin_key, shape

    def get_window(self, extent):
        # TODO extentinfo notwendig?
        T, cornerIndex, shape = self.extentinfo(self.transform, extent)
        mask = self.keys(shape) + cornerIndex
        return self[zip(mask.T)].reshape(shape)


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
