"""Handling of grided data, like voxels or rasters.
"""

import pandas
import numpy as np

from .. import (
    assertion,
    nptools,
    projection,
    GeoRecords,
    transformation,
)

from .transformation import (
    keys_to_coords,
    coords_to_keys,
    coords_to_coords,
    extentinfo,
    keys_to_indices,
    indices_to_keys,
)    


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

    Examples
    --------

    Create a raster with projection and a transformation matrix.

    >>> from pointspy import transformation
    >>> proj = projection.Proj()
    >>> data = np.recarray((4, 3), dtype=[('values', int)])
    >>> data['values'] = np.arange(np.product(data.shape)).reshape(data.shape)
    >>> T = transformation.matrix(t=[10, 20], s=[0.5, 0.4], order='rst')
    
    >>> raster = Grid(proj, data, T)
    >>> print(raster.dtype.descr)
    [('values', '<i8'), ('coords', '<f8', (2,))]
    >>> print(raster.t.origin)
    [10. 20.]
    >>> print(raster.shape)
    (4, 3)
    >>> print(raster.dim)
    2
    
    >>> print(raster.coords)
    [[[10.25 20.2 ]
      [10.75 20.2 ]
      [11.25 20.2 ]]
    <BLANKLINE>
     [[10.25 20.6 ]
      [10.75 20.6 ]
      [11.25 20.6 ]]
    <BLANKLINE>
     [[10.25 21.  ]
      [10.75 21.  ]
      [11.25 21.  ]]
    <BLANKLINE>
     [[10.25 21.4 ]
      [10.75 21.4 ]
      [11.25 21.4 ]]]
    >>> print(raster.values)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    
    Convert coordinates to indices and reverse.
    
    >>> print(raster.coords_to_keys(raster.t.origin))
    [0 0]
    >>> print(raster.keys_to_coords([-0.5, -0.5]))
    [10. 20.]
        
    >>> print(raster.coords_to_keys(raster.coords))
    [[[0 0]
      [0 1]
      [0 2]]
    <BLANKLINE>
     [[1 0]
      [1 1]
      [1 2]]
    <BLANKLINE>
     [[2 0]
      [2 1]
      [2 2]]
    <BLANKLINE>
     [[3 0]
      [3 1]
      [3 2]]]
    >>> print(raster.keys_to_coords(raster.keys))
    [[[10.25 20.2 ]
      [10.75 20.2 ]
      [11.25 20.2 ]]
    <BLANKLINE>
     [[10.25 20.6 ]
      [10.75 20.6 ]
      [11.25 20.6 ]]
    <BLANKLINE>
     [[10.25 21.  ]
      [10.75 21.  ]
      [11.25 21.  ]]
    <BLANKLINE>
     [[10.25 21.4 ]
      [10.75 21.4 ]
      [11.25 21.4 ]]]
    
    See Also
    --------
    GeoRecords

    """
    def __new__(cls, proj, npRecarray, T):

        if not isinstance(proj, projection.Proj):
            raise TypeError("'proj' needs to be of type 'Proj'")
        if not isinstance(npRecarray, np.recarray):
            raise TypeError('numpy record array required')
        if not len(npRecarray.shape) > 1:
            raise ValueError('at least two dimensions required')
        T = assertion.ensure_tmatrix(T)

        if 'coords' not in npRecarray.dtype.names:
            keys = nptools.indices(npRecarray.shape)
            coords = keys_to_coords(T, keys)
            dtype = [('coords', float, len(npRecarray.shape))]
            data = nptools.add_fields(npRecarray, dtype, data=[coords])

        grid = GeoRecords(proj, data, T=T).reshape(npRecarray.shape).view(cls)
        return grid

    def transform(self, T):
        # overwrites super
        T = assertion.ensure_tmatrix(T, min_dim=self.dim, max_dim=self.dim)
        self.t = T * self.t
        self.coords[:] = self.keys_to_coords(self.t, self.keys())
        
    def keys_to_coords(self, keys):
        """Convert raster indices to coordinates.
        
        See Also
        --------
        poynts.transformation.keys_to_coords
        
        """
        return keys_to_coords(self.t, keys)
        
    def coords_to_keys(self, coords):
        """Convert coordinates to raster indices.
        
        See Also
        --------
        poynts.transformation.coords_to_keys
        
        """
        return coords_to_keys(self.t, coords)

    def coords_to_coords(self, coords):
        """Set coordinate to closest grid coordinate.
        
        See Also
        --------
        poynts.transformation.coords_to_coords
        
        """
        return coords_to_coords(self.t, coords)

    def get_window(self, extent):
        # TODO extentinfo notwendig?
        # M, min_corner_key, shape = self.extentinfo(self.transform, extent)
        T, cornerIndex, shape = extentinfo(self.transform, extent)
        mask = self.keys(shape) + cornerIndex
        return self[zip(mask.T)].reshape(shape)
    
    """
    def voxelize(self, geoRecords, agg_function=None):
        "Convert a point cloud to a voxel or raster.
    
        Parameters
        ----------
        geoRecords :
        
        T : array_like(Number, shape=(k+1, k+1))
            Transformation matrix in a `k` dimensional space.
        dtype : 
            
        Examples
        --------
        
        Create record array with coordinates.
        
        >>> coords = [(0, 0), (1, 0.5), (2, 2), (4, 6), (3, 2), (1, 5), (3, 0)]
        >>> rec = np.recarray(len(coords), dtype=[('coords', float, 2)])
        >>> rec['coords'] = coords
        >>> proj = projection.Proj()
        >>> geoRecords = GeoRecords(projection.Proj(), rec)
        >>> print(geoRecords)
        
        Create Grid.
        
        >>> T = transformation.matrix(s=(2.5, 3), t=[0, 0])
        >>> dtype = dtype=[('points', GeoRecords), ('count', int)]
        >>> grid = Grid(proj, np.recarray((3, 4), dtype=dtype), T)
        >>> print(grid)
        
        >>> grid = grid.voxelize(geoRecords)
        >>> print(grid)
        
        >>> print(grid.shape)
        
        >>> print(grid[0, 0])
      
        "
        if not isinstance(geoRecords, GeoRecords):
            raise TypeError("'geoRecords' need to be of type GeoRecords")

        keys = self.coords_to_keys(geoRecords.coords)
        
        shape = tuple(keys.max(0) + 1)
        
        # group keys
        df = pandas.DataFrame({'indices': keys_to_indices(keys, shape)})
        groupDict = df.groupby(by=df.indices).groups
        keys = indices_to_keys(list(groupDict.keys()), shape)
        
        lookup = np.empty(shape, dtype=list)
        lookup[keys.T.tolist()] = list(groupDict.values())
        
        # Aggregate per cell
        dtype = [type(geoRecords)]
        if agg_function is None:
            agg_function = lambda ids: geoRecords[ids]
        v = np.vectorize(agg_function, otypes=dtype)
        self[:] = v(lookup)
        return v(lookup)
    """