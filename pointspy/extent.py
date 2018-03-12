import numpy as np
import itertools as it

# TODO module description

class Extent(np.recarray, object):
    """Specifies spatial extent (or bounding box) of coordinates in k 
    dimensions.
    
    Parameters
    ----------
    ext: (2*k) or (n,k), `array_like` 
        Defines spatial extent of k dimensions as either minimum corner and 
        maximum corner or as a set of n points. If a set of points is given,
        the extent is calculated based on the coordinates.
    
    Examples
    --------
    
    Basic handling of extents.
    >>> points = np.array([(0,0),(1,4),(0,1),(1,0.5)])
    >>> print points
    [[0.  0. ]
     [1.  4. ]
     [0.  1. ]
     [1.  0.5]]
    >>> ext = Extent(points)
    >>> print ext
    [0. 0. 1. 4.]
    >>> print ext.dim
    2
    >>> print ext.min_corner
    [0. 0.]
    >>> print ext.max_corner
    [1. 4.]
    >>> print ext.ranges
    [1. 4.]
    >>> print ext.corners
    [[0. 0.]
     [1. 0.]
     [1. 4.]
     [0. 4.]]
        
    """

    # __new__ to extend np.ndarray
    def __new__(cls, ext):
        
        assert hasattr(ext,'__getitem__')
        if not isinstance(ext, np.ndarray):
            ext = np.array(ext)
        
        assert len(ext.shape) <= 2
            
        if len(ext.shape) == 2:
            # points given
            min_ext = np.amin(ext, axis=0)
            max_ext = np.amax(ext, axis=0)
            ext = np.concatenate((min_ext, max_ext))            
        return ext.view(cls)

    @property
    def dim(self):
        """Number of coordinate dimensions.
        
        Returns
        -------
        dim: `int`
            Number of coordinate axes.
        """
        return len(self) / 2

    @property
    def ranges(self):
        """Provides ranges in each coordinate dimension.
        
        Returns
        -------
        ranges: (self.dim), `array_like`
            Ranges in each coordinate dimension.
        """
        return self.max_corner - self.min_corner

    @property
    def min_corner(self):
        """ Provides minimum corner of the extent.
        
        Returns
        -------
        min_corner: (self.dim), `array_like`
            Minimum coordinate values in each coordinate axis.
        """
        return self[:self.dim]

    @property
    def max_corner(self):
        """ Provides maximum corner of the extent.
        
        Returns
        -------
        max_corner: (self.dim), `array_like`
            Maximum coordinate values in each coordinate axis.
        """
        return self[self.dim:]

    @property
    def center(self):
        """ Provides center of the extent.
        
        Returns
        -------
        center: (self.dim), `array_like`
            Focal point of the extent.
        """
        return (self.max_corner + self.min_corner) * 0.5

    @property
    def split(self):
        """Splits the extent into minium and maximum corner.
        
        Returns
        -------
        min_corner: (self.dim), `array_like`
            Minimum coordinate values in each coordinate axis.
        max_corner: (self.dim), `array_like`
            Maximum coordinate values in each coordinate axis.
        """
        return self.min_corner, self.max_corner

    @property
    def corners(self):
        """Provides each corner of the extent.
        
        Returns
        -------
        corners: (2**self.dim,self.dim), `array_like`
            Corners of the extent.
            
        Two dimensional case.
        >>> ext = Extent([-1,-2,1,2])
        >>> print ext.corners
        [[-1 -2]
         [ 1 -2]
         [ 1  2]
         [-1  2]]
         
        Three dimensional case.
        >>> ext = Extent([-1,-2,-4,1,2,4])
        >>> print ext.corners
        [[-1 -2 -4]
         [ 1 -2 -4]
         [ 1  2 -4]
         [-1  2 -4]
         [-1  2  4]
         [ 1  2  4]
         [ 1 -2  4]
         [-1 -2  4]]
        """      
        
        def combgen(dim):
            # generates order of corners
            if dim == 1:
                return np.array([[0,1]],dtype=int).T
            else:
                comb = combgen(dim-1)
                col = np.array([np.hstack((
                        np.zeros(len(comb)),
                        np.ones(len(comb)),
                    ))],dtype=int).T
                comb = np.vstack((comb,comb[::-1,:]))
                return np.hstack((comb,col))

        combs = combgen(self.dim)
        combs = combs * self.dim + range(self.dim)
        return self[combs]


    def intersection(self, coords, dim=None):
        """Tests if coordinates are located within the extent.
        
        Parameters
        ----------
        coords: (n,k) or (k), `array_like`
             Represents n data points with k dimensions.
        dim: positive `int`
            Desired number of dimensions to consider.
            
        Returns
        -------
        indices: `array_like`
            Indices of coordinates, which are within the extent. If just a
            single point is given, a boolean value is returned.
            
        Examples
        --------
        
        Point within extent?
        >>> points = np.array([(1,4),(0,1),(1,0.5)])
        >>> ext = Extent(points)
        >>> print ext.intersection([(0.5,1)])
        True
        
        Points within extent?
        >>> print ext.intersection([(1,2),(-1,1),(0.5,1)])
        [0 2]

        Corners are considered to be within extent.
        >>> print ext.intersection(ext.corners)
        [0 1 2 3]

        """
        
        assert hasattr(coords,'__iter__')

        # normalize data
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        if len(coords.shape) == 1:
            coords = np.array([coords])

        # set desired dimension
        dim = self.dim if dim is None else dim
        assert dim > 0

        # check
        n, c_dim = coords.shape
        assert c_dim <= self.dim, 'Dimensions do not match.'

        min_ext, max_ext = self.split

        # Order axes by range to speed up the process (heuristic)
        order = np.argsort(self.ranges[0:dim])
        mask = np.any(
            (np.abs(
                min_ext[order]) < np.inf, np.abs(
                max_ext[order]) < np.inf), axis=0)
        axes = order[mask]

        indices = np.arange(n)
        for axis in axes:
            values = coords[indices, axis]

            # Minimum
            mask = values >= min_ext[axis]
            indices = indices[mask]
            if len(indices) == 0:
                break
            values = values[mask]

            # Maximum
            mask = values <= max_ext[axis]
            indices = indices[mask]
            if len(indices) == 0:
                break
            values = values[mask]
            
        if n == 1:
            return len(indices) == 1

        return indices
