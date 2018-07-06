# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 10:26:23 2018

@author: sebastian
"""

import numpy as np
from .indexkd import IndexKD
from .extent import Extent
from . import assertion


class Coords(np.ndarray, object):
    """Class to represent point coordinates. It provides a spatial index to
    convieniantly optimize performance.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system.

    Attributes
    ----------
    dim : positive int
        Number of coordinate dimensions of the `coords` field.

    Examples
    --------

    >>> coords = Coords([(0, 1), (2, 1), (2, 3), (0, -1)])
    >>> print(coords)
    [[ 0  1]
     [ 2  1]
     [ 2  3]
     [ 0 -1]]
    >>> print([coord for coord in coords])
    [array([0, 1]), array([2, 1]), array([2, 3]), array([ 0, -1])]

    Get Extent.
    >>> print(coords.extent())
    [ 0 -1  2  3]
    >>> print(coords.extent(dim=1))
    [0 2]

    Use IndexKD, which updates automatically.

    >>> coords.indexKD().ball([0, 0], 2)
    [0, 3]
    >>> coords[1, 0] = -1
    >>> coords.indexKD().ball([0, 0], 2)
    [0, 1, 3]

    """
    def __init__(self, coords, by_col=False):
        pass

    def __new__(cls, coords, by_col=False):
        return assertion.ensure_coords(coords, by_col=by_col).view(cls)

    @property
    def dim(self):
        return self.shape[1]

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self._clear_cache()

    def __iter__(self):
        return iter(self.view(np.ndarray))

    def _clear_cache(self):
        # Deletes cached data.
        print('COORDS _clear_cache')
        if hasattr(self, '_indices'):
            del self._indices
        if hasattr(self, '_extents'):
            del self._extents

    def indexKD(self, dim=None):
        """Spatial index of the coordinates.

        Parameters
        ----------
        dim : optional, positive int
            Desired dimension of the spatial index. If None the all coordinate
            dimensions are used.

        Returns
        -------
        IndexKD
            Spatial index of the coordinates of desired dimension.

        Notes
        -----
        The spatial indices are calculated on demand and are cached
        automatically. Setting new coordinates clears the cache.

        See Also
        --------
        IndexKD

        Examples
        --------

        >>> coords = Coords([(2, 3, 1), (3, 2, 3), (0, 1, 0), (9, 5, 4)])
        >>> print(coords.indexKD().dim)
        3
        >>> print(coords.indexKD(dim=2).dim)
        2

        """
        if dim is None:
            dim = self.dim
        elif not (isinstance(dim, int) and dim > 0 and dim <= self.dim):
            m = "'dim' needs to be an integer in range [1, %i]" % self.dim
            raise ValueError(m)

        # use cache for performance reasons
        if not hasattr(self, '_indices'):
            self._indices = {}
        indexKD = self._indices.get(dim)
        if indexKD is None:
            indexKD = IndexKD(self[:, :dim], copy=False)
            self._indices[dim] = indexKD
        return indexKD

    def extent(self, dim=None):
        """Provides the spatial extent of the coordinates.

        Parameters
        ----------
        dim : optional, positive int
            Define which coordinates to use. If not given all dimensions are
            used.

        Returns
        -------
        extent : Extent
            Spatial extent of the coordinates.

        Notes
        -----
        The extents are calculated on demand and are cached automatically.
        Setting new coordinates clears the cache.

        See Also
        --------
        Extent

        Examples
        --------

        >>> coords = Coords([(2, 3), (3, 2), (0, 1), (-1, 2.2), (9, 5)])
        >>> print(coords.extent())
        [-1.  1.  9.  5.]
        >>> print(coords.extent(dim=1))
        [-1.  9.]
        >>> print(coords.indexKD().ball([0, 0], 2))
        [2]

        """
        if dim is None:
            dim = self.dim
        elif not (isinstance(dim, int) and dim > 0 and dim <= self.dim):
            m = "'dim' needs to be an integer in range [1, %i]" % self.dim
            raise ValueError(m)

        # use cache for performance reasons
        if not hasattr(self, '_extents'):
            self._extents = {}
        ext = self._extents.get(dim)
        if ext is None:
            ext = Extent(self[:, :dim])
            self._extents[dim] = ext
        return ext
