# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University,
# lamprecht@uni-trier.de
#
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""Data structures to handle multi-dimensional point data.
"""

import numpy as np
from .indexkd import IndexKD
from .extent import Extent
from . import (
    assertion,
    transformation,
)

from .misc import print_rounded


class Coords(np.ndarray, object):
    """Class to represent coordinates. It provides a spatial index to
    conveniently optimize spatial neighborhood queries. The spatial index is
    calculated on demand only and deleted, if coordinates have been changed.

    Parameters
    ----------
    coords : array_like(Number)
        Represents `n` data points of `k` dimensions in a Cartesian coordinate
        system. Any desired shape of at least length two is allowed to enable
        the representation point, voxel or raster data. The last shape element
        always represents the coordinate dimension.

    Attributes
    ----------
    dim : positive int
        Number of coordinate dimensions.
    flattened : array_like(Number, shape=(n, k))
        List representation of the coordinates.

    Examples
    --------

    Create some 2D points.

    >>> coords = Coords([(0, 1), (2, 1), (2, 3), (0, -1)])
    >>> print_rounded(coords)
    [[ 0  1]
     [ 2  1]
     [ 2  3]
     [ 0 -1]]

    Get Extent.

    >>> print_rounded(coords.extent())
    [ 0 -1  2  3]
    >>> print_rounded(coords.extent(dim=1))
    [0 2]

    Use IndexKD, which updates automatically, if a coordinate has changed.

    >>> coords.indexKD().ball([0, 0], 2)
    [0, 3]
    >>> coords[1, 0] = -1
    >>> coords.indexKD().ball([0, 0], 2)
    [0, 1, 3]

    """
    def __new__(cls, coords):
        return assertion.ensure_numarray(coords).view(cls)

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self._clear_cache()

    def __iter__(self):
        return iter(self.view(np.ndarray))

    def _clear_cache(self):
        if hasattr(self, '_indices'):
            del self._indices
        if hasattr(self, '_extents'):
            del self._extents

    @property
    def dim(self):
        return self.shape[-1]

    @property
    def flattened(self):
        if len(self.shape) == 2:
            return self
        else:
            s = (np.product(self.shape[:self.dim]), self.dim)
            return self.reshape(s)

    def transform(self, T):
        """Transform coordinates.

        Parameters
        ----------
        T : array_like(Number, shape=(self.dim+1, self.dim+1))
            Transformation matrix to apply.

        Returns
        -------
        Coords(shape=self.shape)
            transformed coords.

        Examples
        --------

        Transform structured coordinates.

        >>> coords = Coords(
        ...             [[(2, 3), (2, 4), (3, 2)], [(0, 0), (3, 5), (9, 4)]])
        >>> print_rounded(coords)
        [[[2 3]
          [2 4]
          [3 2]]
        <BLANKLINE>
         [[0 0]
          [3 5]
          [9 4]]]

        >>> T = transformation.matrix(t=[10, 20], s=[0.5, 1])
        >>> tcoords = coords.transform(T)
        >>> print_rounded(tcoords)
        [[[ 11.   23. ]
          [ 11.   24. ]
          [ 11.5  22. ]]
        <BLANKLINE>
         [[ 10.   20. ]
          [ 11.5  25. ]
          [ 14.5  24. ]]]

        """
        T = assertion.ensure_tmatrix(T, dim=self.dim)
        tcoords = transformation.transform(self.flattened, T)
        return tcoords.reshape(self.shape).view(Coords)

    def indexKD(self, dim=None):
        """Get a spatial index of the coordinates.

        Parameters
        ----------
        dim : optional, positive int
            Desired dimension of the spatial index. If None, the all coordinate
            dimensions are used.

        Returns
        -------
        IndexKD
            Spatial index of the coordinates of desired dimension.

        Notes
        -----
        The spatial indices are generated on demand and are cached
        automatically. Setting new coordinates clears the cache.

        See Also
        --------
        poynts.IndexKD

        Examples
        --------

        >>> coords = Coords([(2, 3, 1), (3, 2, 3), (0, 1, 0), (9, 5, 4)])
        >>> print_rounded(coords.indexKD().dim)
        3
        >>> print_rounded(coords.indexKD(dim=2).dim)
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
            indexKD = IndexKD(self.flattened[:, :dim], copy=False)
            self._indices[dim] = indexKD
        return indexKD

    def extent(self, dim=None):
        """Provides the spatial extent of the coordinates.

        Parameters
        ----------
        dim : optional, positive int
            Define how many coordinate dimensions to use. If None, all
            dimensions are used.

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
        >>> print_rounded(coords.extent())
        [-1.  1.  9.  5.]
        >>> print_rounded(coords.extent(dim=1))
        [-1.  9.]
        >>> print_rounded(coords.indexKD().ball([0, 0], 2))
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
            ext = Extent(self.flattened[:, :dim])
            self._extents[dim] = ext
        return ext
