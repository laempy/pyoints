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
"""Spatial interpolation methods.
"""

import numpy as np
from numbers import Number

from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from . import assertion


class Interpolator:
    """Abstract generic multidimensional interpolation class.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    values : array_like(Number, shape=(n)) or array_like(Number, shape=(n, m))
        One dimensional or `m` dimensional values to interpolate.

    Attributes
    ----------
    coords : np.ndarray(Number, shape=(n, k))
        Provided coordinates.
    dim : positive int
        Number of coordinate dimensions.

    """

    def __init__(self, coords, values):
        self._coords = assertion.ensure_coords(coords)
        values = assertion.ensure_numarray(values)
        if not len(self._coords) == len(values):
            raise ValueError("array dimensions do not fit")
        self._shift = self._coords.min(0)
        self._dim = len(self._shift)

    def _interpolate(self, coords):
        raise NotImplementedError()

    def __call__(self, coords):
        """Apply interpolation.

        Parameters
        ----------
        coords : array_like(Number, shape=(m, self.dim))
            Represents `m` points to interpolate.

        Returns
        -------
        np.ndarray(Number, shape=(m))
            Interpolated values.

        """
        coords = assertion.ensure_numarray(coords)
        is_point = len(coords.shape) == 1
        if is_point:
            coords = [coords]
        coords = assertion.ensure_coords(coords)
        values = self._interpolate(coords[:, :self.dim])
        if is_point:
            values = values[0]
        return values

    @property
    def coords(self):
        return self._coords

    @property
    def dim(self):
        return self._dim


class LinearInterpolator(Interpolator):
    """Linear interpolation using Delaunay triangilation.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    values : array_like(Number, shape=(n)) or array_like(Number, shape=(n, m))
        One dimensional or `m` dimensional values to interpolate.
        
    See Also
    --------
    Interpolator, scipy.interpolate.LinearNDInterpolator

    Examples
    --------

    Interpolation of one-dimensional values.

    >>> coords = [(0, 0), (0, 2), (2, 1)]
    >>> values = [0, 3, 6]

    >>> interpolator = LinearInterpolator(coords, values)
    
    >>> print(interpolator([0, 1]))
    1.5
    >>> print(interpolator([(0, 1), (0, 0), (0, -1)]))
    [1.5 0.  nan]

    Interpolation of multi-dimensional values.

    >>> coords = [(0, 0), (0, 2), (2, 3)]
    >>> values = [(0, 1, 3), (2, 3, 8), (6, 4, 4)]

    >>> interpolator = LinearInterpolator(coords, values)
    
    >>> print(interpolator([0, 1]))
    [1.  2.  5.5]
    >>> print(interpolator([(0, 1), (0, 0), (0, -1)]))
    [[1.  2.  5.5]
     [0.  1.  3. ]
     [nan nan nan]]

    """

    def __init__(self, coords, values):
        Interpolator.__init__(self, coords, values)
        self._interpolator = LinearNDInterpolator(coords, values, rescale=True)

    def _interpolate(self, coords):
        return self._interpolator(coords)


class KnnInterpolator(Interpolator):
    """Nearest neighbour interpolation.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    values : array_like(Number, shape=(n)) or array_like(Number, shape=(n, m))
        One dimensional or `m` dimensional values to interpolate.
    k : optional, positive int
        Number of neighbours used for interpolation.
    max_dist : optional, positive Number
        Maximum distance of a neigbouring point to be used for interpolation.

    See Also
    --------
    Interpolator, sklearn.neighbors.KNeighborsRegressor


    Examples
    --------

    Interpolation of one-dimensional values.

    >>> coords = [(0, 0), (0, 2), (2, 1)]
    >>> values = [0, 3, 6]

    >>> interpolator = KnnInterpolator(coords, values, k=2, max_dist=1)
    
    >>> print(interpolator([0, 1]))
    1.5
    >>> print(interpolator([(0, 1), (0, 0), (0, -1)]))
    [1.5 0.  0. ]

    Interpolation of multi-dimensional values.

    >>> coords = [(0, 0), (0, 2), (2, 3)]
    >>> values = [(0, 1, 3), (2, 3, 8), (6, 4, 4)]

    >>> interpolator = KnnInterpolator(coords, values, k=2)
    
    >>> print(interpolator([0, 1]))
    [1.  2.  5.5]
    >>> print(interpolator([(0, 1), (0, 0), (0, -1)]))
    [[1.   2.   5.5 ]
     [0.   1.   3.  ]
     [0.5  1.5  4.25]]

    """

    def __init__(self, coords, values, k=None, max_dist=None):
        Interpolator.__init__(self, coords, values)

        if k is None:
            k = self.dim + 1
        else:
            if not (isinstance(k, int) and k > 0):
                raise ValueError("'k' needs to be an integer greater 0")

        if max_dist is None:
            weight_function = 'distance'
        else:
            if not (isinstance(max_dist, Number) and max_dist > 0):
                raise ValueError("'max_dist' needs to be a number greater 0")

            def weight_function(dists):
                w = np.zeros(dists.shape)
                zeroMask = dists == 0

                w[~zeroMask] = 1.0 / dists[~zeroMask]
                w[dists > max_dist] = 0

                w[np.any(zeroMask, axis=1), :] = 0
                w[zeroMask] = 1
                return w

        self._interpolator = KNeighborsRegressor(
            n_neighbors=k,
            weights=weight_function,
        )
        self._interpolator.fit(coords, values)

    def _interpolate(self, coords):
        pred = self._interpolator.predict(coords)
        return pred


class PolynomInterpolator(Interpolator):
    """Polynomial interpolation.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    values : array_like(Number, shape=(n)) or array_like(Number, shape=(n, m))
        One dimensional or `m` dimensional values to interpolate.
    deg : optional, positive int
        The degree of the polynomial features.
    weights : optional, array_like(shape=(n))
        Weights for each sample.
    interaction_only : optional, bool
        Indicates whether or not to calculate interaction only.

    See Also
    --------
    Interpolator, sklearn.preprocessing.PolynomialFeatures,
    sklearn.linear_model.LinearRegression

    Examples
    --------

    Interpolation of one-dimensional values.

    >>> coords = [(0, 0), (0, 2), (2, 1)]
    >>> values = [0, 3, 6]

    >>> interpolator = PolynomInterpolator(coords, values, deg=1)
    
    >>> print(interpolator([0, 1]))
    1.5
    >>> print(interpolator([(0, 1), (0, 0), (0, -1)]))
    [ 1.5  0.  -1.5]

    Interpolation of multi-dimensional values.

    >>> coords = [(0, 0), (0, 2), (2, 3)]
    >>> values = [(0, 1, 3), (2, 3, 8), (6, 4, 4)]

    >>> interpolator = PolynomInterpolator(coords, values, deg=0)
    
    >>> print(np.round(interpolator([0, 1]), 2))
    [2.67 2.67 5.  ]
    >>> print(np.round(interpolator([(0, 1), (0, 0), (0, -1)]), 2))
    [[2.67 2.67 5.  ]
     [2.67 2.67 5.  ]
     [2.67 2.67 5.  ]]

    """

    def __init__(
            self,
            coords,
            values,
            deg=2,
            weights=None,
            interaction_only=False):

        Interpolator.__init__(self, coords, values)
        if not (isinstance(deg, int)):
            raise ValueError("'deg' needs to be an integer")

        self._deg = deg
        self._interaction_only = interaction_only
        self._interpolator = LinearRegression()
        self._interpolator.fit(
            self._prepare(coords),
            values,
            sample_weight=weights
        )

    def _interpolate(self, coords):
        pred = self._interpolator.predict(self._prepare(coords))
        return pred

    def _prepare(self, coords):
        return PolynomialFeatures(
            self._deg,
            interaction_only=self._interaction_only
        ).fit_transform(coords)

    @property
    def coef(self):
        return self._interpolator.coef_
