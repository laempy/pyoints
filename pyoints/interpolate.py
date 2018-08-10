# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
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
    values : array_like(Number, shape=(n))
        Values to interpolate.

    Attributes
    ----------
    coords : np.ndarray(Number, shape=(n, k))
        Provided coordinates.
    dim : positive int
        Number of coordinate dimensions.

    """

    def __init__(self, coords, values):
        self._coords = assertion.ensure_coords(coords)
        values = assertion.ensure_numvector(values)
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
        coords = assertion.ensure_coords(coords)
        return self._interpolate(coords[:, :self.dim])

    @property
    def coords(self):
        return self._coords

    @property
    def dim(self):
        return self._dim


class LinearInterpolator(Interpolator):
    """Linear interpolation.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    values : array_like(Number, shape=(n))
        Values to interpolate.

    See Also
    --------
    Interpolator

    Examples
    --------

    >>> coords = [(0, 0), (0, 2), (2, 1)]
    >>> values = [0, 3, 6]

    >>> interpolator = LinearInterpolator(coords, values)
    >>> print(interpolator([(1, 1)]))
    [3.75]
    >>> print(interpolator([(-1, -1)]))
    [nan]

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
    values : array_like(Number, shape=(n))
        Values to interpolate.
    k : optional, positive int
        Number of neighbours used for interpolation.
    max_dist : optional, positive Number
        Maximum distance of a neigbouring point to be used for interpolation.

    See Also
    --------
    Interpolator

    Examples
    --------

    >>> coords = [(0, 0), (0, 2), (2, 1)]
    >>> values = [0, 3, 6]

    >>> interpolator = KnnInterpolator(coords, values, k=2, max_dist=1)
    >>> print(interpolator([(1, 1)]))
    [6.]
    >>> print(interpolator([(-0.5, -0.5)]))
    [0.]

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
    """Polynom interpolation.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    values : array_like(Number, shape=(n))
        Values to interpolate.
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

    >>> coords = [(0, 0), (0, 2), (2, 1)]
    >>> values = [0, 3, 6]

    >>> interpolator = PolynomInterpolator(coords, values, deg=1)
    >>> print(interpolator([(1, 1)]))
    [3.75]
    >>> print(interpolator([(-1, -0.5)]))
    [-3.]

    >>> interpolator = PolynomInterpolator(coords, values, deg=0)
    >>> print(interpolator([(1, 1)]))
    [3.]
    >>> print(interpolator([(-1, -0.5)]))
    [3.]


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