import numpy as np

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
        Represents `n` data points of `k` dimensions.
    dim : positive int
        Number of coordinate dimensions.

    """

    def __init__(self, coords, values):
        self._coords = assertion.ensure_coords(coords)
        assertion.ensure_numvector(values)
        if not len(self._coords) == len(values):
            raise ValueError("Array dimensions do not fit")
        self._shift = self._coords.min(0)
        self._dim = len(self._shift)

    def _interpolate(self, coords):
        raise NotImplementedError()

    def __call__(self, coords):
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

    Examples
    --------
    TODO

    See Also
    --------
    Interpolator

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
    k : optional, positive int
        Number of neighbours used for interpolation.
    max_dist : optional, positive float
        Maximum distance of a neigbouring point to be used for interpolation.

    Examples
    --------
    TODO

    See Also
    --------
    Interpolator

    """

    def __init__(self, coords, values, k=None, max_dist=None):
        # TODO assetion
        Interpolator.__init__(self, coords, values)
        if k is None:
            k = self.dim + 1
        if max_dist is None:
            weight_function = 'distance'
        else:
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
    deg : optional, positive int
        TODO
    weights : optional, TODO
        TODO
    interaction_only : optional, bool
        TODO

    See Also
    --------
    Interpolator

    Examples
    --------
    TODO

    """

    def __init__(
            self,
            coords,
            values,
            deg=2,
            weights=None,
            interaction_only=False):
        # TODO assertion
        Interpolator.__init__(self, coords, values)
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
