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
    coords : (n,k), array_like
        Represents `n` data points of `k` dimensions.
    values : (n), array_like
        Values to interpolate.

    Attributes
    ----------
    coords : (n,k), array_like
        Represents `n` data points of `k` dimensions.
    dim : positive int
        Number of coordinate dimensions.

    """

    def __init__(self, coords, values):
        self._coords = assertion.ensure_coords(coords)
        assert hasattr(values, '__len__')
        assert len(self._coords) == len(values)
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

    Examples
    --------
    TODO

    See Also
    --------
    Interpolator

    """

    def __init__(self, coords, values, k=None, maxDist=None):
        Interpolator.__init__(self, coords, values)
        if k is None:
            k = self.dim + 1
        if maxDist is None:
            weightFunction = 'distance'
        else:
            def weightFunction(dists):
                w = np.zeros(dists.shape)
                zeroMask = dists == 0

                w[~zeroMask] = 1.0 / dists[~zeroMask]
                w[dists > maxDist] = 0

                w[np.any(zeroMask, axis=1), :] = 0
                w[zeroMask] = 1
                return w

        # self._interpolator=NearestNDInterpolator(coords,values)
        self._interpolator = KNeighborsRegressor(
            n_neighbors=k,
            weights=weightFunction
        )
        self._interpolator.fit(coords, values)

    def _interpolate(self, coords):
        pred = self._interpolator.predict(coords)
        return pred


class PolynomInterpolator(Interpolator):
    """Polynom interpolation.

    Parameters
    ----------
    TODO

    Examples
    --------
    TODO

    See Also
    --------
    Interpolator

    """

    def __init__(
            self,
            coords,
            values,
            deg=2,
            weights=None,
            interaction_only=False):
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