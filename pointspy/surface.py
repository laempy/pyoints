# BEGIN OF LICENSE NOTE
# This file is part of Pointspy.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
#
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Create surface models.
"""

from . import interpolate
from . import assertion


class Surface:
    """Creates a surface model based on representative points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions representing a surface.
    method : optional, Interpolator
        Interpolation method to use.
    \*\*kwargs : optional
        Arguments passed to the interpolation `method`.

    See Also
    --------
    poynts.interpolate.Interpolator

    Examples
    --------

    >>> method = interpolate.LinearInterpolator
    >>> surface = Surface([(0, 0, 0), (0, 2, 0), (2, 1, 4)], method=method)
    >>> print(surface([(1, 1)]))
    [2.]

    """

    def __init__(self, coords, method=interpolate.KnnInterpolator, **kwargs):
        coords = assertion.ensure_coords(coords)
        self._interpolator = method(coords[:, :-1], coords[:, -1], **kwargs)

    def __call__(self, coords):
        return self._interpolator(coords)
