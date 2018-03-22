# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 17:49:08 2018

@author: sebastian
"""

import numpy as np

from . import interpolate
from . import assertion


class Surface:
    """Creates a surface model based on points representing the surface.

    Parameters
    ----------
    coords : (n,k), array_like
        Represents `n` data points of `k` dimensions representing a surface.
    method : optional, `Interpolator`
        Interpolation method to use.
    **kwargs : optional
        Arguments passed to the interpolation `method`.

    See Also
    --------
    interpolate.Interpolator

    Examples
    --------

    >>> method = interpolate.LinearInterpolator
    >>> surface = Surface([(0,0,0),(0,2,0),(2,1,4)], method=method)
    >>> print surface([(1,1)])
    [2.]


    """
    def __init__(self, coords, method=interpolate.KnnInterpolator, **kwargs):
        coords = assertion.ensure_coords(coords)
        self._interpolator = method(coords[:, :-1], coords[:, -1], **kwargs)

    def __call__(self, coords):
        return self._interpolator(coords)