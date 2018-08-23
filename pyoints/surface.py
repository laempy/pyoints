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
    2.0
    >>> print(surface([(1, 1), (0.5, 1)]))
    [2. 1.]
    >>> print(surface([(1, 1, 2), (0.5, 1, 9)]))
    [2. 1.]

    """

    def __init__(self, coords, method=interpolate.KnnInterpolator, **kwargs):
        coords = assertion.ensure_coords(coords)
        self._interpolator = method(coords[:, :-1], coords[:, -1], **kwargs)

    def __call__(self, coords):
        return self._interpolator(coords)
