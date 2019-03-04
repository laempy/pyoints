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
__version__ = '0.2.0rc2'
"""Pyoints: A Python package for point cloud, voxel and raster processing."""

from .indexkd import IndexKD
from .coords import Coords
from .extent import Extent
from .projection import Proj
from .grid import Grid
from .surface import Surface
from .georecords import (
    GeoRecords,
    LasRecords,
)
from . import (
    assertion,
    assign,
    classification,
    clustering,
    coords,
    distance,
    examples,
    extent,
    filters,
    fit,
    georecords,
    grid,
    indexkd,
    interpolate,
    misc,
    normals,
    nptools,
    polar,
    projection,
    registration,
    smoothing,
    storage,
    surface,
    transformation,
    vector,
)
from .misc import print_rounded