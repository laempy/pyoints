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
# END OF LICENSE NOTE
__version__ = '0.1'
"""Pyoints: A Python package for point cloud, voxel and raster processing."""

from .indexkd import IndexKD
from .coords import Coords
from .extent import Extent
from .projection import Proj
from .georecords import GeoRecords
from . import (
    grid,
    registration,
    storage,
    assertion,
    assign,
    classification,
    clustering,
    coords,
    distance,
    extent,
    filters,
    fit,
    georecords,
    indexkd,
    interpolate,
    misc,
    nptools,
    polar,
    projection,
    smoothing,
    surface,
    transformation,
    vector,
    examples,
)
