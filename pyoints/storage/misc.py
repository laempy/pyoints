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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Some random funtions, which ease development.
"""

import numpy as np

from . import (
    projection,
    GeoRecords
)


def create_random_GeoRecords(center=None, epsg=25832, dim=3, n=1000, scale=10):
    # Create GeoRecords from scratch (for examples)
    dtype = [
        ('coords', float, dim),
        ('intensity', int),
        ('classification', int),
        ('values', float)
    ]
    records = np.recarray(n, dtype=dtype)

    records['coords'] = np.random.rand(n, dim) * scale
    records['intensity'] = np.random.rand(n) * 255
    records['classification'] = records['intensity'] < 40
    records['values'] = np.arange(n)

    if center is not None:
        records['coords'] = records['coords'] + center

    proj = projection.Proj.from_epsg(epsg)

    return GeoRecords(proj, records)
