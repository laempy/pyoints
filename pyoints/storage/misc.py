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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Some random functions, which ease development.
"""

import numpy as np

from .. import projection
from ..georecords import GeoRecords


def create_random_GeoRecords(center=None, epsg=25832, dim=3, n=1000, scale=10):
    # Create GeoRecords from scratch (for examples)
    dtype = [
        ('coords', np.float, dim),
        ('intensity', np.uint),
        ('classification', np.uint),
        ('values', np.float),
        ('keypoint', np.bool),
        ('synthetic', np.bool),
        ('withheld', np.bool)
    ]
    records = np.recarray(n, dtype=dtype)

    records['coords'] = np.random.rand(n, dim) * scale
    records['intensity'] = np.random.rand(n) * 255
    records['classification'] = 2
    records['classification'][records.coords[:, 2] > 0.1 * scale] = 3
    records['classification'][records.coords[:, 2] > 0.3 * scale] = 4
    records['classification'][records.coords[:, 2] > 0.5 * scale] = 5
    records['synthetic'][:4] = False
    records['synthetic'][1] = True
    records['keypoint'][:4] = False
    records['keypoint'][2] = True
    records['withheld'][:4] = False
    records['withheld'][3] = True
    
    records['values'] = np.arange(n)

    if center is not None:
        records['coords'] = records['coords'] + center

    proj = projection.Proj.from_epsg(epsg)

    return GeoRecords(proj, records)
