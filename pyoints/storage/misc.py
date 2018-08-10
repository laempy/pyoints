# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
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
