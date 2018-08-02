#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:00:33 2018

@author: sebastian
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
