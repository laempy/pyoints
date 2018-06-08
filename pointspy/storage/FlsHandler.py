# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 09:59:19 2018

@author: sebastian
"""

import os
import re
import numpy as np

from .BaseGeoHandler import GeoFile


# Helper functions to extract meta data
def _meta_extract_value(string, name):
    pattern = r'(?<=Name "%s"\nValue\s)(.*?)(?=\n)' % name
    return re.search(pattern, string).group(0)


def _meta_extract_vector(string, name):
    value = _meta_extract_value(string, name)
    return np.array(value.split(" "), dtype=np.float64)


def _meta_extract_transformation(string):
    pattern = r'(?<=Transformation\s)(.*?)(?=\s\s\n)'
    value = re.search(pattern, string).group(0)
    rows = value.split("  ")
    t = []
    for row in rows:
        t.append(row.split(" "))
    return np.array(t, dtype=np.float64).T


class FlsReader(GeoFile):

    def __init__(self, filename, proj=None):
        GeoFile.__init__(self, filename, directory=True)

        with open(os.path.join(filename, 'Main')) as f:
            s = f.read()
            self.t = _meta_extract_transformation(s)

        self.proj = proj


