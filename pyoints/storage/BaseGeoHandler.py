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
"""Basic handling of spatial files.
"""

import os
from .. import (
    assertion,
    projection
)
import warnings
from datetime import datetime


class GeoFile:
    """Interface to read files containing spatial information.

    Parameters
    ----------
    infile : String
        File to be read. It should contain data to be interpreted as points of
        `k` dimensions.
    directory : bool
        Indicates if the file is a composite of several files stored in a
        directory.

    Properties
    ----------
    t : np.matrix(Number, shape=(k+1, k+1))
        Transformation matrix to transform the `k`-dimensional points. Usually
        this matrix defines the origin of a local coordinate system.
    proj : Proj
        Coordinate projection system.
    extent : Extent(Number, shape=(2 * k))
        Defines the spatial extent of the points.
    corners : np.ndarray(Number, shape=(2**k, k))
        Corners of the extent.
    date : datetime
        Date of capture.

    See Also
    --------
    Proj, Extent

    """

    def __init__(self, infile, directory=False):
        if directory:
            if not os.path.isdir(infile):
                raise IOError('directory "%s" not found' % infile)
        elif not os.path.isfile(infile):
            raise IOError('file "%s" not found' % infile)

        self.file_name, self.extension = os.path.splitext(
            os.path.basename(infile))
        self.extension = self.extension[1:]
        self.path = os.path.dirname(infile)
        self.file = os.path.abspath(infile)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        t = assertion.ensure_tmatrix(t)
        self._t = t

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, date):
        if (date is not None) and (not isinstance(date, datetime)):
            m = "'date' needs to be of type 'datetime', got %s" % type(date)
            raise TypeError(m)
        self._date = date

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, proj):
        if proj is None:
            proj = projection.Proj()
            warnings.warn("'proj' not set, so I assume '%s'" % proj.proj4)
        elif not isinstance(proj, projection.Proj):
            m = "'proj' needs to be of type 'Proj', got %s" % type(proj)
            raise TypeError(m)
        self._proj = proj

    @property
    def extent(self):
        raise NotImplementedError()

    @property
    def corners(self):
        raise NotImplementedError()

    def __len__():
        """Return the number of points.

        Returns
        -------
        positive int
            Number of objects within the file.

        """
        raise NotImplementedError()

    def load(self, extent=None):
        """Load data on demand.

        Parameters
        ----------
        extent : optional, array_like(Number, shape=(2*self.dim))
            Defines in which volume or area points shall be loaded.

        Returns
        -------
        GeoRecords
            Desired geo-data of the file.

        """
        raise NotImplementedError()

    def clean_cache(self):
        """Cleans all cached data.
        """
        raise NotImplementedError()
