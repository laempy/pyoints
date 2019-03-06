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

__all__ = [
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
]

__version__ = '0.2.0rc3'

__title__ = "Pyoints"
__summary__ = "A Python package for point cloud, voxel and raster processing."
__uri__ = "https://github.com/laempy/pyoints"

__author__ = "Sebastian Lamprecht"
__email__ = "lamprecht@uni-trier.de"

__license__ = "GPLv3+"
__copyright__ = "2019, %s" % __author__


def version():
    """Get the version of pyoints.
    
    Returns
    -------
    str
        Version specification.
    
    """
    return __version__