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
"""Run all tests of Pyoints.
"""

import unittest
import doctest
import pkgutil

import pyoints


def get_tests(root_package):
    """Collect all doctests within a package.

    Parameters
    ----------
    root_package : package
        Root package containing docstrings to tests.

    Returns
    -------
    list of doctest.DocTestSuite
        Docstring test suites.

    """
    # https://stackoverflow.com/questions/1615406/configure-django-to-find-all-doctests-in-all-modules
    suites = []
    if hasattr(root_package, '__path__'):
        modules = pkgutil.walk_packages(
            root_package.__path__,
            root_package.__name__ + '.')
        for _, module_name, _ in modules:
            try:
                suite = doctest.DocTestSuite(module_name)
            except ValueError:
                # Presumably a "no docstrings" error. That's OK.
                pass
            else:
                suites.append(suite)
    else:
        suites.append(doctest.DocTestSuite(root_package))
    return suites


def load_tests(loader, tests, ignore):
    tests.addTests(get_tests(pyoints.storage))
    tests.addTests(get_tests(pyoints.assertion))
    tests.addTests(get_tests(pyoints.assign))
    tests.addTests(get_tests(pyoints.classification))
    tests.addTests(get_tests(pyoints.clustering))
    tests.addTests(get_tests(pyoints.coords))
    tests.addTests(get_tests(pyoints.distance))
    tests.addTests(get_tests(pyoints.extent))
    tests.addTests(get_tests(pyoints.filters))
    tests.addTests(get_tests(pyoints.fit))
    tests.addTests(get_tests(pyoints.georecords))
    tests.addTests(get_tests(pyoints.grid))
    tests.addTests(get_tests(pyoints.indexkd))
    tests.addTests(get_tests(pyoints.interpolate))
    tests.addTests(get_tests(pyoints.misc))
    tests.addTests(get_tests(pyoints.normals))
    tests.addTests(get_tests(pyoints.nptools))
    tests.addTests(get_tests(pyoints.polar))
    tests.addTests(get_tests(pyoints.projection))
    tests.addTests(get_tests(pyoints.registration))
    tests.addTests(get_tests(pyoints.smoothing))
    tests.addTests(get_tests(pyoints.surface))
    tests.addTests(get_tests(pyoints.transformation))
    tests.addTests(get_tests(pyoints.vector))

    tests.addTests(get_tests(pyoints.examples.file_examples))
    tests.addTests(get_tests(pyoints.examples.base_example))
    tests.addTests(get_tests(pyoints.examples.grid_example))
    tests.addTests(get_tests(pyoints.examples.stemfilter_example))

    return tests


class test_pyoints(unittest.TestCase):

    def test_about(self):
        self.assertTrue(hasattr(pyoints, '__version__'))
        self.assertTrue(hasattr(pyoints, '__title__'))
        self.assertTrue(hasattr(pyoints, '__summary__'))
        self.assertTrue(hasattr(pyoints, '__uri__'))
        self.assertTrue(hasattr(pyoints, '__author__'))
        self.assertTrue(hasattr(pyoints, '__email__'))
        self.assertTrue(hasattr(pyoints, '__license__'))
        self.assertTrue(hasattr(pyoints, '__copyright__'))

print('Run doctests for Pyoints %s' % pyoints.__version__)
unittest.main()
