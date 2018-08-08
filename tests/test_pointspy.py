# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
import unittest
import doctest
import pkgutil

import pointspy


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

    tests.addTests(get_tests(pointspy.grid))
    tests.addTests(get_tests(pointspy.registration))
    tests.addTests(get_tests(pointspy.storage))
    tests.addTests(get_tests(pointspy.assertion))
    tests.addTests(get_tests(pointspy.assign))
    tests.addTests(get_tests(pointspy.classification))
    tests.addTests(get_tests(pointspy.clustering))
    tests.addTests(get_tests(pointspy.coords))
    tests.addTests(get_tests(pointspy.distance))
    tests.addTests(get_tests(pointspy.extent))
    tests.addTests(get_tests(pointspy.filters))
    tests.addTests(get_tests(pointspy.fit))
    tests.addTests(get_tests(pointspy.georecords))
    tests.addTests(get_tests(pointspy.indexkd))
    tests.addTests(get_tests(pointspy.interpolate))
    tests.addTests(get_tests(pointspy.misc))
    tests.addTests(get_tests(pointspy.nptools))
    tests.addTests(get_tests(pointspy.polar))
    tests.addTests(get_tests(pointspy.projection))
    tests.addTests(get_tests(pointspy.smoothing))
    tests.addTests(get_tests(pointspy.surface))
    tests.addTests(get_tests(pointspy.transformation))
    tests.addTests(get_tests(pointspy.vector))

    return tests


class test_pointpy(unittest.TestCase):

    def test_version(self):
        self.assertTrue(hasattr(pointspy, '__version__'))
        self.assertTrue(float(pointspy.__version__) > 0)


if __name__ == '__main__':
    print('unittest poynts')
    unittest.main()
