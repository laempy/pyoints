# Pyoints

*Pyoints* is a python package to conveniently process and analyze point
cloud data, voxels and raster images. It is intended to be used to support
the development of advanced algorithms for geo-data processing.

## General concept

The key idea of *Pyoints* is to provide unified data structures to handle
points, voxels and rasters in the same manner. It is assumed that the
data can be interpreted as a two or three dimensional point cloud. Thus we have
a collection of geo-objects, here called `GeoRecords`, which are characterized
by two or three dimensional coordinates `coords`, a spatial reference `proj`
and a transformation matrix `t`. The spatial reference and transformation
matrix are required to define the global location of the geo-objects. Next to
the origin, the transformation matrix also stores the scale and rotation of the
local coordinate system.

The unified data structures simplify the joint use of point clouds, voxels
and rasters significantly, while keeping their natural characteristics. To
ease the processing and analysis, each point, voxel or raster cell are stored
in the commonly used numpy record array according to its natural structure. For
example, a three dimensional point cloud represents a set of points, each
characterized by a coordinate and none to many additional attributes. So the
points are stored in a one dimensional record array. Conversely raster
images are stored in a two dimensional record array, which allows for the
commonly used index access of raster cells. Still, each cell is characterized
by a coordinate and none to many additional attributes. Three dimensional
voxels are accordingly represented by three dimensional record arrays.

Since all objects of a voxel or raster are also interpreted as a point cloud,
spatial neighborhood queries, like nearest neighbors or distance search, can
be performed with unified functions. Thus, each extension of `GeoRecords`
provides a spatial index `IndexKD` to conveniently perform efficient
neighborhood queries. The class `IndexKD` is a wrapper of different spatial
indices, to handle *k*-dimensional spatial queries of different kinds. The
spatial indices are always only initialized on demand for performance reasons,
but cached for future queries. For example, an instance `geoRecords` of
`GeoRecords` representing three dimensional points allows to create a three
dimensional spatial index by calling `geoRecords.indexKD()`. If you are not
interested in the third dimension, you can call `geoRecords.indexKD(2)` to
perform two dimensional spatial queries.

If you just want to handle coordinates without additional attributes, the class
`Coords` might interest you. This class also provides the `IndexKD` feature,
but waives to use of a projection and transformation matrix.


## Is Pyoints suitable for me?

Based on the general concept of *Pyoints* presented above, a bunch of
algorithms, functions and filters have been implemented, to process geo-data
with low programming efforts.

Of course, one might wonder whether Python is adequate to analyze point cloud
data since other languages languages are much more efficient. This is obviously
true, but in the experience of the author, Python is very useful to implement
and test new algorithms very quickly. Algorithms for point cloud analysis and
processing often rely on spatial neighborhood queries. Since *Pyoints*
takes advantage of very efficient Python libraries, which are basically
wrappers for binaries written in more efficient languages, the performance
loss is limited. Additionally, *Pyoints* takes care of selecting and caching
appropiate spatial indices for you, so you can focus on the core of your
algorithm idea.

So, if you have an algorithm idea and you want to implement it quickly and play
around with different settings, *Pyoints* is made for you. After finding an
approiate algorithm it can be implemented in a more efficient language
if you like. Thus *Pyoints* is particulary designed for scientists and
developers of new geo-spatial algorithms.


## Installation

### Conda Installation

Conda installation is recommended. To install *Pyoints* with
[Conda](https://conda.io/docs/user-guide/getting-started.html) run:

```
conda install -c laempy pyoints
```

Currently Linux (64 bit) and Windows (64 bit) have been packaged.


### Conda Build

If your system is not supported yet, you might build and install it using the
[conda reciepe](conda/meta.yaml) specifying your desired Python version.
Currently Linux (64 bit) and Windows (64 bit) have been tested successfully
with Python 3.5, 3.6 and 3.7.

#### Build

```
conda build /path/to/pyoints/conda --python=3.x
```

#### Installation

Create a virtual environment `pyoints_env` and add the newly build *Pyoints* package.
The package path can be found with `conda build . --output`.

```
conda create -n pyoints_env /path/to/builded/package/pyoints-*-py*_*.tar.bz2 --use-local
```

Finally install the package:

```
conda activate pyoints_env
conda install pyoints
```

To enable full LAS support (spatial reference system definition), you need to 
install *liblas* and its python bindings manually.

```
pip install liblas
```


#### Tests

Test the package by running the provided doctests:

```
python /path/to/pyoints/tests/test_pyoints.py
```


### Installation from source

We recommend to use a virtual environment (like Conda) to install *Pyoints*.

Installation of the external dependencies:
```
conda install gdal pyproj numpy rtree
```

Cloning of the repository:
```
git clone https://github.com/laempy/pyoints.git
cd pyoints
```

Installation of the python packages via *pip*.
```
pip install pygdal==$(gdal-config --version).* -e .
```
Unfortunately, the gdal version is not detected automatically by *pygdal*.
Thus, we have to specify the gdal version expicilely with
`gdal-config --version`.


## Documentation

You can find the documentation at
[docs](https://laempy.github.io/pyoints). Here you can find some introducing
tutorials and docstring examples for each module.



## License

Copyright (c) 2018, Sebastian Lamprecht, Trier University,
lamprecht@uni-trier.de

*Pyoints* is free software made available under the GNU General Public
License v3 or later (GPLv3+). See [LICENSE](LICENSE) for details.



## Citing

Please cite *Pyoints* as follows:

S. Lamprecht (2019):
"Pyoints: A Python package for point cloud, voxel and raster processing".
[![DOI](https://zenodo.org/badge/144562205.svg)](https://zenodo.org/badge/latestdoi/144562205)



## Contributing

Any conribution to the *pyoints* project is welcome. See
[CONTRIBUTING](CONTRIBUTING.md) for details.



## Acknowledgements

### Funding

This work has been supported by the European Commission under the grant
agreement number 774571 Project [PANTHEON](http://www.project-pantheon.eu).


### Datasets

For the tutorials the "Stanford Bunny" dataset has been used:

Stanford Computer Graphics Laboratory (1994): "Stanford Bunny", URL:
[https://graphics.stanford.edu/data/3Dscanrep/](https://graphics.stanford.edu/data/3Dscanrep/),
(Accessed: 17 January 2019)


### Python Depencencies

This software dynamically binds the following Python packages (distributed via
pip). Many thanks to the authors for providing such great open source software.


#### cylinder_fitting

Xingjie Pan
* [PyPI](https://pypi.org/project/cylinder_fitting/)
* [homepage](https://github.com/xingjiepan/cylinder_fitting)
* [3-Clause BSD license](https://github.com/xingjiepan/cylinder_fitting/blob/master/license.txt)


#### dill

Mike McKerns
* [PyPI](https://pypi.org/project/dill/)
* [homepage](https://github.com/uqfoundation/dill)
* [3-Clause BSD license](https://github.com/uqfoundation/dill/blob/master/LICENSE)


#### pygdal

Frank Warmerdam
* [PyPI](https://pypi.org/project/pygdal/)
* [homepage](https://github.com/nextgis/pygdal)
* [2-Clause BSD license](https://github.com/laspy/laspy/blob/master/LICENSE.txt)


#### laspy

Grant Brown and Howard Butler
* [PyPI](https://pypi.org/project/laspy/)
* [homepage](https://github.com/laspy/laspy)
* [2-Clause BSD license](https://github.com/laspy/laspy/blob/master/LICENSE.txt)


#### liblas

Romulo Goncalves
* [PyPI](https://pypi.org/project/libLAS/)
* [homepage](https://liblas.org/)
* [3-Clause BSD license](https://liblas.org/copyright.html#license)


#### networkx

NetworkX Developers
* [PyPI](https://pypi.org/project/networkx/)
* [homepage](http://networkx.github.io/)
* [3-Clause BSD license](https://github.com/networkx/networkx/blob/master/LICENSE.txt)


#### numpy

Travis E. Oliphant et al.
* [PyPI](https://pypi.org/project/numpy/)
* [homepage](http://www.numpy.org/)
* [3-Clause BSD license](http://www.numpy.org/license.html#license)


#### opencv-python

Olli-Pekka Heinisuo
* [PyPI](https://pypi.org/project/opencv-python/)
* [homepage](https://github.com/skvark/opencv-python)
* [MIT license](https://github.com/opencv/opencv/blob/master/LICENSE)


#### pandas

The PyData Development Team
* [PyPI](https://pypi.org/project/pandas/)
* [homepage](http://pandas.pydata.org/)
* [3-Clause BSD license](https://github.com/pandas-dev/pandas/blob/master/LICENSE)


#### plyfile

Darsh Ranjan
* [PyPI](https://pypi.org/project/plyfile/https://pypi.org/project/plyfile/)
* [homepage](https://github.com/dranjan/python-plyfile)
* [GPL v3.0 (or later)](https://github.com/dranjan/python-plyfile/blob/master/COPYING)


#### pyproj

Jeff Whitaker
* [PyPI](https://pypi.org/project/pyproj/)
* [homepage](https://github.com/jswhit/pyproj)
* [OSI approved license](https://github.com/jswhit/pyproj/blob/master/LICENSE)


#### scikit-learn

Andreas Mueller
* [PyPI](https://pypi.org/project/scikit-learn/)
* [homepage](http://scikit-learn.org/stable/)
* [3-Clause BSD license](https://github.com/scikit-learn/scikit-learn/blob/master/COPYING)


#### scipy

SciPy Developers
* [PyPI](https://pypi.org/project/scipy/)
* [homepage](https://www.scipy.org/)
* [3-Clause BSD license](https://github.com/scipy/scipy/blob/master/LICENSE.txt)


#### rtree

Howard Butler
* [PyPI](https://pypi.org/project/Rtree/)
* [homepage](http://toblerity.org/rtree/)
* [LGPL v2.1 (or later)](https://github.com/Toblerity/rtree/blob/master/LICENSE.txt)



### External Depencencies

*Pyoints* binds following external libraries as required by some of the pip
packages.


#### GDAL

Frank Warmerdam
* [homepage](https://www.gdal.org/)
* [X11/MIT license](https://trac.osgeo.org/gdal/wiki/FAQGeneral#WhatlicensedoesGDALOGRuse)


#### Liblas

Howard Butler, Mateusz Loskot et. al.
* [homepage](https://liblas.org/)
* [X11/MIT license](https://liblas.org/copyright.html#license)


#### libspatialindex

Marios Hadjieleftheriou
* [homepage](https://libspatialindex.github.io/)
* [MIT license](https://libspatialindex.github.io/)


