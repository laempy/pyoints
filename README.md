# PoYnts

**PoYnts** is a python library to conveniently process and analyze point 
cloud data, voxels and raster images. It is intended to be used to support
the development of advanced algorithms for geo-data processing.

## General concept

The key idea of the concept is to provide unified data structure to handle 
points, voxels and rasters in the same manner. It is always assumed that the 
data can be interpreted as two or three dimensional point cloud. Thus we have
a collection of geo-objects (here called `GeoRecords`), which are characterized
by two or three dimensional coordinates *coords* a spatial reference (*proj*) 
and a transformation matrix (*t*). The spatial reference and transformation 
matrix required to  define the location of the geo-objects on globe. Next to 
the origin the transformation  matrix also stores the scale and rotation of the
local coordinate system.

The unified data structures simplifies the joint use of point clouds, voxels
and rasters significantly, while keeping their natural characteristics. To 
simplify the processing analysis of the geo-data each point or raster cell 
is stored in a numpy record array aaccording to it's common structure. For
example, a three dimensional point cloud represents a list of points, each
characterized by a coordinate and none to many additional attributes. So the
points are stored in a one dimensional record array. On the other hand raster
images are stored in a two dimensional record array, which allows for the
commonly used index acces of raster cells. Still each cell is characterized
by a coordinate and none to many additional attributes. We also can create
three dimensional record arrays to represent three dimensional voxels.

Since all objects of a voxel or raster are also interpreted as a point cloud,
spatial neighborhood queries, like nearest neighbours or distance search, can
be performed with unified functions. So, each extention of *GeoRecords*
provides a spatial index *IndexKD* to conveniently perform efficient
neighborhood queries. The class `IndexKD` is wrapper of different spatial 
indices, to handle *k*-dimensional spatial queries of different kinds. The 
spatial indices are always initialized on demand for performance reasions, but
cached for future queries. For example, an instance *geoRecords* of
*GeoRecords* representing  three dimensional point allows to use create a three
dimensional spatial index by calling `geoRecords.indexKD()`. If you are not 
interrested in the third dimension, you can call `geoRecords.indexKD(2)` to
perform two dimensional spatial queries.

If you just want handle coordinates without additional attributes, the class
`Coords` might be interrest you. This class also provides the *IndexKD*
feature, but waives the use of a projection and transformation matrix. 
 
 
## When should I use PoYnts?

Beased on the general concept presented abova, a bunch of algorithms, functions 
and filters have been implemented, to process geo-data with low programming 
effords. Of course, you might think: "Why should I use python for point cloud 
processing? Other languages are much more efficient." This is obviously true,
but in the experience of the author, python is very useful to implement and 
test new algorithms very quickly. Algorithms for point cloud analysis and 
processing often rely on spatial neighborhood queries. Since here **PoYnts**
takes advantage of very efficient python libraries, which are basically 
wrappers for binaries written in more efficient languages, the performance 
loss is limited. So, if you have an algorithm idea and you want to implement 
and test it quickly and want to play around with different settings, *PoYnts* 
is made for you. After finding an approiate algorithm it can be implemented in 
a more efficient language (if required). Thus *PoYnts* is particulary designed 
for scientists and developers. 



# Installation

## Dependencies

Following dependencies have to be installed manually.

### Python

The software targets python3 >= 3.5. Most code is also compatible to 
python2 >= 2.7, but hardly maintained.


### Gdal

Installation (Ubuntu)
```
sudo apt install gdal-bin
sudo apt install libgdal-dev
```

### libspatialindex

Installation (Ubuntu)
```
apt install libspatialindex-dev
```

### Liblas

Installation (Ubuntu)
```
apt install liblas-c3
```


## Install library 

### Installation via pip

```
python setup.py build
python setup.py install
```

### Installation from source

```
python setup.py build
python setup.py install
```


# Development

## Virtualenv

### Install Virtualenv

Installation (Ubuntu)
```
apt install virtualenv
```

Initialize
```
cd /path/to/library
virtualenv venv
```

### Activate virtualenv

Linux
```
cd /path/to/library
. venv/bin/activate
```

Windows
```
cd path\to\library
venv\Scripts\activate.bat
```

## Install python dependencies

Linux
```
./pipInstall.sh venv
```
or activate virtualenv and install manually with pip.
```
pip install -r requirements.txt
```


# Acknowledgements


## Python Depencencies

This software binds following python packages (distributed via pip) 
dynamically. Many thanks for providing such great open source software.


### psycopg2-binary

Federico Di Gregorio
[PyPI](https://pypi.org/project/psycopg2-binary/)
[homepage](http://initd.org/psycopg/)
[LGPL v3.0 (or later)](https://github.com/psycopg/psycopg2/blob/master/LICENSE)


### plyfile

Darsh Ranjan
[PyPI](https://pypi.org/project/plyfile/https://pypi.org/project/plyfile/)
[homepage](https://github.com/dranjan/python-plyfile)
[GPL v3.0 (or later)](https://github.com/dranjan/python-plyfile/blob/master/COPYING)


### rtree

Howard Butler
[PyPI](https://pypi.org/project/Rtree/)
[homepage](http://toblerity.org/rtree/)
[LGPL v2.1 (or later)](https://github.com/Toblerity/rtree/blob/master/LICENSE.txt)


### numpy

Travis E. Oliphant et al.
[PyPI](https://pypi.org/project/numpy/)
[homepage](http://www.numpy.org/)
[3-Clause BSD license](http://www.numpy.org/license.html#license)


### laspy

Grant Brown
[PyPI](https://pypi.org/project/laspy/)
[homepage](https://github.com/laspy/laspy)
[2-Clause BSD license](https://github.com/laspy/laspy/blob/master/LICENSE.txt)


### scipy

SciPy Developers
[PyPI](https://pypi.org/project/scipy/)
[homepage](https://www.scipy.org/)
[3-Clause BSD license](https://github.com/scipy/scipy/blob/master/LICENSE.txt)


### scikit-learn

Andreas Mueller
[PyPI](https://pypi.org/project/scikit-learn/)
[homepage](http://scikit-learn.org/stable/)
[3-Clause BSD license](https://github.com/scikit-learn/scikit-learn/blob/master/COPYING)


### pyproj

Jeff Whitaker
[PyPI](https://pypi.org/project/pyproj/)
[homepage](https://github.com/jswhit/pyproj)
[OSI approved license](https://github.com/jswhit/pyproj/blob/master/LICENSE)


### affine

Sean Gillies
[PyPI](https://pypi.org/project/affine/)
[homepage](https://github.com/sgillies/affine)
[3-Clause BSD license](https://github.com/sgillies/affine/blob/master/LICENSE.txt)


### dill

Mike McKerns
[PyPI](https://pypi.org/project/dill/)
[homepage](https://github.com/uqfoundation/dill)
[3-Clause BSD license](https://github.com/uqfoundation/dill/blob/master/LICENSE)


### pandas

The PyData Development Team
[PyPI](https://pypi.org/project/pandas/)
[homepage](http://pandas.pydata.org/)
[3-Clause BSD license](https://github.com/pandas-dev/pandas/blob/master/LICENSE)


### opencv-python

Olli-Pekka Heinisuo
[PyPI](https://pypi.org/project/opencv-python/)
[homepage](https://github.com/skvark/opencv-python)
[MIT licence](https://github.com/opencv/opencv/blob/master/LICENSE)


### networkx

NetworkX Developers
[PyPI](https://pypi.org/project/networkx/)
[homepage](http://networkx.github.io/)
[3-Clause BSD license](https://github.com/networkx/networkx/blob/master/LICENSE.txt)


### cylinder_fitting

Xingjie Pan
[PyPI](https://pypi.org/project/cylinder_fitting/)
[homepage](https://github.com/xingjiepan/cylinder_fitting)
[3-Clause BSD license](https://github.com/xingjiepan/cylinder_fitting/blob/master/license.txt)


## Optional Dependencies (for development)

Following python packages (distributed via pip) were used for software
development and documentation.


### matplotlib

John D. Hunter, Michael Droettboom
[PyPI](https://pypi.org/project/matplotlib/)
[homepage](https://matplotlib.org/)
[BSD compatible](https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE)


### autopep8

Hideo Hattori
[PyPI](https://pypi.org/project/autopep8/)
[homepage](https://github.com/hhatto/autopep8)
[MIT compatible licence](https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE)


### pycodestyle

Ian Lee
[PyPI](https://pypi.org/project/pycodestyle/)
[homepage](https://pycodestyle.readthedocs.io/en/latest/)
[Expat licence](https://pycodestyle.readthedocs.io/en/latest/index.html#license)


### Sphinx

Georg Brandl
[PyPI](https://pypi.org/project/Sphinx/)
[homepage](http://www.sphinx-doc.org/en/master/)
[3-Clause BSD license](https://github.com/sphinx-doc/sphinx)


### sphinxcontrib-napoleon

Rob Ruana
[PyPI](https://pypi.org/project/sphinxcontrib-napoleon/)
[homepage](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/)
[2-Clause BSD license](https://github.com/sphinx-contrib/napoleon/blob/master/LICENSE)



## External Depencencies

**PoYnts* binds following external libraries required for some of the python
dependencies.


### libspatialindex

Marios Hadjieleftheriou
[homepage](https://libspatialindex.github.io/)
[MIT licence](https://libspatialindex.github.io/)


### Gdal

Frank Warmerdam
[homepage](https://www.gdal.org/)
[X11/MIT licence](https://trac.osgeo.org/gdal/wiki/FAQGeneral#WhatlicensedoesGDALOGRuse)


### Liblas

Howard Butler, Mateusz Loskot and others
[homepage](https://liblas.org/)
[X11/MIT licence](https://liblas.org/copyright.html#license)



# Licence

Please see LICENCE file.


# TODO

Packaging
https://packaging.python.org/tutorials/packaging-projects/

setup.py
setup.cfg
README.rst
MANIFEST.in
LICENSE.txt




