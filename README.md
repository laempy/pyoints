# PoYnts

**PoYnts** is a python library to conveniently process and analyze point 
cloud data, voxels and raster images. It is intended to be used to support
the development of advanced algorithms for geo-data processing.

## General concept

The key idea of the concept is to provide unified data structure to handle 
points, voxels and rasters in the same manner. It is always assumed that the 
data can be interpreted as two or three dimensional point cloud. Thus we have
a collection of geo-objects (called `GeoRecords`), which are characterized by 
coordinates *coords* a spatial reference (*proj*) and a transformation matrix 
(*t*). The spatial reference and transformation matrix required to define the 
location of the geo-objects on globe. Next to the origin the transformation 
matrix also stores the scale and rotation of the local coordinate system.

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

## Depencencies

This software binds following python packages (distributed via pip) 
dynamically. Many thanks for providing such great open source software.


psycopg2-binary  # LGPL v3 licence, https://github.com/psycopg/psycopg2/blob/master/LICENSE

plyfile  # GNU General Public License v3.0, https://github.com/dranjan/python-plyfile/blob/master/COPYING

rtree  # GNU Lesser General Public License v2.1 (or any later version), https://github.com/Toblerity/rtree/blob/master/LICENSE.txt




libspatialindex-dev  # MIT licence, https://libspatialindex.github.io/


numpy  # 3-Clause BSD license, https://github.com/numpy/numpy/blob/master/LICENSE.txt


laspy  # 2-Clause BSD license, https://github.com/laspy/laspy/blob/master/LICENSE.txt


scipy  # 3-Clause BSD license, https://github.com/scipy/scipy/blob/master/LICENSE.txt


scikit-learn  # 3-Clause BSD license, https://github.com/scikit-learn/scikit-learn/blob/master/COPYING


pyproj  # https://github.com/jswhit/pyproj/blob/master/LICENSE

Contact:  Jeffrey Whitaker <jeffrey.s.whitaker@noaa.gov

copyright (c) 2013 by Jeffrey Whitaker.

Permission to use, copy, modify, and distribute this software
and its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both the copyright notice and this permission
notice appear in supporting documentation. THE AUTHOR DISCLAIMS
ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT
SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


affine # 3-Clause BSD license, https://github.com/sgillies/affine/blob/master/LICENSE.txt


dill  # 3-Clause BSD license, https://github.com/uqfoundation/dill/blob/master/LICENSE


pandas  # 3-Clause BSD license, https://github.com/pandas-dev/pandas/blob/master/LICENSE


opencv-python  # 3-Clause BSD license, https://github.com/opencv/opencv/blob/master/LICENSE


networkx  # 3-Clause BSD license, https://github.com/networkx/networkx/blob/master/LICENSE.txt


cylinder_fitting  # 3-Clause BSD license, https://github.com/xingjiepan/cylinder_fitting/blob/master/license.txt

matplotlib  # PSF licence, https://github.com/matplotlib/matplotlib/blob/master/LICENSE


## For development

Following python packages (distributed via pip) were used for software 
development and documentation.

* autopep8
* pycodestyle
* matplotlib
* Sphinx
* sphinxcontrib-napoleon



## Licence

Please see LICENCE file


# TODO

Packaging
https://packaging.python.org/tutorials/packaging-projects/

setup.py
setup.cfg
README.rst
MANIFEST.in
LICENSE.txt




