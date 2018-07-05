A python library to conveniently process point cloud data and rasters.



## Dependencies

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


## Development with Virtualenv

Installation (Ubuntu)
```
apt install virtualenv
```

### Initialize Virtualenv

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

### Install python packages

Linux
```
./pipInstall.sh venv
```

# TODO

Packaging
https://packaging.python.org/tutorials/distributing-packages/

setup.py
setup.cfg
README.rst
MANIFEST.in
LICENSE.txt