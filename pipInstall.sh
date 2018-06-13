#!/bin/bash
. ./venv/bin/activate;
pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal" 
#pip install --global-option=build_ext --global-option="-I/usr/include/gdal" -r requirements.txt;