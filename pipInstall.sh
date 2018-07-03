#!/bin/bash

venv=$1
if [ -n "$venv" ]
then
    source ./${venv}/bin/activate
    pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal" -r requirements.txt;
else
   echo 'Please specify virtualenv'
fi
