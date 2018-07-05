#!/bin/bash

venv=$1
if [ -n "$venv" ]
then
    source ./${venv}/bin/activate
    pip install pygdal==$(gdal-config --version).* -r requirements.txt
else
   echo 'Please specify virtualenv'
fi
