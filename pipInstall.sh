#!/bin/bash

venv="venv"
requirements="requirements.txt"
usage="Usage: pipInstall.sh -v venv -r requirements.txt"

while getopts r:v:h option
do
    case "${option}" in
    r) requirements=${OPTARG};;
    v) venv=${OPTARG};;
    h)
        echo $usage
        exit 0
    ;;
    \?)
        echo $usage
        exit 0
    ;;
    esac
done

if [ -n "$venv" ]
then
    source ./${venv}/bin/activate
    pip install pygdal==$(gdal-config --version).* -r $requirements --upgrade
else
   echo 'Please specify virtualenv'
fi
