#!/bin/bash

VENV=""
REQUIREMENTS='requirements.txt'
DEV_REQUIREMENTS='dev_requirements.txt'
USAGE='Usage:\ninstall.sh -v venv (-d)\n-v [arg] : path of virtualenv'

requirements="-r $REQUIREMENTS"
while getopts v:dh option
do
    case "${option}" in
    d) requirements="-r $REQUIREMENTS -r $DEV_REQUIREMENTS";;
    v) VENV=${OPTARG};;
    h)
        echo $USAGE
        exit 0
    ;;
    *)
        echo $USAGE
        exit 0
    ;;
    esac
done

if [ -n "$VENV" ]
then
    source ./${VENV}/bin/activate
    pip install pygdal==$(gdal-config --version).* $requirements --upgrade
else
   echo 'Please specify virtualenv'
fi
