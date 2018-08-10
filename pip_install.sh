#!/bin/bash

VENV=""
REQUIREMENTS='requirements.txt'
DEV_REQUIREMENTS='dev_requirements.txt'
USAGE='\nUsage:\n\npip_install.sh -v venv (-d)\n\n\t-v [arg]\t: Specify path to virtual 
environment.\n\t-d\t\t: Also install packages used for development.\n\t-h\t\t: 
Display this help.'

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
    echo "Please specify virtualenv with '-v [arg]'".
    echo -e $USAGE
    exit 0
fi
