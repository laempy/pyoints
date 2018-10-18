#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))
VENV=""
REQUIREMENTS="${SCRIPT_PATH}/../requirements.txt"
DEV_REQUIREMENTS="${SCRIPT_PATH}/../dev_requirements.txt"
USAGE='\nUsage:\n\npip_install.sh -v venv (-d)\n\n\t-v [arg]\t: Specify path to virtual
environment.\n\t-d\t\t: Also install packages used for development.\n\t-h\t\t:
Display this help.'

requirements="-r $REQUIREMENTS"
while getopts v:dhc option
do
    case "${option}" in
    d) requirements="-r $REQUIREMENTS -r $DEV_REQUIREMENTS";;
    v) VENV=${OPTARG};;
    c) CREATE=true;;
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
    if [ -n "$CREATE" ]
    then
        virtualenv -p python3 ${VENV};
    fi 
    source ${VENV}/bin/activate
    pip install pygdal==$(gdal-config --version).* $requirements --upgrade
else
    echo "Please specify virtualenv with '-v [arg]'".
    echo -e $USAGE
    exit 0
fi
