#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))

# build conda package
cd ${SCRIPT_PATH}/../conda
conda build .


#anaconda login
#anaconda upload $(conda build . --output)
