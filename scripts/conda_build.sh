#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))

# build conda package
cd ${SCRIPT_PATH}/../conda
conda build .


# test environment
#conda env remove -y -n pyoints_test
#conda create -y -n pyoints_test -c $(conda build . --output) pyoints
#conda activate pyoints_test

# upload new package
#anaconda login
#anaconda upload $(conda build . --output)