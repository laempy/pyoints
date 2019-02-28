#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))

# build conda package
cd ${SCRIPT_PATH}/../conda
conda build .


# Testing the Environment
##########################
#conda env remove -y -n pyoints_test
#conda create -y -n pyoints_test -c $(conda build . --output) pyoints


# upload new package
#anaconda login
#anaconda upload $(conda build . --output)