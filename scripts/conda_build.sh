#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))

# build conda package
cd ${SCRIPT_PATH}/../conda
conda build .

# In case of failure
#####################
#conda build purge-all
#conda clean -t
#conda clean -p
#conda clean -a


# Testing the Environment
##########################
#conda env remove -y -n pyoints_test
#conda create -y -n pyoints_test -c $(conda build . --output) pyoints
#conda create -n pyoints_env /path/to/builded/package/pyoints-*-py*_*.tar.bz2 --use-local


# upload new package
#anaconda login
#anaconda upload $(conda build . --output)

# Windows
#############
#conda build . --debug 1> conda.log 2>&1 