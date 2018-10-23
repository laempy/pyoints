#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))
PYTHON_VERSION="3.5"

# build conda package
conda build --python ${PYTHON_VERSION} ${SCRIPT_PATH}/../conda
