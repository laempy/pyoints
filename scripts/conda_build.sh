#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))

# build conda package
conda build ${SCRIPT_PATH}/../conda
