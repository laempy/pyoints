#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))
cd $SCRIPT_PATH
sphinx-apidoc -f -o ../docs ../pyoints
cd ../docs
make html

