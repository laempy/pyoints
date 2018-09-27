#!/bin/bash
SCRIPT_PATH=$(dirname $(realpath -s $0))
cd $SCRIPT_PATH
if [ -d "../docs" ]; then
  # "remove docs"
  rm -r ../docs
fi
sphinx-apidoc -f -o ../sphinx ../pyoints
cd ../sphinx
make html
mv ./html ../docs
touch ../docs/.nojekyll
