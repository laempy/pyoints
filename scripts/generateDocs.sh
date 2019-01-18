#!/bin/bash

SCRIPT_PATH=$(dirname $(realpath -s $0))
SOURCE_PATH='../pyoints'
TUTORIALS_PATH='../tutorials'
OUT_PATH='../docs'
COMPILE_PATH='../sphinx_docs'
SPHINX_PATH='../sphinx'

cd $SCRIPT_PATH

if [ -d "$OUT_PATH" ]; then
  rm -r "$OUT_PATH"
fi
if [ -d "$COMPILE_PATH" ]; then
  rm -r "$COMPILE_PATH"
fi

jupyter nbconvert --to script "$TUTORIALS_PATH/*.ipynb"

cp -r "$SPHINX_PATH" "$COMPILE_PATH"
cp -r "$TUTORIALS_PATH" "$COMPILE_PATH"

sphinx-apidoc -f -o "$COMPILE_PATH" "$SOURCE_PATH"
python3 -m sphinx "$COMPILE_PATH" "$OUT_PATH"

touch "$OUT_PATH/.nojekyll"
