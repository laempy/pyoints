#!/bin/bash
sphinx-apidoc -f -o ./docs ./pointspy
cd docs
make html

