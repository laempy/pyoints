#!/bin/bash
#epydoc -o ./doc pointspy
#cd /home/sebastian/promotion/ownLibaries/pointspy/docs
#make html

#sphinx-apidoc -F -H 'pointspy-Docs' -A 'Sebastian Lamprecht' -o ./docs  ./pointspy

sphinx-apidoc -f -o ./docs ./pointspy
cd docs
make html
