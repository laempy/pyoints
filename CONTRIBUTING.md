# Contributing

We gladly invite you to contribute to *Pyoints* by making suggestions, report
bugs, or making changes on the code. Contributions can be made via your GitHub 
account.


## Making Suggestions 

You can make suggestions or report bugs by using the issue board on GitHub or 
by sensing an email to the package maintainer(s).


## Making Changes

To contribute your code, please fork *Pyoints* on GitHub. Push your changes 
with meaningfull commit messages to a topic branch. Then create a pull request 
to propose to incorporate your changes to the main project. To ensure a uniform 
quality of code, please follow our [coding conventions](#coding-conventions).


## Coding Conventions


### Style Guide.

Please follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style 
guidelines.

### Documentation

Please make sure to document your contributed code appropiatly. The *Pyoints*
project uses *Docstrings* 
([PEP257](https://www.python.org/dev/peps/pep-0257/) 
to document the code of each class or function. The documentation and comments 
must be written in English. Wherever possible, the *Examples* section should 
illustrate the basic usage of the code. The examples should also cover special
cases, which might reveal unexpected behavior or major bugs.

### Testing

To test the software please take a look at the [tests](pyoints/tests) 
directory. Currently doctests are used to test the functionality of a majority
of *Pyoints* classes and functions. You can run the file 
``tests/test_pyoints.py`` to run the doctests. After adding a new file, please
add a *doctest* reference to ``tests/test_pyoints.py``. To increase the quality
of our code we encourage you to write additional tests.


## Installation from source

Please install the external dependencies first (see [README](README.md)). We 
recommend to use a virtual python environment to install *Pyoints*. 
Unfortunately,  the gdal version is not detected automatically by `pgdal`. 
Thus, instead run:

```
pip install pygdal==$(gdal-config --version).* -r requirements.txt --upgrade
```


## Software recommendations

The following Python packages were used for software development, testing and
documentation.


### autopep8

Hideo Hattori
* [PyPI](https://pypi.org/project/autopep8/)
* [homepage](https://github.com/hhatto/autopep8)
* [MIT compatible license](https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE)


### CloudCompare

Daniel Girardeau-Montaut
* [homepage](https://www.danielgm.net/cc/)
* [GPL v2](https://github.com/CloudCompare/CloudCompare/blob/master/license.txt)


### matplotlib

John D. Hunter, Michael Droettboom
* [PyPI](https://pypi.org/project/matplotlib/)
* [homepage](https://matplotlib.org/)
* [BSD compatible license](https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE)


### pycodestyle

Ian Lee
* [PyPI](https://pypi.org/project/pycodestyle/)
* [homepage](https://pycodestyle.readthedocs.io/en/latest/)
* [Expat license](https://pycodestyle.readthedocs.io/en/latest/index.html#license)


### Sphinx

Georg Brandl
* [PyPI](https://pypi.org/project/Sphinx/)
* [homepage](http://www.sphinx-doc.org/en/master/)
* [3-Clause BSD license](https://github.com/sphinx-doc/sphinx)


### sphinxcontrib-napoleon

Rob Ruana
* [PyPI](https://pypi.org/project/sphinxcontrib-napoleon/)
* [homepage](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/)
* [2-Clause BSD license](https://github.com/sphinx-contrib/napoleon/blob/master/LICENSE)

