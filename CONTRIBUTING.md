# Contributing

We gladly invite you to contribute to *Pyoints* by making suggestions, report
bugs, or making changes on the code. Contributions can be made via your GitHub 
account.


## Making Suggestions 

You can make suggestions or report bugs by using the issue board on GitHub or 
by sensing an email to lamprecht@uni-trier.de.


## Making Changes

To contribute your code please fork the *Pyoints* repository on GitHub. Push 
your changes with meaningfull commit messages to a topic branch. Then create 
a pull request to propose to incorporate your changes to the main project.

To ensure a uniform quality of code, please follow our 
[coding conventions](#coding-conventions).


## Coding Conventions

### Documentation

Please make sure to document your contributed code appropiatly. The *Pyoints*
project uses *Docstrings* 
([PEP257](https://www.python.org/dev/peps/pep-0257/) 
to document the code of each class or function. The documentation and comments 
must be written in english. Wherever possible the *Examples* section should 
illustrate the basic usage of the code. The examples should also cover special
cases, which might reveal unexpected behavior or major bugs.
 

### Style Guide.

Please follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style 
guidelines.


### Testing

To test the software please take a look at the [tests](tests) directory.
Currently doctests are used to test the functionality of a majority of 
*Pyoints* classes and functions. You can run the file ``tests/test_pyoints.py``
to run the doctests. After adding a new file, please add a *doctest* reference 
to ``tests/test_pyoints.py``. To increase the quality of our code we encourage
you to write additional tests.

