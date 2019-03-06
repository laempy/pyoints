import os
import setuptools

# get path of script
script_path = os.path.dirname(os.path.abspath(__file__))

# get long description from README
with open(os.path.join(script_path, "README.md"), "r") as f:
    long_description = f.read()

# get requirements
with open(os.path.join(script_path, 'requirements.txt'), "r") as f:
    install_requires = []
    for line in f:
        pkgname = line.partition('#')[0].rstrip()
        if pkgname is not '':
            install_requires.append(pkgname)
            
# get package meta data
with open(os.path.join(script_path, 'pyoints', 'about.py')) as f:
    about = {}
    exec(f.read(), about)


setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['__summary__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about['__uri__'],
    project_urls={
        'Documentation': 'https://laempy.github.io/pyoints',
        'GitHub': 'https://github.com/laempy/pyoints',
    },
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=(
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
    ),
    python_requires='>=3.5',
    license=about['__license__'],
    include_package_data=True,
)
