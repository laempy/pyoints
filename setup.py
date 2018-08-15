import setuptools

# get long description from README
with open("README.md", "r") as fh:
    long_description = fh.read()
    
# get requirements
with open('requirements.txt') as f:
    install_requires = []
    for line in f:
        pkgname = line.partition('#')[0].rstrip()
        if pkgname is not '':
            install_requires.append(pkgname)

setuptools.setup (
    name="Pyoints",
    version="0.1",
    author="Sebastian Lamprecht",
    author_email="lamprecht@uni-trier.de",
    description="A Python package for point cloud, voxel and raster processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TODO",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=(
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ),

)
