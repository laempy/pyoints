from setuptools import setup, find_packages

setup (
	name='pointspy',
	version='0.1',
	packages=find_packages(),

	# Declare your packages' dependencies here, for eg:
	install_requires=[
	    'numpy>=1.14.0',
	    'rtree>=0.8.3',
	    'liblas>=1.8.3',
	    'laspy>=1.5.1',
	    'scipy>=1.0.0',
	    'scikit-learn>=0.19.1',
	    'pyproj>=1.9.5.1',
	    'gdal>=1.11.3',
	    'pandas>=0.22.0',
	    'configobj>=5.0.6',
	    'pyyaml>=3.12'
	],

	# Fill in these to make your Egg ready for upload to
	# PyPI
	author='Sebastian Lamprecht',
	author_email='lamprecht@uni-trier.de',

	#summary = 'Just another Python package for the cheese shop',
	url='',
	license='',
	long_description='A python library to conveniently process point cloud data.',

	# could also include long_description, download_url, classifiers, etc.

    )