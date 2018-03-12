from setuptools import setup, find_packages

setup (
	name='pointspy',
	version='0.1',
	packages=find_packages(),

	# Declare your packages' dependencies here, for eg:
	install_requires=[
	    'numpy',
	    'rtree',
	    'liblas',
	    'laspy',
	    'scipy',
	    'scikit-learn',
	    'pyproj',
	    'gdal',
	    'pandas',
	    'configobj',
	    'pyyaml',
            'psycopg2-binary',
            'Sphinx', # docs
            'sphinxcontrib-napoleon', # docs
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