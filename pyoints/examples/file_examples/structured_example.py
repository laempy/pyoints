# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University,
# lamprecht@uni-trier.de
#
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""Learn how to save and load structured files.

>>> import os
>>> import numpy as np
>>> from pyoints import storage

Create output path.

>>> outpath = os.path.join(
...             os.path.dirname(os.path.abspath(__file__)), '..', 'output')

Create structured data from scratch.

>>> data = {
...     'my_text': 'Some text',
...     'my_integer': 4,
...     'my_list': [0, 1, 2, 2],
...     'my_bool': False,
...     'my_ndarray': np.array([1, 4, 5]),
...     'nested': {
...         'my_text': 'Nested text.',
...         'my_float': 3.5
...     },
...     'my_recarray': np.array(
...         [(1, 'text 1'), (6, 'text 2'), (2, 'text 3')],
...         dtype=[('A', int), ('B', object)]
...      ).view(np.recarray)
... }
>>> print(data['my_text'])
Some text
>>> print(data['nested'])
{'my_text': 'Nested text.', 'my_float': 3.5}
>>> print(type(data['my_ndarray']))
<class 'numpy.ndarray'>
>>> print(type(data['my_recarray']))
<class 'numpy.recarray'>

Save as a .json-file.

>>> outfile = os.path.join(outpath, 'test.json')
>>> storage.writeJson(data, outfile)

Load the a .json-file again. Be carefull, since some data types might have been
changed.

>>> data = storage.loadJson(outfile)
>>> print(data['my_text'])
Some text
>>> print(data['nested'])
{'my_text': 'Nested text.', 'my_float': 3.5}
>>> print(type(data['my_ndarray']))
<class 'list'>
>>> print(type(data['my_recarray']))
<class 'dict'>

"""
