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
import os
import pandas
import numpy as np

from .. import nptools


def loadCsv(
        infile,
        sep=",",
        multicol_sep=".",
        dtype=None,
        header=True,
        ignore='# ',
        **kwargs):
    """Simplified loading of .csv files.

    Parameters
    ----------
    infile : String
        File to be read.
    sep : optional, Character
        Character separating the columns.
    multicol_sep : optional, Character
        Indicates how the column index of multi-column are separated form the
        column name.
    dtype : np.dtype
        Desired data type of the output numpy record array.
    header : bool
        Indicates
    *\\*kwargs : optional
        Arguments passed to `pandas.read_csv`.

    Returns
    -------
    np.recarary
        Loaded data.

    See Also
    --------
    writeCsv, pandas.read_csv

    """
    if not os.path.isfile(infile):
        raise IOError('file "%s" not found' % infile)
    if dtype is None and not header:
        raise ValueError("please specify a header or data types")
    if not isinstance(header, bool):
        raise TypeError("'header' needs to be boolean")

    # specify meta data
    if dtype is not None:
        dtype = np.dtype(dtype)
        flat_names, flat_types = _flatten_dype(dtype, sep=multicol_sep)
        pd_header = 0 if header else None
    else:
        flat_types = None
        flat_names = None
        pd_header = 0 if header else 1

    if header and flat_names is None:
        with open(infile, 'r') as f:
            line = f.readline().replace(os.linesep, '').replace(ignore, '')
            flat_names = line.replace('\n', '').split(sep)

    # laod using pandas
    df = pandas.read_csv(
        infile,
        sep=sep,
        dtype=flat_types,
        names=flat_names,
        header=pd_header,
        skiprows=0,
        skip_blank_lines=False,
        **kwargs
    )

    if dtype is None:
        # collect nested attributes automatically
        records = df.to_records()

        # collect information on multi-columns
        shape_dict = {}
        for name in records.dtype.names:
            v = name.split(multicol_sep)
            dt = records.dtype[name]
            if len(v) > 1:
                name = v[0]
                if name not in shape_dict:
                    shape_dict[name] = int(v[1])
                else:
                    i = int(v[1])
                    if i < 0:
                        raise ValueError("multi-columns need to start with 1")
                    shape_dict[name] = max(shape_dict[name], i)

        if len(shape_dict) > 0:

            # collect multicolumns
            data_dict = {}
            for name in records.dtype.names:
                v = name.split(multicol_sep)
                if len(v) > 1:
                    key = v[0]
                    if key not in data_dict:
                        shape = (len(records), shape_dict[key])
                        dt = records.dtype[name]
                        data_dict[key] = np.empty(shape, dtype=dt)
                    i = int(v[1]) - 1
                    data_dict[key][:, i] = records[name]
                else:
                    data_dict[name] = records[name]
            records = nptools.recarray(data_dict)

    else:
        data_dict = {}
        i = 0
        for key in dtype.names:
            dt = dtype[key]
            if len(dt.shape) > 0:
                data_dict[key] = np.array(df.iloc[:, i:i + dt.shape[0]])
                i = i + dt.shape[0]
            else:
                data_dict[key] = np.array(df.iloc[:, i], dtype=dt)
                i = i + 1
        records = nptools.recarray(data_dict, dtype=dtype)

    return records


def writeCsv(data, outfile, sep=",", multicol_sep=".", **kwargs):
    """Write an array to a csv-file.

    Parameters
    ----------
    data : array_like
        Data to store.
    outfile : string
        File to write the data to.
    sep : optional, Character
        Desired field separator.
    multicol_sep : optional, Character
        Indicates how the column index of multi-column shall be separated form
        the column name. For example, the column names 'normal.1', 'normal.2'
        indicate a two dimensional attribute 'normal'.
    \*\*kwargs : optional
        Arguments passed to np.save_txt`

    See Also
    --------
    loadCsv, np.save_txt

    Notes
    -----
    Limited type validation.

    """
    if not isinstance(data, (np.recarray, np.ndarray)):
        raise ValueError("'data' needs to be an numpy (record) array")
    if not os.access(os.path.dirname(outfile), os.W_OK):
        raise IOError('File %s is not writable' % outfile)

    # set column names
    names = _flatten_dype(data.dtype, sep=multicol_sep)[0]

    # unnest columns
    data = nptools.unnest(data, deep=True)
    data = list(zip(*data))

    header = sep.join(names)

    np.savetxt(
        outfile,
        data,
        fmt="%s",
        delimiter=sep,
        header=header,
        *kwargs
    )


def _flatten_dype(dtype, sep='.'):
    # Helper function to get multi-column names.
    dtype = np.dtype(dtype)
    names = []
    types = []
    for name in dtype.names:
        dt = dtype[name]
        if len(dt.shape) > 0:
            for i in range(dt.shape[0]):
                flat_name = "%s%s%i" % (name, sep, i + 1)
                names.append(flat_name)
                types.append((flat_name, dt.subdtype[0].str))
        else:
            names.append(name)
            types.append((name, dt.str))
    return names, types
