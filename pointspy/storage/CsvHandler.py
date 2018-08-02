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
        Character seperating the columns.
    multicol_sep : optional, Character
        Indicates how the column index of multi-column are seperated form the
        column name.
    dtype : np.dtype
        Desired data type of the output numpy record array.
    header : bool
        Indicates
    **kwargs : optional
        Parameters passed to `pandas.read_csv`.

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
            flat_names = line.split(sep)

    # laod using pandas
    df = pandas.read_csv(
        infile,
        sep=sep,
        dtype=flat_types,
        names=flat_names,
        header=pd_header,
        skiprows=0,
        skip_blank_lines=False,
        **kwargs,
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
                data_dict[key] = np.array(df.iloc[:, i:i+dt.shape[0]])
            else:
                data_dict[key] = np.array(df.iloc[:, i], dtype=dt)
            i = i + 1
        records = nptools.recarray(data_dict, dtype=dtype)

    return records


def writeCsv(data, filename, sep=",", multicol_sep=".", **kwargs):
    """Write a array to a csv-file.

    Parameters
    ----------
    data : array_like
        Data to store.
    filename : string
        File to write the data to.
    sep : optional, Character
        Desired field seperator.
    multicol_sep : optional, Character
        Indicates how the column index of multi-column shall be seperated form
        the column name. For example, the column names 'normal.1', 'normal.2'
        indicate a two dimensional attribute 'normal'.
    **kwargs : optional
        Arguments passed to np.save_txt`

    See Also
    --------
    loadCsv, np.save_txt

    Notes
    -----
    Limited type validation.

    """
    # set column names
    names = _flatten_dype(data.dtype, sep=multicol_sep)[0]

    # unnest columns
    data = nptools.unnest(data, deep=True)
    data = list(zip(*data))

    header = sep.join(names)

    np.savetxt(
        filename,
        data,
        fmt="%s",
        delimiter=sep,
        header=header,
        *kwargs,
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
                flat_name = "%s%s%i" % (name, sep, i+1)
                names.append(flat_name)
                types.append((flat_name, dt.subdtype[0].str))
        else:
            names.append(name)
            types.append((name, dt.str))
    return names, types
