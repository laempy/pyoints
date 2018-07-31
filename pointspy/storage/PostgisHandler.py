import os
import numpy as np

import subprocess
import psycopg2

from .DumpHandler import dumpstring_from_object
from .. import nptools
from .. extent import Extent


# Postgis
##########


class PostgisHandler(psycopg2._psycopg.connection):

    # Type specification
    _sFLOAT = "%f"
    _sINTEGER = "%i"
    _sSTRING = "'%s'"
    _sARRAY = "ARRAY[%s]"
    _NULL = 'NULL'
    _sPOINT = "ST_GeomFromEWKT('SRID=%i;POINT(%s)')"
    _sMULTIPOINT = "ST_GeomFromEWKT('SRID=%i;MULTIPOINT(%s)')"
    _sCONVEXHULL = "ST_ConvexHull(%s)"
    _sEXTENT = 'ST_MakeEnvelope(%f,%f,%f,%f,%i)'
    _sLINESTRING = "ST_GeomFromEWKT('SRID=%i;LINESTRING(%s)')"

    def __init__(self, connectionString):
        psycopg2._psycopg.connection.__init__(self, connectionString)
        self.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    @staticmethod
    def connectionString(d):
        return ('host=%s port=%s user=%s password=%s dbname=%s')\
            % (d['host'], d['port'], d['user'], d['password'], d['dbname'])

    @staticmethod
    def npDtypes2dtypes(dtypes):
        dtypes = np.dtype(dtypes)
        outTypes = []

        for key in dtypes.names:
            dtype = np.dtype(dtypes[key])
            outType = '"%s" ' % key

            shape = None
            if dtype.subdtype is not None:
                dtype, shape = dtype.subdtype

            if np.issubdtype(dtype, np.float):
                outType += "DOUBLE PRECISION"
            elif np.issubdtype(dtype, np.int):
                outType += "INTEGER"
            elif np.issubdtype(dtype, np.str) or dtype == np.object:
                outType += "VARCHAR"
            elif np.issubdtype(dtype, np.bool):
                outType += "BOOLEAN"
            else:
                raise ValueError('Datatype %s not found' % (str(dtype)))

            if shape is not None:
                outType += '[%i]' % shape

            outTypes.append(outType)
        return outTypes

    @staticmethod
    def npDtypes2Converters(dtypes):
        dtypes = np.dtype(dtypes)
        converters = {}
        for key in dtypes.names:
            dtype = dtypes[key]

            shape = None
            if dtype.subdtype is not None:
                dtype, shape = dtype.subdtype

            if shape is not None:
                converter = PostgisHandler.sARRAY
            elif np.issubdtype(dtype, np.float):
                converter = PostgisHandler.sFLOAT
            elif np.issubdtype(dtype, np.int):
                converter = PostgisHandler.sINTEGER
            elif np.issubdtype(dtype, np.str):
                converter = PostgisHandler.sSTRING
            elif np.issubdtype(dtype, np.bool_):
                converter = PostgisHandler.sBOOLEAN
            elif np.issubdtype(dtype, np.object):
                converter = PostgisHandler.sSTRING
            else:
                raise ValueError('Datatype "%s" not found' % (str(dtype)))
            converters[key] = converter
        return converters

    @staticmethod
    def sFLOAT(value):
        return PostgisHandler._NULL if value is None or np.isnan(
            value) else PostgisHandler._sFLOAT % value

    @staticmethod
    def sARRAY(arr):
        values = ','.join(np.array(arr).astype(str))
        return PostgisHandler._NULL if len(arr) == 0 or np.isnan(
            arr).sum() == len(arr) else PostgisHandler._sARRAY % values

    @staticmethod
    def sSTRINGARRAY(arr):
        values = "'%s'" % "','".join(np.array(arr).astype(str))
        return PostgisHandler._NULL if len(
            arr) == 0 else PostgisHandler._sARRAY % values

    @staticmethod
    def sINTEGER(value):
        return PostgisHandler._NULL if value is None or np.isnan(
            value) else PostgisHandler._sINTEGER % value

    @staticmethod
    def sBOOLEAN(value):
        return PostgisHandler._NULL if value is None or np.isnan(
            value) else PostgisHandler._sSTRING % value

    @staticmethod
    def sSTRING(value):
        return PostgisHandler._NULL if value is None or value is '' else PostgisHandler._sSTRING % value

    @staticmethod
    def sPOINT(coords, srid):
        values = ' '.join([str(coord) for coord in coords])
        return PostgisHandler._sPOINT % (srid, values)

    @staticmethod
    def sMULTIPOINT(coords, srid):
        values = '(' + '),('.join([' '.join(coord)
                                   for coord in coords.astype(str)]) + ')'
        return PostgisHandler._sMULTIPOINT % (srid, values)

    @staticmethod
    def sLINESTRING(coords, srid):
        values = ','.join([' '.join(coord)
                           for coord in np.array(coords).astype(str)])
        return PostgisHandler._sLINESTRING % (srid, values)

    @staticmethod
    def sCONVEXHULL(coords, srid):
        if coords is not None and len(
                coords.shape) == 2 and coords.shape[0] > 2 and coords.shape[1] > 1:
            ext = Extent(coords)
            A = np.prod(ext.ranges[0:2])
            if A > 0:
                coordStr = PostgisHandler.sMULTIPOINT(coords, srid)
                return PostgisHandler._sCONVEXHULL % (coordStr)
        return PostgisHandler._NULL

    @staticmethod
    def sEXTENT(ext, srid):
        if len(ext) == 6:
            ext = (ext[0], ext[1], ext[3], ext[4])
        return PostgisHandler._sEXTENT % (ext[0], ext[1], ext[2], ext[3], srid)

    @staticmethod
    def sPYDUMP(data):
        return "'" + dumpstring_from_object(data) + "'"

    @staticmethod
    def _queryGenerator(converter, stream, bulk):
        while True:
            query = ''
            try:
                for _ in range(bulk):
                    query += converter(next(stream))
            except StopIteration:
                break
            finally:
                if query is not '':
                    yield query

    def colNames(self, query):
        columnQuery = 'SELECT * FROM (%s) AS "count_foo" LIMIT 1;' % (
            query.rstrip(';'))
        return [desc[0] for desc in self(columnQuery).description]

    def vacuum(self, table, schema='public'):
        query = 'VACUUM "%s"."%s";' % (schema, table)
        self(query)

    def insertStream(
            self,
            table,
            stream,
            colNames=None,
            bulk=5000,
            schema='public'):
        # Stream of precompiled attributes
        # e.g. postgisWriter.POINT(coords,srid),value1,value2

        insertQuery = 'INSERT INTO "%s"."%s" ' % (schema, table)
        if colNames is not None:
            insertQuery += '("%s")' % ('","'.join(colNames))
        insertQuery += 'VALUES(%s);'

        def converter(row):
            return insertQuery % (','.join(row))

        for query in self._queryGenerator(converter, iter(stream), bulk):
            self(query)

    def updateStream(
            self,
            table,
            stream,
            colNames,
            idField,
            bulk=5000,
            schema='public'):
        # Stream of precompiled attributes
        # stream of format id,(value1,value2)
        fields = '","'.join(colNames)
        updateQuery = 'UPDATE "%s"."%s" SET ("%s")=' % (schema, table, fields)
        updateQuery += '(%s) WHERE ' + '"%s"' % idField + '=%s;'

        def converter(row):
            return updateQuery % (','.join(row[1]), row[0])

        for query in self._queryGenerator(converter, iter(stream), bulk):
            self(query)
            self.vacuum(table, schema)  # vacuum necessary for updates

    def upsertStream(
            self,
            table,
            stream,
            colNames,
            idField,
            bulk=5000,
            schema='public'):
        fields = ('","'.join(colNames))

        upsertQuery = 'INSERT INTO "%s"."%s" ("%s","%s") ' % (
            schema, table, idField, fields)
        upsertQuery += 'VALUES(%s,%s) '
        upsertQuery += 'ON CONFLICT ("%s") DO ' % idField
        upsertQuery += 'UPDATE SET ("%s")=' % (fields)
        upsertQuery += '(%s);'

        def converter(row):
            id = row[0]
            values = ','.join(row[1])
            return upsertQuery % (id, values, values)

        for query in self._queryGenerator(converter, iter(stream), bulk):
            self(query)
            self.vacuum(table, schema)  # vacuum necessary for updates

    def __call__(self, sqlStatement, where=None, bulk=None, schema=None):
        if where is not None:
            if sqlStatement[-1] is ';':
                sqlStatement = query[:-1]
            sqlStatement = 'SELECT * FROM (%s) AS "tempQuery" WHERE %s;' % (
                sqlStatement,
                where)

        if bulk is not None:
            stream = iter(sqlStatement.split(';\n'))

            def converter(query): return '%s;' % query
            queryGen = self._queryGenerator(converter, stream, bulk)
        else:
            queryGen = [sqlStatement]

        try:
            cursor = self.cursor()
            if schema is not None:
                query = 'SET search_path TO "%s";' % schema
                cursor.execute(query)
            for query in queryGen:
                try:
                    cursor.execute(query)
                except Exception as e:
                    print(query)
                    raise e
            if schema is not None:
                query = 'SET search_path TO "public";'
                cursor.execute(query)
        finally:
            self.commit()
        return cursor

    def load(self, query, npDtypes=[], converters={}, where=None, bulk=None):
        cursor = self(query, where)

        if bulk is None:
            return self.recArrayFromCursor(
                cursor, npDtypes=npDtypes, converters=converters)
        else:
            def gen():
                while True:
                    recArray = PostgisHandler.recArrayFromCursor(
                        cursor, npDtypes=npDtypes, converters=converters, bulk=bulk)
                    if recArray is None:
                        break
                    yield recArray
            return gen()

    @staticmethod
    def recArrayFromCursor(cursor, npDtypes=[], converters={}, bulk=None):
        assert isinstance(
            converters, dict), 'Converters have to be a dictionary'

        if bulk is None:
            rawData = cursor.fetchall()
        else:
            rawData = cursor.fetchmany(bulk)

        if len(rawData) == 0:
            return []

        cols = zip(*rawData)

        # Apply converters
        dataDict = {}
        colNames = [desc[0] for desc in cursor.description]
        for colName, col in zip(colNames, cols):
            if colName in converters:
                col = map(converters[colName], col)
            dataDict[colName] = col

        # Using pandas for record-array generation
        return nptools.recarray(dataDict, dtype=npDtypes)

    def importShape(
            self,
            fileName,
            epsg,
            schema='public',
            bulk=500,
            outTable=None):
        fileInfo = os.path.splitext(os.path.basename(fileName))
        extension = fileInfo[1]
        if outTable is None:
            outTable = fileInfo[0]
        assert extension == '.shp', 'File extension has to be ".shp", not "%s"' % extension

        query = 'DROP TABLE IF EXISTS "%s"."%s" CASCADE;' % (schema, outTable)
        self(query)

        command = 'shp2pgsql -k -I -s %i %s "%s"."%s"  ' % (
            epsg, fileName, schema, outTable)
        print(command)
        query = shell(command)
        print('insert into DB')
        self(query, schema=schema, bulk=bulk)

    def importMdb(self, fileName, outTable, schema='public', bulk=500):
        extension = os.path.splitext(os.path.basename(fileName))[1]
        assert extension == '.mdb', 'File extension has to be ".mdb", not "%s"' % extension

        query = 'DROP TABLE IF EXISTS "%s"."%s" CASCADE;' % (schema, outTable)
        print(query)
        self(query)

        command = 'mdb-schema "%s" -T %s postgres' % (fileName, outTable)
        print(command)
        query = shell(command)
        print('insert DB')
        self(query, bulk=bulk, schema=schema)

        command = 'mdb-export -I postgres -q "\'" %s %s' % (fileName, outTable)
        print(command)
        query = shell(command)
        print('insert DB')
        self(query, bulk=bulk, schema=schema)


def shell(command):
    return subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE).stdout.read()
