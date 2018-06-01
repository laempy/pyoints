#import json
import os
import numpy as np
from configobj import ConfigObj
import yaml


class Config(ConfigObj):

    def __init__(self, configFiles=[], data={}):

        if isinstance(configFiles, str):
            ConfigObj.__init__(self, configFiles, list_values=False)
            configFiles = []
        else:
            ConfigObj.__init__(self)
            try:
                configFiles = list(configFiles)
            except TypeError:
                raise IOError("Wrong 'configFiles' type!")

        for configFile in configFiles[::-1]:
            if os.path.isfile(configFile):
                conf = Config(configFile)
                self.update(conf)
            else:
                raise IOError('File %s not found' % configFile)

        self.update(data)

    def parse(self, converterDict):

        # converterDict={
        #    'Section1':{
        #        'Option1': int, # Integer value
        #        'Option2': float # Float value
        #    },
        #    'Section2':{
        #        'Option1': list, # Array of Integers
        #        'Option2': converter # Array of Floats
        #    }
        #}
        def recursiveParse(dictA, dictB):
            for key, value in dictB.iteritems():
                if isinstance(value, dict):
                    recursiveParse(dictA[key], value)
                else:
                    dictA[key] = value(dictA[key])
            return dictA

        return Config(data=recursiveParse(self.dict(), converterDict))

    # def __getitem__(self,*keys):
        # keys=keys[0]
        # if isinstance(keys,str):
        #    data=ConfigObj.__getitem__(self,keys)
        # else:
        #    data=ConfigObj.__getitem__(self,keys[0])
        #    for key in keys[1:]:
        #        data=data[key]

        # if isinstance(data,str):
        #    return data
        # else:
        #    return Config(data=data)

    def __getitem__(self, key):
        data = ConfigObj.__getitem__(self, key)
        if isinstance(data, str):
            return data
        else:
            return Config(data=data)

    def hasItem(self, *keys):
        try:
            self.str(*keys)
        except KeyError:
            return False
        return True

    def str(self, *keys):
        res = self
        for key in keys:
            res = res[key]
        return res

    # Implement ConfigParser methods
    def float(self, *keys):
        return float(self.str(*keys))

    def int(self, *keys):
        return int(self.str(*keys))

    def bool(self, *keys):
        value = self.str(*keys)
        if value in ['True', 'true', 'TRUE']:
            return True
        elif value in ['False', 'false', 'FALSE']:
            return False
        else:
            raise ValueError('Value "%s" is not boolean!' % value)

    def inFile(self, *keys):
        fileName = self.str(*keys)
        if not os.path.isfile(fileName):
            raise OSError('File "%s" does not exist.' % fileName)
        return fileName

    def outFile(self, *keys):
        fileName = self.str(*keys)
        path = os.path.dirname(fileName)
        if not os.path.isdir(path):
            raise OSError('Directory of %s does not exist.' % fileName)
        return fileName

    def path(self, *keys):
        pathName = self.str(*keys)
        if not os.path.isdir(pathName):
            raise OSError('Path %s not found.' % pathName)
        return pathName

    def array(self, *keys, **args):
        values = self.str(*keys)
        if values is '':
            return None
        # return np.array(json.loads(values),**args)
        return np.array(yaml.safe_load(values), **args)

    def dict(self, *keys):
        values = self.str(*keys)
        if values is '':
            return {}
        return yaml.safe_load(values)

    def subset(self, *keys):
        d = self.str(*keys)
        return Config(data=d)

    def update(self, dictionary):
        for key, value in dictionary.iteritems():
            if key in self and isinstance(value, dict):
                value.update(self[key])
                # self[key].update(value)
            self[key] = value
