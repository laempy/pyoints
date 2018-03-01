import os

    
class GeoHandler:
    
    @property
    def proj(self):
        raise NotImplementedError()
        
    @property
    def extent(self):
        raise NotImplementedError()
        
    @property
    def corners(self):
        raise NotImplementedError()
    
    @property
    def date(self):
        return None
    
    def __len__():
        raise NotImplementedError()
    
    def load(self,extent=None):
        raise NotImplementedError()
    
    def cleanCache(self):
        raise NotImplementedError()
    
    
class GeoFile(GeoHandler):
    
    def __init__(self,file):
        if not os.path.isfile(file):
            raise IOError('File "%s" Not Found'%file)
        self.fileName,self.extension=os.path.splitext(os.path.basename(file))
        self.extension=self.extension[1:]
        self.path=os.path.dirname(file)
        self.file=os.path.abspath(file)
    