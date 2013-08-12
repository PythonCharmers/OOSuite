from .misc import SpaceFuncsException
class baseGeometryObject:
    _AttributesDict = {'spaceDimension': '_spaceDimension'}
    
    def __init__(self, *args, **kw):
        self.__dict__.update(kw)

    #__call__ = lambda self, *args, **kwargs: ChangeName(self, args[0]) #if len(args) == 1 and type(args[0]) == str else tmp(*args, **kwargs)
        
    def plot(self, *args, **kw): # should be overwritten by derived classes
        raise SpaceFuncsException('plotting for the object is not implemented yet')
        
    def __getattr__(self, attr):
        if attr in self._AttributesDict:
            tmp = getattr(self, self._AttributesDict[attr])()
            setattr(self, attr, tmp)
            return tmp
        else:
            raise AttributeError('no such method "%s" for the object' % attr)
    
    def _spaceDimension(self):
        raise SpaceFuncsException('this function should be overwritten by derived class')

