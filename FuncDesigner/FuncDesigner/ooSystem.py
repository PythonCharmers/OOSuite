from misc import FuncDesignerException
from ooFun import oofun


class ooSystem(set):
    def __init__(self, *args,  **kwargs):
        assert len(kwargs) == 0, 'ooSystem constructor has no implemented kwargs yet'
        for arg in args:
            if isinstance(arg, set):
               self.update(arg)
            elif isinstance(arg, list) or isinstance(arg, tuple):
                self.update(set(arg))
            elif isinstance(arg, oofun):
                self.add(arg)
            else:
                raise FuncDesignerException('Incorrect type %s in ooSystem constructor' % type(arg))
    
    # [] yields result wrt allowed contol:
    #__getitem__ = lambda self, point: ooSystemState([(elem, elem[point]) for elem in self])
    def __getitem__(self, item):
        raise FuncDesignerException('ooSystem __getitem__ is reserved for future purposes')
    
    # () yields exact result (ignoring contol):
    def __call__(self, point):
        assert isinstance(point,  dict), 'argument should be Python dictionary'
        return ooSystemState([(elem, elem[point]) for elem in self])
#        r = []
#        for key, val in tmp.items():
#            r.append((key, key(point, contol=0.0))) if key.isConstraint else r.append((key, val))
#        return ooSystemState(r)
        
        
        
####################### ooSystemState #########################
class ooSystemState(dict):
    _byNames = None
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        if self._byNames is None:
            self._byNames = dict([(key.name, val) for key, val in self.items()])
        #self.update(self._byNames)
    
    __repr__ = lambda self: ''.join(['\n'+key+'='+str(val) for key, val in self._byNames.items()])[1:]
    
    def __call__(self, *args,  **kwargs):
        assert len(kwargs) == 0, "ooSystemState method '__call__' has no implemented kwargs yet"
        r = [(self._byNames[arg] if isinstance(arg,  str) else self[arg]) for arg in args]
        return r[0] if len(r)==1 else r
    
    #__call__ = _call
    #__getitem__ = _call
        # TODO: implement it
        # access by elements and their names
















