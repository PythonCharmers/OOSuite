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
    __getitem__ = lambda self, point: ooSystemState([(elem, elem(point)) for elem in self])
    
    # () yields exact result (ignoring contol):
    def __call__(self, point):
        assert isinstance(point,  dict), 'argument should be Python dictionary'
        tmp = self.__getitem__(point)
        r = []
        for key, val in tmp.items():
            r.append((key, key(point, contol=0.0))) if key.isConstraint else r.append((key, val))
        return ooSystemState(r)
        
        
        
####################### ooSystemState #########################
class ooSystemState(dict):
    __repr__ = lambda self: ''.join(['\n'+str(key)+'='+str(val) for key, val in self.items()])[1:]
    def __call__(self, *args,  **kwargs):
        assert len(kwargs) == 0, "ooSystemState method '__call__' has no implemented kwargs yet"
        # TODO: implement it
        # access by elements and thier names















