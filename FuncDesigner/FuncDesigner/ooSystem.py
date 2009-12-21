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
        
    __call__ = lambda self, point: ooSystemState([(elem, elem(point)) for elem in self])
        
class ooSystemState(dict):
   
    def __repr__(self):
        r = [str(key)+'='+str(val)+'\n' for key, val in self.items()]
        return ''.join(r)[:-1]















