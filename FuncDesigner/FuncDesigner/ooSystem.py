from misc import FuncDesignerException
from ooFun import oofun, BaseFDConstraint, _getAllAttachedConstraints
from ooPoint import ooPoint
from numpy import isnan, ndarray, isfinite

class ooSystem:
    def __init__(self, *args,  **kwargs):
        assert len(kwargs) == 0, 'ooSystem constructor has no implemented kwargs yet'
        self.items = set()
        self.constraints = set()
        for arg in args:
            if isinstance(arg, set):
               self.items.update(arg)
            elif isinstance(arg, list) or isinstance(arg, tuple):
                self.items.update(set(arg))
            elif isinstance(arg, oofun):
                self.items.add(arg)
            else:
                raise FuncDesignerException('Incorrect type %s in ooSystem constructor' % type(arg))
    
    # [] yields result wrt allowed contol:
    #__getitem__ = lambda self, point: ooSystemState([(elem, elem[point]) for elem in self])
    def __getitem__(self, item):
        raise FuncDesignerException('ooSystem __getitem__ is reserved for future purposes')
    
    def __iadd__(self, *args, **kwargs):
        assert len(kwargs) == 0, 'not implemented yet'
        if type(args[0]) in [list, tuple, set]:
            assert len(args) == 1
            Args = args[0]
        else:
            Args = args
        for elem in Args:
            assert isinstance(elem, oofun), 'ooSystem operation += expects only oofuns'
        self.items.update(set(Args))
        return self
        
    def __iand__(self, *args, **kwargs):
        assert len(kwargs) == 0, 'not implemented yet'
        if type(args[0]) in [list, tuple, set]:
            assert len(args) == 1
            Args = args[0]
        else:
            Args = args
        for elem in Args:
            assert isinstance(elem, BaseFDConstraint), 'ooSystem operation &= expects only FuncDesigner constraints'
        self.constraints.update(set(Args))
        return self
    
        
    # TODO: provide a possibility to yield exact result (ignoring contol)    
    def __call__(self, point):
        assert isinstance(point,  dict), 'argument should be Python dictionary'
        if not isinstance(point, ooPoint):
            point = ooPoint(point)
        r = ooSystemState([(elem, simplify(elem(point))) for elem in self.items])

        # handling constraints
        #!!!!!!!!!!!!!!!!!!! TODO: perform attached constraints lookup only once if ooSystem wasn't modified by += or &= etc
        
        cons = self.constraints
        cons.update(_getAllAttachedConstraints(self.items | self.constraints))
        
        activeConstraints = []
        allAreFinite = all([isfinite(elem(point)) for elem in self.items])
        
        for c in cons:
            val = c.oofun(point)
            if c(point) is False or any(isnan(val)): 
                activeConstraints.append(c)
                #_activeConstraints.append([c, val, max((val-c.ub, c.lb-val)), c.tol])
                
        r.isFeasible = True if len(activeConstraints) == 0 and allAreFinite else  False
        #r._activeConstraints = activeConstraints
        r.activeConstraints = activeConstraints
        return r
        
        
#        r = []
#        for key, val in tmp.items():
#            r.append((key, key(point, contol=0.0))) if key.isConstraint else r.append((key, val))
#        return ooSystemState(r)
        
        
        
####################### ooSystemState #########################
class ooSystemState:
    def __init__(self, keysAndValues, *args, **kwargs):
        assert len(args) ==0
        assert len(kwargs) ==0
        #dict.__init__(self, *args, **kwargs)
        self._byID = dict([(key, val) for key, val in  keysAndValues])
        self._byNames = dict([(key.name, val) for key, val in keysAndValues])
        #self.update(self._byNames)
    
    __repr__ = lambda self: ''.join(['\n'+key+' = '+str(val) for key, val in self._byNames.items()])[1:]
    
    def __call__(self, *args,  **kwargs):
        assert len(kwargs) == 0, "ooSystemState method '__call__' has no implemented kwargs yet"
        r = [(self._byNames[arg] if isinstance(arg,  str) else self._byID[arg]) for arg in args]
        return r[0] if len(r)==1 else r
    
    #__call__ = _call
    #__getitem__ = _call
        # TODO: implement it
        # access by elements and their names


simplify = lambda val: val[0] if isinstance(val, ndarray) and val.size == 1 else val













