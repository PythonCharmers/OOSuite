from misc import FuncDesignerException
from ooFun import oofun, BaseFDConstraint
from numpy import nan, zeros, isscalar

class sle:
    # System of linear equations
    def __init__(self, equations):
        if type(equations) not in [list, tuple]:
            raise FuncDesignerException('argument of sle constructor should be Python tuple or list of equations or oofuns')
        self.equations = equations
        
    def solve(self, *args, **kwargsForOpenOptSLEconstructor):
        try:
            from openopt import SLE
        except:
            s = "Currently to solve SLEs via FuncDesigner you should have OpenOpt installed; maybe in future the dependence will be ceased"
            raise FuncDesignerException(s)
            
        assert len(args) <= 1, 'incorrect lse definition, no more than 1 arg expected'
        if len(args)>0:
            startPoint = args[0]
        else:
            involvedOOVars = set()
            for Elem in self.equations:
                elem = Elem.oofun if Elem.isConstraint else Elem
                if elem.is_oovar:
                    involvedOOVars.add(elem)
                else:
                    involvedOOVars.update(elem._getDep())
            startPoint = {}
            for oov in involvedOOVars:
                if isscalar(oov.size):
                    startPoint[oov] = zeros(oov.size)
                else:
                    startPoint[oov] = 0
        if 'iprint' not in kwargsForOpenOptSLEconstructor.keys():
            kwargsForOpenOptSLEconstructor['iprint'] = -1
        p = SLE(self.equations, startPoint, **kwargsForOpenOptSLEconstructor)
        r = p.solve()
        if r.istop >= 0:
            return r.xf
        else:
            R = {}
            for key, value in startPoint.items(): 
                R[key] = value * nan
            return R
            
#        Z = self._vector2point(zeros(self.n))
#        for c in self.constraints:
#            f = c.oofun
#            dep = f._getDep()
#            if dep is None: # hence it's oovar
#                assert f.is_oovar
#                dep = set([f])
#
#            if f.is_linear:
#                Aeq.append(self._pointDerivative2array(f._D(Z, **D_kwargs)))      
#                beq.append(-f(Z))
                
