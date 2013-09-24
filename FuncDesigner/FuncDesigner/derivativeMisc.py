from FDmisc import Len, FuncDesignerException, DiagonalType, scipyAbsentMsg, pWarn, scipyInstalled, Diag, \
Copy

from baseClasses import OOFun
from numpy import isscalar, ndarray, atleast_2d, prod, int64
from multiarray import multiarray

try:
    from DerApproximator import get_d1#, check_d1
    DerApproximatorIsInstalled = True
except:
    DerApproximatorIsInstalled = False
    

def getDerivativeSelf(Self, x, fixedVarsScheduleID, Vars,  fixedVars):
    Input = Self._getInput(x, fixedVarsScheduleID=fixedVarsScheduleID, Vars=Vars,  fixedVars=fixedVars)
    expectedTotalInputLength = sum([Len(elem) for elem in Input])
    
#        if hasattr(Self, 'size') and isscalar(Self.size): nOutput = Self.size
#        else: nOutput = Self(x).size 

    hasUserSuppliedDerivative = Self.d is not None
    if hasUserSuppliedDerivative:
        derivativeSelf = []
        if type(Self.d) == tuple:
            if len(Self.d) != len(Self.input):
               raise FuncDesignerException('oofun error: num(derivatives) not equal to neither 1 nor num(inputs)')
               
            for i, deriv in enumerate(Self.d):
                inp = Self.input[i]
                if not isinstance(inp, OOFun) or inp.discrete: 
                    #if deriv is not None: 
                        #raise FuncDesignerException('For an oofun with some input oofuns declared as discrete you have to set oofun.d[i] = None')
                    continue
                
                #!!!!!!!!! TODO: handle fixed cases properly!!!!!!!!!!!!
                #if hasattr(inp, 'fixed') and inp.fixed: continue
                if inp.is_oovar and ((Vars is not None and inp not in Vars) or (fixedVars is not None and inp in fixedVars)):
                    continue
                    
                if deriv is None:
                    if not DerApproximatorIsInstalled:
                        raise FuncDesignerException('To perform gradients check you should have DerApproximator installed, see http://openopt.org/DerApproximator')
                    derivativeSelf.append(get_d1(Self.fun, Input, diffInt=Self.diffInt, stencil = Self.stencil, \
                                                 args=Self.args, varForDifferentiation = i, pointVal = Self._getFuncCalcEngine(x), exactShape = True))
                else:
                    # !!!!!!!!!!!!!! TODO: add check for user-supplied derivative shape
                    tmp = deriv(*Input)
                    if not isscalar(tmp) and type(tmp) in (ndarray, tuple, list) and type(tmp) != DiagonalType: # i.e. not a scipy.sparse matrix
                        tmp = atleast_2d(tmp)
                        
                        ########################################

                        _tmp = Input[i]
                        Tmp = 1 if isscalar(_tmp) or prod(_tmp.shape) == 1 else len(Input[i])
                        if tmp.shape[1] != Tmp: 
                            # TODO: add debug msg
#                                print('incorrect shape in FD AD _getDerivativeSelf')
#                                print tmp.shape[0], nOutput, tmp
                            if tmp.shape[0] != Tmp: raise FuncDesignerException('error in getDerivativeSelf()')
                            tmp = tmp.T
                                
                        ########################################

                    derivativeSelf.append(tmp)
        else:
            tmp = Self.d(*Input)
            if not isscalar(tmp) and type(tmp) in (ndarray, tuple, list): # i.e. not a scipy.sparse matrix
                tmp = atleast_2d(tmp)
                
                if tmp.shape[1] != expectedTotalInputLength: 
                    # TODO: add debug msg
                    if tmp.shape[0] != expectedTotalInputLength: raise FuncDesignerException('error in getDerivativeSelf()')
                    tmp = tmp.T
                    
            ac = 0
            if isinstance(tmp, ndarray) and hasattr(tmp, 'toarray') and not isinstance(tmp, multiarray): tmp = tmp.A # is dense matrix
            
            #if not isinstance(tmp, ndarray) and not isscalar(tmp) and type(tmp) != DiagonalType:
            if len(Input) == 1:
#                    if type(tmp) == DiagonalType: 
#                            # TODO: mb rework it
#                            if Input[0].size > 150 and tmp.size > 150:
#                                tmp = tmp.resolve(True).tocsc()
#                            else: tmp =  tmp.resolve(False) 
                derivativeSelf = [tmp]
            else:
                for i, inp in enumerate(Input):
                    t = Self.input[i]
                    if t.discrete or (t.is_oovar and ((Vars is not None and t not in Vars) or (fixedVars is not None and t in fixedVars))):
                        ac += inp.size
                        continue                                    
                    if isinstance(tmp, ndarray):
                        TMP = tmp[:, ac:ac+Len(inp)]
                    elif isscalar(tmp):
                        TMP = tmp
                    elif type(tmp) == DiagonalType: 
                        if tmp.size == inp.size and ac == 0:
                            TMP = tmp
                        else:
                            # print debug warning here
                            # TODO: mb rework it
                            if inp.size > 150 and tmp.size > 150:
                                tmp = tmp.resolve(True).tocsc()
                            else: tmp =  tmp.resolve(False) 
                            TMP = tmp[:, ac:ac+inp.size]
                    else: # scipy.sparse matrix
                        TMP = tmp.tocsc()[:, ac:ac+inp.size]
                    ac += Len(inp)
                    derivativeSelf.append(TMP)
                
        # TODO: is it required?
#                if not hasattr(Self, 'outputTotalLength'): Self(x)
#                
#                if derivativeSelf.shape != (Self.outputTotalLength, Self.inputTotalLength):
#                    s = 'incorrect shape for user-supplied derivative of oofun '+Self.name+': '
#                    s += '(%d, %d) expected, (%d, %d) obtained' % (Self.outputTotalLength, Self.inputTotalLength,  derivativeSelf.shape[0], derivativeSelf.shape[1])
#                    raise FuncDesignerException(s)
    else:
        if Vars is not None or fixedVars is not None: raise FuncDesignerException("sorry, custom oofun derivatives don't work with Vars/fixedVars arguments yet")
        if not DerApproximatorIsInstalled:
            raise FuncDesignerException('To perform this operation you should have DerApproximator installed, see http://openopt.org/DerApproximator')
            
        derivativeSelf = get_d1(Self.fun, Input, diffInt=Self.diffInt, stencil = Self.stencil, args=Self.args, pointVal = Self._getFuncCalcEngine(x), exactShape = True)
        if type(derivativeSelf) == tuple:
            derivativeSelf = list(derivativeSelf)
        elif type(derivativeSelf) != list:
            derivativeSelf = [derivativeSelf]
    
    #assert all([elem.ndim > 1 for elem in derivativeSelf])
   # assert len(derivativeSelf[0])!=16
    #assert (type(derivativeSelf[0]) in (int, float)) or derivativeSelf[0][0]>480.00006752 or derivativeSelf[0][0]<480.00006750
    return derivativeSelf


def considerSparse(t1, t2):  
    # TODO: handle 2**15 & 0.25 as parameters
    if int64(prod(t1.shape)) * int64(prod(t2.shape)) > 2**15 \
    and ((isinstance(t1, ndarray) and t1.nonzero()[0].size < 0.25*t1.size) or \
    (isinstance(t2, ndarray) and t2.nonzero()[0].size < 0.25*t2.size)):
        if not scipyInstalled: 
            pWarn(scipyAbsentMsg)
            return t1,  t2
            
        from scipy.sparse import csc_matrix, csr_matrix
        if not isinstance(t1, csc_matrix): 
            t1 = csc_matrix(t1)
        if t1.shape[1] != t2.shape[0]: # can be from flattered t1
            assert t1.shape[0] == t2.shape[0], 'bug in FuncDesigner Kernel, inform developers'
            t1 = t1.T
        if not isinstance(t2, csr_matrix): 
            t2 = csr_matrix(t2)
            
    return t1, t2

def mul_aux_d(x, y):
    Xsize, Ysize = Len(x), Len(y)
    if Xsize == 1:
        return Copy(y)
    elif Ysize == 1:
        return Diag(None, scalarMultiplier = y, size = Xsize)
    elif Xsize == Ysize:
        return Diag(y)
    else:
        raise FuncDesignerException('for oofun multiplication a*b should be size(a)=size(b) or size(a)=1 or size(b)=1')   


