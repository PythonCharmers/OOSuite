__docformat__ = "restructuredtext en"
from numpy import *
from setDefaultIterFuncs import USER_DEMAND_EXIT
from ooMisc import killThread, setNonLinFuncsNumber
from nonOptMisc import scipyInstalled, Vstack, isspmatrix
try:
    from DerApproximator import get_d1
    DerApproximatorIsInstalled = True
except:
    DerApproximatorIsInstalled = False


class nonLinFuncs:
    def __init__(self): pass

    def wrapped_func(p, x, IND, userFunctionType, ignorePrev, getDerivative):
        if isinstance(x, dict):
            if not p.isFDmodel: p.err('calling the function with argument of type dict is allowed for FuncDesigner models only')
            x = p._point2vector(x)
        if not getattr(p.userProvided, userFunctionType): return array([])
        if p.istop == USER_DEMAND_EXIT:
            if p.solver.__cannotHandleExceptions__:
                return nan
            else:
                raise killThread
                
        if getDerivative and not p.isFDmodel and not DerApproximatorIsInstalled:
            p.err('To perform gradients check you should have DerApproximator installed, see http://openopt.org/DerApproximator')

        #userFunctionType should be 'f', 'c', 'h'
        funcs = getattr(p.user, userFunctionType)
        funcs_num = getattr(p, 'n'+userFunctionType)
        if IND is not None:
            ind = p.getCorrectInd(IND)
        else: ind = None

        # this line had been added because some solvers pass tuple instead of
        # x being vector p.n x 1 or matrix X=[x1 x2 x3...xk], size(X)=[p.n, k]
        if not isspmatrix(x): 
            x = asfarray(x)
        
        if not ignorePrev: 
            prevKey = p.prevVal[userFunctionType]['key']
        else:
            prevKey = None
            
        # TODO: move it into runprobsolver or baseproblem
        if p.prevVal[userFunctionType]['val'] is None:
            p.prevVal[userFunctionType]['val'] = zeros(getattr(p, 'n'+userFunctionType))
            
        if prevKey is not None and p.iter > 0 and array_equal(x,  prevKey) and ind is None and not ignorePrev:
            #TODO: add counter of the situations
            if not getDerivative:
                r = copy(p.prevVal[userFunctionType]['val'])
                #if p.debug: assert array_equal(r,  p.wrapped_func(x, IND, userFunctionType, True, getDerivative))
                if ind is not None: r = r[ind]

                if userFunctionType == 'f':
                    if p.isObjFunValueASingleNumber: r = r.sum(0)
                    if p.invertObjFunc: r = -r
                    if  p.solver.__funcForIterFcnConnection__=='f' and any(isnan(x)):
                        p.nEvals['f'] += 1
                        if p.nEvals['f']%p.f_iter == 0:
                            p.iterfcn(x, fk = r)
                return r

        args = getattr(p.args, userFunctionType)

        # TODO: handle it in prob prepare
        if not hasattr(p, 'n'+userFunctionType): setNonLinFuncsNumber(p,  userFunctionType)

        if ind is None:
            nFuncsToObtain = getattr(p, 'n'+ userFunctionType)
        else:
            nFuncsToObtain = len(ind)

        if x.shape[0] != p.n and (x.ndim<2 or x.shape[1] != p.n): 
            p.err('x with incorrect shape passed to non-linear function')

        #TODO: code cleanup (below)
        if getDerivative or x.ndim <= 1 or x.shape[0] == 1:
            nXvectors = 1
            x_0 = copy(x)
        else:
            nXvectors = x.shape[0]

        # TODO: mb replace D by _D?
        if getDerivative and p.isFDmodel:
            if p.optVars is None or (p.fixedVars is not None and len(p.optVars) < len(p.fixedVars)):
                funcs2 = [(lambda x, i=i: \
                  p._pointDerivative2array(funcs[i].D(x, Vars = p.optVars, resultKeysType = 'names', asSparse='auto'), asSparse='auto')) \
                  for i in xrange(len(funcs))]
            else:
                funcs2 = [(lambda x, i=i: \
                  p._pointDerivative2array(funcs[i].D(x, fixedVars = p.fixedVars, resultKeysType = 'names', asSparse='auto'), asSparse='auto', func=funcs[i], point=x)) \
                  for i in xrange(len(funcs))]
        else:
            funcs2 = funcs
            
        if ind is None: 
            Funcs = funcs2
        elif ind is not None and p.functype[userFunctionType] == 'some funcs R^nvars -> R':
            Funcs = [funcs2[i] for i in ind]
        else:
            Funcs = getFuncsAndExtractIndexes(p, funcs2, ind, userFunctionType)

        agregate_counter = 0
        
        if p.isFDmodel:
            Args = ()
        else:
            Args = args
            
        if nXvectors == 1:
            if p.isFDmodel:
                X = p._vector2point(x) 
            else:
                X = x
        
        if nXvectors > 1: # and hence getDerivative isn't involved
            #temporary, to be fixed
            assert userFunctionType == 'f' and p.isObjFunValueASingleNumber
            if p.isFDmodel:
                X = [p._vector2point(x[i]) for i in xrange(nXvectors)]
            elif len(Args) == 0:
                X = [x[i] for i in xrange(nXvectors)]
            else:
                X = [((x[i],) + Args) for i in xrange(nXvectors)]
            r = hstack([map(fun, X) for fun in Funcs]).reshape(1, -1)
        elif not getDerivative:
            r = hstack([fun(*(X, )+Args) for fun in Funcs])
            if not ignorePrev and ind is None:
                p.prevVal[userFunctionType]['key'] = copy(x_0)
                p.prevVal[userFunctionType]['val'] = r.copy()                
        elif getDerivative and p.isFDmodel:
            #raise 0
            rr = [fun(X) for fun in Funcs]
            r = Vstack(rr) if scipyInstalled and any([isspmatrix(elem) for elem in rr]) else vstack(rr)
            #assert prod(r.shape) != 177878
        else:
            if getDerivative:
                r = zeros((nFuncsToObtain, p.n))
                diffInt = p.diffInt
                abs_x = abs(x)
                finiteDiffNumbers = 1e-10 * abs_x
                if p.diffInt.size == 1:
                    finiteDiffNumbers[finiteDiffNumbers < diffInt] = diffInt
                else:
                    finiteDiffNumbers[finiteDiffNumbers < diffInt] = diffInt[finiteDiffNumbers < diffInt]
            else:
                r = zeros((nFuncsToObtain, nXvectors))
            
            for index, fun in enumerate(Funcs):
                v = ravel(fun(*((X,) + Args)))
                if  (ind is None or funcs_num == 1) and not ignorePrev:
                    #TODO: ADD COUNTER OF THE CASE
                    if index == 0: p.prevVal[userFunctionType]['key'] = copy(x_0)
                    p.prevVal[userFunctionType]['val'][agregate_counter:agregate_counter+v.size] = v.copy()                
                r[agregate_counter:agregate_counter+v.size,0] = v

                """                                                 geting derivatives                                                 """
                if getDerivative:
                    func = lambda x: fun(*((x,) + Args))
                    d1 = get_d1(func, x, pointVal = None, diffInt = finiteDiffNumbers, stencil=p.JacobianApproximationStencil)
                    r[agregate_counter:agregate_counter+d1.size] = d1
                    
                agregate_counter += atleast_1d(asarray(v)).shape[0]

        #assert p.iter != 176 or userFunctionType != 'f' or not getDerivative
        if userFunctionType == 'f' and p.isObjFunValueASingleNumber and r.size > 1 and (type(r) == ndarray or min(r.shape) > 1): 
            r = r.sum(0)

#        if type(r) == matrix: 
#            raise 0
#            r = r.A # if _dense_numpy_matrix !
        #assert p.iter != 176 or userFunctionType != 'f' or not getDerivative
        if nXvectors == 1 and (not getDerivative or r.size == 1): #if min(r.shape) == 1:
            r = r.flatten() if type(r) == ndarray else r.toarray().flatten()

        if p.invertObjFunc and userFunctionType=='f':
            r = -r

        if ind is None:
            p.nEvals[userFunctionType] += nXvectors
        else:
            p.nEvals[userFunctionType] = p.nEvals[userFunctionType] + float(nXvectors * len(ind)) / getattr(p, 'n'+ userFunctionType)

        if getDerivative:
            assert x.size == p.n#TODO: add python list possibility here
            x = x_0 # for to suppress numerical instability effects while x +/- delta_x

        if userFunctionType == 'f' and hasattr(p, 'solver') and p.solver.__funcForIterFcnConnection__=='f' and hasattr(p, 'f_iter'):
            if p.nEvals['f']%p.f_iter == 0:
                p.iterfcn(x, r)
        return r




    def wrapped_1st_derivatives(p, x, ind_, funcType, ignorePrev, asSparse):
        if isinstance(x, dict):
            if not p.isFDmodel: p.err('calling the function with argument of type dict is allowed for FuncDesigner models only')
            if ind_ is not None:p.err('the operation is turned off for argument of type dict when ind!=None')
            x = p._point2vector(x)
        if ind_ is not None:
            ind = p.getCorrectInd(ind_)
        else: ind = None

        if p.istop == USER_DEMAND_EXIT:
            if p.solver.__cannotHandleExceptions__:
                return nan
            else:
                raise killThread
        derivativesType = 'd'+ funcType
        prevKey = p.prevVal[derivativesType]['key']
        if prevKey is not None and p.iter > 0 and array_equal(x, prevKey) and ind is None and not ignorePrev:
            #TODO: add counter of the situations
            assert p.prevVal[derivativesType]['val'] is not None
            return copy(p.prevVal[derivativesType]['val'])

        if ind is None and not ignorePrev: p.prevVal[derivativesType]['ind'] = copy(x)

        #TODO: patterns!
        nFuncs = getattr(p, 'n'+funcType)
        if not getattr(p.userProvided, derivativesType):
            #                                            x, IND, userFunctionType, ignorePrev, getDerivative
            derivatives = p.wrapped_func(x, ind, funcType, True, True)
            if ind is None:
                p.nEvals[derivativesType] -= 1
            else:
                p.nEvals[derivativesType] = p.nEvals[derivativesType] - float(len(ind)) / nFuncs
        else:
                
            funcs = getattr(p.user, derivativesType)
            
            if ind is None or (nFuncs == 1 and p.functype[funcType] == 'single func'): 
                Funcs = funcs
            elif ind is not None and p.functype[funcType] == 'some funcs R^nvars -> R':
                Funcs = [funcs[i] for i in ind]
            else:
                Funcs = getFuncsAndExtractIndexes(p, funcs, ind, funcType)
            
            if ind is None: derivativesNumber = nFuncs
            else: derivativesNumber = len(ind)
                
            #derivatives = empty((derivativesNumber, p.n))
            derivatives = []
            #agregate_counter = 0
            for fun in Funcs:#getattr(p.user, derivativesType):
                tmp = atleast_1d(fun(*(x,)+getattr(p.args, funcType)))
                # TODO: replace tmp.size here for sparse matrices
                if mod(tmp.size, p.n) != 0:
                    if funcType=='f':
                        p.err('incorrect user-supplied (sub)gradient size of objective function')
                    elif funcType=='c':
                        p.err('incorrect user-supplied (sub)gradient size of non-lin inequality constraints')
                    elif funcType=='h':
                        p.err('incorrect user-supplied (sub)gradient size of non-lin equality constraints')
                
                if tmp.ndim == 1: m= 1
                else: m = tmp.shape[0]
                if p.functype[funcType] == 'some funcs R^nvars -> R' and m != 1:
                    # TODO: more exact check according to stored p.arr_of_indexes_* arrays
                    p.err('incorrect shape of user-supplied derivative, it should be in accordance with user-provided func size')
                derivatives.append(tmp)
                #derivatives[agregate_counter : agregate_counter + m] =  tmp#.reshape(tmp.size/p.n,p.n)
                #agregate_counter += m
            #TODO: inline ind modification!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            derivatives = Vstack(derivatives) if any(isspmatrix(derivatives)) else vstack(derivatives)

            if ind is None:
                p.nEvals[derivativesType] += 1
            else:
                #derivatives = derivatives[ind]
                p.nEvals[derivativesType] = p.nEvals[derivativesType] + float(len(ind)) / nFuncs

            if funcType=='f':
                if p.invertObjFunc: derivatives = -derivatives
                if p.isObjFunValueASingleNumber: 
                    if not isinstance(derivatives, ndarray): derivatives = derivatives.toarray()
                    derivatives = derivatives.flatten()
        if asSparse is False or not scipyInstalled or not hasattr(p, 'solver') or not p.solver._canHandleScipySparse: 
            # p can has no attr 'solver' if it is called from checkdf, checkdc, checkdh
            if not isinstance(derivatives, ndarray): 
                derivatives = derivatives.toarray()
        if min(derivatives.shape) == 1: 
            if isspmatrix(derivatives): derivatives = derivatives.A
            derivatives = derivatives.flatten()
        if ind is None and not ignorePrev: p.prevVal[derivativesType]['val'] = derivatives

        if funcType=='f':
            if hasattr(p, 'solver') and not p.solver.__iterfcnConnected__  and p.solver.__funcForIterFcnConnection__=='df':
                if p.df_iter is True: p.iterfcn(x)
                elif p.nEvals[derivativesType]%p.df_iter == 0: p.iterfcn(x) # call iterfcn each {p.df_iter}-th df call

        return derivatives


    # the funcs below are not implemented properly yet
    def user_d2f(p, x):
        assert x.ndim == 1
        p.nEvals['d2f'] += 1
        assert(len(p.user.d2f)==1)
        r = p.user.d2f[0](*(x, )+p.args.f)
        if p.invertObjFunc:# and userFunctionType=='f': 
            r = -r
        return r

    def user_d2c(p, x):
        return ()

    def user_d2h(p, x):
        return ()

    def user_l(p, x):
        return ()

    def user_dl(p, x):
        return ()

    def user_d2l(p, x):
        return ()

    def getCorrectInd(p, ind):
        if ind is None or type(ind) in [list, tuple]:
            result = ind
        else:
            try:
                result = atleast_1d(ind).tolist()
            except:
                raise ValueError('%s is an unknown func index type!'%type(ind))
        return result

def getFuncsAndExtractIndexes(p, funcs, ind, userFunctionType):
    if ind is None: return funcs
    if len(funcs) == 1 : 
        def f (*args, **kwargs): 
            tmp = funcs[0](*args, **kwargs)
            if isspmatrix(tmp):
                tmp = tmp.tocsc()
            elif not isinstance(tmp,  ndarray):
                tmp = atleast_1d(tmp)
            return tmp[ind]
        return [f]
    
    #getting number of block and shift
    arr_of_indexes = getattr(p, 'arr_of_indexes_' + userFunctionType)
    left_arr_indexes = searchsorted(arr_of_indexes, ind) 
    
    indLenght = len(ind)
    
    Funcs2 = []
    # TODO: try to get rid of cycles, use vectorization instead
    IndDict = {}
    for i in xrange(indLenght):
        
        if left_arr_indexes[i] != 0:
            num_of_funcs_before_arr_left_border = arr_of_indexes[left_arr_indexes[i]-1]
            inner_ind = ind[i] - num_of_funcs_before_arr_left_border - 1
        else:
            inner_ind = ind[i]
            
        if left_arr_indexes[i] in IndDict.keys():
            IndDict[left_arr_indexes[i]].append(inner_ind)
        else:
            IndDict[left_arr_indexes[i]] = [inner_ind]
            Funcs2.append([funcs[left_arr_indexes[i]], IndDict[left_arr_indexes[i]]])
    
    Funcs = []

    for i in xrange(len(Funcs2)):
        def f_aux(x, i=i): 
            r = Funcs2[i][0](x)
            # TODO: are other formats better?
            r = r.tocsc()[Funcs2[i][1]] if isspmatrix(r) else r[Funcs2[i][1]]
            return r
        Funcs.append(f_aux)
        #Funcs.append(lambda x, i=i: Funcs2[i][0](x)[Funcs2[i][1]])
    return Funcs#, inner_ind
    
