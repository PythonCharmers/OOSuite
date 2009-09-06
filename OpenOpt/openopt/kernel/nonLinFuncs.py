__docformat__ = "restructuredtext en"
from numpy import *
from setDefaultIterFuncs import USER_DEMAND_EXIT
from ooMisc import killThread, setNonLinFuncsNumber
try:
    from DerApproximator import get_d1
    DerApproximatorIsInstalled = True
except:
    DerApproximatorIsInstalled = False

class nonLinFuncs:
    def __init__(self): pass

    def wrapped_func(p, x, IND, userFunctionType, ignorePrev, getDerivative):
        if not getattr(p.userProvided, userFunctionType): return array([])
        if p.istop == USER_DEMAND_EXIT:
            if p.solver.__cannotHandleExceptions__:
                return nan
            else:
                raise killThread
                
        if getDerivative and not p.namedVariablesStyle and not DerApproximatorIsInstalled:
            self.err('To perform gradients check you should have DerApproximator installed, see http://openopt.org/DerApproximator')

        #userFunctionType should be 'f', 'c', 'h'
        funcs = getattr(p.user, userFunctionType)
        #print 'funcs', funcs
        funcs_num = getattr(p, 'n'+userFunctionType)
        if IND is not None:
            ind = p.getCorrectInd(IND)
        else: ind = None

        # this line had been added because some solvers pass tuple instead of
        # x being vector p.n x 1 or matrix X=[x1 x2 x3...xk], size(X)=[p.n, k]
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
            #x_0 = x[0]

        #extractInd = None
#        if userFunctionType == 'c':
#            raise 'asdf'

        # TODO: mb replace D by _D?
        if getDerivative and p.namedVariablesStyle:
            if p.optVars is None or (p.fixedVars is not None and len(p.optVars) < len(p.fixedVars)):
                funcs2 = \
                [(lambda x, i=i: p._pointDerivative2array(funcs[i].D(x, Vars = p.optVars, resultKeysType = 'names'))) for i in xrange(len(funcs))]
            else:
                funcs2 = \
                [(lambda x, i=i: p._pointDerivative2array(funcs[i].D(x, fixedVars = p.fixedVars, resultKeysType = 'names'))) for i in xrange(len(funcs))]
        else:
            funcs2 = funcs
            
        if ind is None: 
            Funcs = funcs2
            #print 'case 1'
#            if userFunctionType == 'c':
#                for i,  f in enumerate(funcs2):
#                    print '!', i, getDerivative, f(p._vector2point(x))
        elif ind is not None and p.functype[userFunctionType] == 'some funcs R^nvars -> R':
            #print 'case 2'
            Funcs = [funcs2[i] for i in ind]
        else:
            #print 'case 3'
            Funcs = getFuncsAndExtractIndexes(p, funcs2, ind, userFunctionType)
        
          #Funcs = getFuncsAndExtractIndexes(p, funcs2, ind, userFunctionType)

#        if ind is not None and p.functype[userFunctionType] == 'block':
#            Funcs, extractInd = getFuncsAndExtractIndexes(p, funcs, ind, userFunctionType)
#            # TODO: add possibility len(extractInd)>1
#        elif ind is not None and len(funcs) > 1:
#            assert p.functype[userFunctionType] == 'some funcs R^nvars -> R'
#            Funcs = [funcs[i] for i in ind]
#        else:
#            Funcs = funcs
#
#        # TODO: get rid of doInplaceCut variable
#        doInplaceCut = ind is not None  and len(funcs) == 1
#        assert not (extractInd is not None and doInplaceCut)
#        
#        if nXvectors > 1: assert extractInd is None and not doInplaceCut

        agregate_counter = 0
        
        if p.namedVariablesStyle:
            Args = ()
        else:
            Args = args
            
        if nXvectors == 1:
            if p.namedVariablesStyle:
                X = p._vector2point(x) 
            else:
                X = x
        
        if nXvectors > 1: # and hence getDerivative isn't involved
            #temporary, to be fixed
            assert userFunctionType == 'f' and p.isObjFunValueASingleNumber
            if p.namedVariablesStyle:
                X = [p._vector2point(x[i]) for i in xrange(nXvectors)]
            elif len(Args) == 0:
                X = [x[i] for i in xrange(nXvectors)]
            else:
                X = [((x[i],) + Args) for i in xrange(nXvectors)]
            r = vstack([map(fun, X) for fun in Funcs])
        elif not getDerivative:
            r = hstack([fun(*(X, )+Args) for fun in Funcs])
            if not ignorePrev and ind is None:
                p.prevVal[userFunctionType]['key'] = copy(x_0)
                p.prevVal[userFunctionType]['val'] = r.copy()                
#            if not doInplaceCut:
#                r = hstack([fun(*(X, )+Args) for fun in Funcs])
#            else:
#                assert len(Funcs) == 1
#                tmp = ravel(Funcs[0](*(X, )+Args))
#                if not ignorePrev: 
#                    p.prevVal[userFunctionType]['key'] = copy(x_0)
#                    p.prevVal[userFunctionType]['val'] = tmp.copy()
#                r = tmp[ind]
        elif getDerivative and p.namedVariablesStyle:
#            if userFunctionType == 'h':
#                raise 0
            #print '1111111', userFunctionType, ind, len(Funcs)
            
            r = vstack([fun(X) for fun in Funcs])

#            import traceback
#            traceback.print_stack()

            #assert r.size < 11
            #print 'asdf','Funcs:', Funcs
            #print 'r:', r
            #r = vstack([p._pointDerivative2array(fun(X)) for fun in Funcs])
            
#            if ind is None:
#                r = vstack([p._pointDerivative2array(fun.D(X)) for fun in Funcs])
#            elif doInplaceCut:  
#                assert len(Funcs) == 1
#                r = p._pointDerivative2array(Funcs[0].D(X))[ind]
#            elif extractInd is not None:  
#                #r = vstack([p._pointDerivative2array(FUNC.D(X))  ])
#                r = p._pointDerivative2array(Funcs[0].D(X))[extractInd]
#            elif len(Funcs) == 1 and ind is not None:
#                assert p.functype[userFunctionType] == 'some funcs R^nvars -> R'
#                r = p._pointDerivative2array(Funcs[0].D(X))
#            else:
#                p.err('error in nonlinfuncs.py, inform developers')
                
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
                #if extractInd is not None:  v = v[extractInd]
                #if doInplaceCut: v = v[ind]
                r[agregate_counter:agregate_counter+v.size,0] = v


                """                                                 geting derivatives                                                 """
                if getDerivative:
#                    if extractInd is not None: 
#                        func = lambda x: ravel(fun(*((x,) + Args)))[extractInd]
#                    elif doInplaceCut:
#                        func = lambda x: ravel(fun(*((x,) + Args)))[ind]
#                    else:
#                        func = lambda x: fun(*((x,) + Args))

                    func = lambda x: fun(*((x,) + Args))
                    
                    d1 = get_d1(func, x, pointVal = None, diffInt = finiteDiffNumbers, stencil=p.JacobianApproximationStencil)
                    r[agregate_counter:agregate_counter+d1.size] = d1
                    
                agregate_counter += atleast_1d(asarray(v)).shape[0]

        if userFunctionType == 'f' and p.isObjFunValueASingleNumber: r = r.sum(0)

        if nXvectors == 1  and  not getDerivative: r = r.flatten()

        if p.invertObjFunc and userFunctionType=='f':
            r = -r

        #if (ind is None or funcs_num==1) and not ignorePrev and nXvectors == 1: p.prevVal[userFunctionType]['key'] = copy(x_0)
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




    def wrapped_1st_derivatives(p, x, ind_, funcType, ignorePrev):
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
            
            if ind is None: Funcs = funcs
            elif ind is not None and p.functype[funcType] == 'some funcs R^nvars -> R':
                Funcs = [funcs[i] for i in ind]
            else:
                Funcs = getFuncsAndExtractIndexes(p, funcs, ind, funcType)
            
#            if ind is not None and p.functype[funcType] == 'block':
#                #Funcs, extractInd = getFuncsAndExtractIndexes(p, funcs, ind, userFunctionType)
#                Funcs = getFuncsAndExtractIndexes(p, funcs, ind, userFunctionType)
#            elif ind is not None and len(funcs) > 1:
#                assert p.functype[funcType] == 'some funcs R^nvars -> R'
#                Funcs = [funcs[i] for i in ind]
#                #extractInd = None
#            else:
#                Funcs = funcs
#                if ind is not None:
#                    extractInd = ind
#                else:
#                    extractInd = None
                
            if ind is None: derivativesNumber = nFuncs
            else: derivativesNumber = len(ind)
                
            derivatives = empty((derivativesNumber, p.n))
            agregate_counter = 0
            for fun in Funcs:#getattr(p.user, derivativesType):
                tmp = atleast_1d(fun(*(x,)+getattr(p.args, funcType)))
                if mod(tmp.size, p.n) != 0:
                    if funcType=='f':
                        p.err('incorrect user-supplied (sub)gradient size of objective function')
                    elif funcType=='c':
                        p.err('incorrect user-supplied (sub)gradient size of non-lin inequality constraints')
                    elif funcType=='h':
                        p.err('incorrect user-supplied (sub)gradient size of non-lin equality constraints')
                #if extractInd is not None: tmp = atleast_2d(tmp)[extractInd]
                if tmp.ndim == 1: m= 1
                else: m = tmp.shape[0]
                if p.functype[funcType] == 'some funcs R^nvars -> R' and m != 1:
                    # TODO: more exact check according to stored p.arr_of_indexes_* arrays
                    p.err('incorrect shape of user-supplied derivative, it should be in accordance with user-provided func size')
                derivatives[agregate_counter : agregate_counter + m] =  tmp#.reshape(tmp.size/p.n,p.n)
                agregate_counter += m
            #TODO: inline ind modification!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if ind is None:
                p.nEvals[derivativesType] += 1
            else:
                #derivatives = derivatives[ind]
                p.nEvals[derivativesType] = p.nEvals[derivativesType] + float(len(ind)) / nFuncs

            if funcType=='f':
                if p.invertObjFunc: derivatives = -derivatives
                if p.isObjFunValueASingleNumber: derivatives = derivatives.flatten()

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

"""
if ind is not None and p.functype[userFunctionType] == 'block':
    Funcs, extractInd = getFuncsAndExtractIndexes(p, funcs, ind, userFunctionType)
    # TODO: add possibility len(extractInd)>1
elif ind is not None and len(funcs) > 1:
    assert p.functype[userFunctionType] == 'some funcs R^nvars -> R'
    Funcs = [funcs[i] for i in ind]
else:
    Funcs = funcs

# TODO: get rid of doInplaceCut variable
doInplaceCut = ind is not None  and len(funcs) == 1
assert not (extractInd is not None and doInplaceCut)

if nXvectors > 1: assert extractInd is None and not doInplaceCut
"""

def getFuncsAndExtractIndexes(p, funcs, ind, userFunctionType):
    if ind is None: return funcs
    #if len(ind) == 1: return [lambda *args, **kwargs: atleast_1d(func(*args, **kwargs))[ind] for func in funcs]
    if len(funcs) == 1 : return [lambda *args, **kwargs: atleast_1d(funcs[0](*args, **kwargs))[ind]]
    
    #p.err("for the solver it doesn't work yet")
    # TODO : assert ind is sorted
    
#    if len(ind) > 1:
#        # TODO! Don't forget to remove ind[0] and use ind instead
#        p.err("multiple index for block problems isn't implemented yet")

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
            #IndDict[i].append(inner_ind)
        else:
            IndDict[left_arr_indexes[i]] = [inner_ind]
            #funcLen = arr_of_indexes[left_arr_indexes[i]] - arr_of_indexes[left_arr_indexes[i]-1]
            #raise 0
            Funcs2.append([funcs[left_arr_indexes[i]], IndDict[left_arr_indexes[i]]])
            #Funcs2.append([funcs[left_arr_indexes[i]], IndDict[left_arr_indexes[i]], funcLen])
    
    Funcs = []
    #if userFunctionType == 'c': 
        #print 'ind',ind,'arr_of_indexes:', arr_of_indexes, 'left_arr_indexes', left_arr_indexes
        #print IndDict
        #raise 0
    #if len(ind)>1: raise 0

        #Funcs.append(lambda x, i=i: (Funcs2[i](x)[IndDict[left_arr_indexes[i]]]))
#    if p.namedVariablesStyle:
#        #Funcs.append(lambda x, i=i: p._pointDerivative2array((Funcs2[i][0](x)), Funcs2[i][2])[Funcs2[i][1]])
#    else:

    #ff = lambda x, i=i: (Funcs2[i][0](x))[Funcs2[i][1]]
#    def ff(x, i):
#        print '1: <<< Funcs2[i][0](x) =',  Funcs2[i][0](x),  ' >>>'
#        print '2: <<< Funcs2[i][1] = ',  Funcs2[i][1],  ' >>>'
#        print 'result= <<<', Funcs2[i][0](x)[Funcs2[i][1]], '>>>'
#        return Funcs2[i][0](x)[Funcs2[i][1]]

    for i in xrange(len(Funcs2)):
        #Funcs.append(lambda x, i=i: (Funcs2[i][0](x))[Funcs2[i][1]])
        #Funcs.append(lambda x, i=i: ff(x, i))
        Funcs.append(lambda x, i=i: Funcs2[i][0](x)[Funcs2[i][1]])
        
        #Funcs.append(lambda *args, **kwargs: (Funcs2[i](*args, **kwargs)[IndDict[left_arr_indexes[i]]] if len(IndDict[left_arr_indexes[i]])==1 else Funcs2[i](*args, **kwargs)[IndDict[left_arr_indexes[i]][]]) 
    
#    for left_arr_ind in left_arr_indexes:
#        if left_arr_ind != 0:
#            num_of_funcs_before_arr_left_border = arr_of_indexes[left_arr_ind-1]
#            inner_ind = ind[0] - num_of_funcs_before_arr_left_border - 1
#        else:
#            inner_ind = ind[0]
#        Funcs.append(lambda *args, **kwargs: funcs[left_arr_ind](*args, **kwargs)) 
    return Funcs#, inner_ind
    
#def getFuncsAndExtractIndexes(p, funcs, ind, userFunctionType):
#    if len(ind) > 1:
#        # TODO! Don't forget to remove ind[0] and use ind instead
#        p.err("multiple index for block problems isn't implemented yet")
#
#    #getting number of block and shift
#    arr_of_indexes = getattr(p, 'arr_of_indexes_' + userFunctionType)
#    left_arr_ind = searchsorted(arr_of_indexes, ind[0]) # CHECKME! is it index of block?
#
#    if left_arr_ind != 0:
#        num_of_funcs_before_arr_left_border = arr_of_indexes[left_arr_ind-1]
#        inner_ind = ind[0] - num_of_funcs_before_arr_left_border - 1
#    else:
#        inner_ind = ind[0]
#    Funcs = (funcs[left_arr_ind], )
#    return Funcs, inner_ind
    
