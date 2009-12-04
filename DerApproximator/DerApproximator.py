"""finite-difference derivatives approximation"""

from numpy import atleast_1d, atleast_2d, isfinite, ndarray, nan, empty, where, ndarray, log10, hstack, floor, ceil, \
argmax, asscalar, abs, isscalar, asfarray, asarray, isnan

class DerApproximatorException:
    def __init__(self,  msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def get_d1(fun, vars, diffInt=1.5e-8, pointVal = None, args=(), stencil = 2, varForDifferentiation = None):
    """
    Usage: get_d1(fun, x, diffInt=1.5e-8, pointVal = None, args=(), stencil = 2, varForDifferentiation = None)
    fun: R^n -> R^m, x0 from R^n: function and point where derivatives should be obtained 
    diffInt - step for stencil
    pointVal - fun(x) if known (it is used from OpenOpt and FuncDesigner)
    args - additional args for fun, if not () fun(x, *args) will be involved 
    stencil = 1: (f(x+diffInt) - f(x)) / diffInt
    stencil = 2: (f(x+diffInt) - f(x-diffInt)) / (2*diffInt)
    varForDifferentiation - the parameter is used from FuncDesigner
    """
    assert type(vars) in [tuple,  list,  ndarray, float, dict]
    
    #assert asarray(diffInt).size == 1,  'vector diffInt are not implemented for oofuns yet'      
    diffInt = atleast_1d(diffInt)
    if atleast_1d(diffInt).size > 1: assert type(vars) == ndarray, 'not implemented yet'
    
    if type(vars) not in [list, tuple] or isscalar(vars[0]):
        Vars = [vars, ]
    else: 
        Vars = list(vars)
        # TODO: IMPLEMENT CHECK FOR SAME VARIABLES IN INPUT
        #if len(set(tuple(Vars))) != len(Vars):
            #raise DerApproximatorException('currently DerApproximator can handle only different input variables')
    
    if type(args) != tuple:
        args = (args, )
    Args = list(tuple([asfarray(v) for v in Vars]) + args)#list(tuple(Vars) + args)

    if pointVal is None:
        v_0 = atleast_1d(fun(*Args))
    else:
        v_0 = pointVal
    if v_0.ndim >= 2: 
        raise DerApproximatorException('Currently DerApproximatorx cannot handle functions with (ndim of output) > 1 , only vectors are allowed')
    M = v_0.size
    r = []

    for i in xrange(len(Vars)):
        if varForDifferentiation is not None and i != varForDifferentiation: continue
        if not isscalar(Args[i]):
            Args[i] = asfarray(Args[i])
            S = Args[i]
        else:
            S = asfarray([Args[i]])
            
        agregate_counter = 0
        assert asarray(Args[i]).ndim <= 1, 'derivatives for more than single dimension variables are not implemented yet'
        
        if diffInt.size == 1: diff_int = asarray([diffInt[0]]*S.size)# i.e. python list of length inp.size
        else: diff_int = diffInt
   
        cmp = atleast_1d(1e-10 * abs(S))
        ind = where(diff_int<cmp)[0]
        diff_int[ind] = cmp[ind]
        
        d1 = empty((M, S.size))
        #d1.fill(nan)
        

        for j in xrange(S.size):
            di = float(asscalar(diff_int[j]))
            tmp = S[j]
            di = diff_int[j]
            S[j] += di
            TMP = fun(*Args)
            if not isscalar(TMP): TMP = hstack(TMP)
            v_right = atleast_1d(TMP)# TODO: fix it for matrices with ndims>1
            S[j] = tmp 
            # not Args[i][j] -= di, because it can change primeval Args[i][j] value 
            # and check for key == same value will be failed
            
            has_nonfinite = not all(isfinite(v_right))
            if stencil == 2 or has_nonfinite:
                S[j] -= di
                v_left = atleast_1d(fun(*Args))
                S[j] = tmp
                if has_nonfinite:
                    d1[:, agregate_counter] = (v_0-v_left) / di
                else:
                    d1[:, agregate_counter] = (v_right-v_left) / (2.0 * di)
            else:
                d1[:, agregate_counter] = (v_right-v_0) / di
            agregate_counter += 1 # TODO: probably other values for n-dim arrays
            
        # TODO: fix it for arrays with ndim > 2
        if min(d1.shape)==1: d1 = d1.flatten()
        
        r.append(asfarray(d1))
    if varForDifferentiation is not None or isscalar(vars) or (type(vars) in [list, tuple, ndarray] and isscalar(vars[0])): r = d1
    else: r = tuple(r)
    return r

def check_d1(fun, fun_d, vars, func_name='func', diffInt=1.5e-8, pointVal = None, args=(), stencil = 2, maxViolation=0.01, varForCheck = None):
    """
    Usage: check_d1(fun, fun_d, x, func_name='func', diffInt=1.5e-8, pointVal = None, args=(), stencil = 2, maxViolation=0.01, varForCheck = None)
    fun: R^n -> R^m, x0 from R^n: function and point where derivatives should be obtained 
    fun_d - user-provided routine for derivatives evaluation to be checked 
    diffInt - step for stencil
    pointVal - fun(x) if known (it is used from OpenOpt and FuncDesigner)
    args - additional args for fun, if not () fun(x, *args) will be involved 
    stencil = 1: (f(x+diffInt) - f(x)) / diffInt
    stencil = 2: (f(x+diffInt) - f(x-diffInt)) / (2*diffInt)
    maxViolation - threshold for reporting of incorrect derivatives
    varForCheck - the parameter is used from FuncDesigner
    
    Note that one of output values RD (relative difference) is defined as
    int(ceil(log10(abs(Diff) / maxViolation + 1e-150)))
    where
    Diff = 1 - (info_user+1e-8)/(info_numerical + 1e-8) 
    """
    info_numerical = get_d1(fun, vars, diffInt=diffInt, pointVal = pointVal, args=args, stencil = stencil, varForDifferentiation = varForCheck)
    
    if type(vars) not in [list, tuple]:
        Vars = [vars, ]
    else: Vars = list(vars)
    
    if type(args) != tuple:
        args = (args, )
    Args = list(tuple(Vars) + args)
    
    if isinstance(fun_d, ndarray):
        info_user = fun_d
    else:
        info_user = asfarray(fun_d(*Args))
    
    if min(info_numerical.shape) == 1: info_numerical = info_numerical.flatten()
    if min(info_user.shape) == 1: info_user = info_user.flatten()
    
    if atleast_2d(info_numerical).shape != atleast_2d(info_user).shape:
        raise DerApproximatorException('user-supplied gradient for ' + func_name + ' has other size than the one, obtained numerically: '+ \
        str(info_numerical.shape) + ' expected, ' + str(info_user.shape) + ' obtained')
    
    Diff = 1 - (info_user+1e-8)/(info_numerical + 1e-8) # 1e-8 to suppress zeros
    log10_RD = log10(abs(Diff)/maxViolation+1e-150)

    #TODO: omit flattering
    d = hstack((info_user.reshape(-1,1), info_numerical.reshape(-1,1), Diff.reshape(-1,1)))
    
    if info_numerical.ndim > 1: useDoubleColumn = True
    else: useDoubleColumn = False

    cond_same = all(abs(Diff).flatten() <= maxViolation)
    if not cond_same:
        ss = '    '
            
        if useDoubleColumn:
            ss = ' i,j: d' + func_name + '[i]/dx[j]'

        s = func_name + ' num  ' + ss + '   user-supplied     numerical               RD'
        print(s)

    ns = ceil(log10(d.shape[0]))
    counter = 0
    fl_info_user = info_user.flatten()
    fl_info_numerical = info_numerical.flatten()
    if len(Diff.shape) == 1:
        Diff = Diff.reshape(-1,1)
        log10_RD = log10_RD.reshape(-1,1)
    for i in xrange(Diff.shape[0]):
        for j in xrange(Diff.shape[1]):
            if abs(Diff[i,j]) < maxViolation: continue
            counter += 1
            k = Diff.shape[1]*i+j
            nSpaces = ns - floor(log10(k+1))+2
            if useDoubleColumn:  ss = str(i) + ' / ' + str(j)
            else: ss = ''

            if len(Diff.shape) == 1 or Diff.shape[1] == 1: n2 = 0
            else: n2 = 15
            RDnumOrNan = 'NaN' if isnan(log10_RD[i,j]) else ('%d' % int(ceil(log10_RD[i,j])))
            s = '    ' + ('%d' % k).ljust(5) + ss.rjust(n2) + ('%+0.3e' % fl_info_user[k]).rjust(19) + ('%+0.3e' % fl_info_numerical[k]).rjust(15) + RDnumOrNan.rjust(15)
            print(s)

    diff_d = abs(d[:,0]-d[:,1])
    ind_max = argmax(diff_d)
    val_max = diff_d[ind_max]
    if not cond_same:
        print('max(abs(d_user - d_numerical)) = ' + str(val_max))
        print('(is registered in func number ' + str(ind_max) + ')')
    else:
        print('derivatives are equal')
    print(75 * '*')
    
#def check_d1(fun, fun_d, vars, func_name='func', diffInt=1e-7, pointVal = None, args=(), twoSized = True, maxViolation=0.01):
#    info_numerical = get_d1(fun, vars, diffInt=diffInt, pointVal = pointVal, args=args, twoSized = twoSized)
#    
#    if type(vars) not in [list, tuple]:
#        Vars = [vars, ]
#    else: Vars = list(vars)
#    
#    if type(args) != tuple:
#        args = (args, )
#    Args = list(tuple(Vars) + args)
#    
#    #info_user = fun_d(*Args)
#    info_user = fun_d()
#    
#    # TODO: check for fixed oofuncs
#    
#    if info_numerical.shape != info_user.shape:
#        raise DerApproximatorException('user-supplied gradient for ' + func_name + ' has other size than the one, obtained numerically: '+ \
#        str(info_numerical.shape) + ' expected, ' + str(info_user.shape) + ' obtained')
#    
#    Diff = 1 - (info_user+1e-8)/(info_numerical + 1e-8) # 1e-8 to suppress zeros
#    log10_RD = log10(abs(Diff)/maxViolation+1e-150)
#
#    #TODO: omit flattering
#    d = hstack((info_user.reshape(-1,1), info_numerical.reshape(-1,1), Diff.reshape(-1,1)))
#    
#    print('Checking user-supplied gradient for ' + func_name + ' of shape ' + str(info_user.shape))
#    print('according to:')
#    print('    diffInt = ' + str(diffInt)) # TODO: ADD other parameters: allowed epsilon, maxDiffLines etc
#    print('    |1 - info_user/info_numerical| < maxViolation = '+ str(maxViolation))
#
#    if info_numerical.ndim > 1: useDoubleColumn = True
#    else: useDoubleColumn = False
#
#    cond_same = all(abs(Diff).flatten() <= maxViolation)
#    if not cond_same:
#        ss = '    '
#            
#        if useDoubleColumn:
#            ss = ' i,j: d' + func_name + '[i]/dx[j]'
#
#        s = func_name + ' num  ' + ss + '   user-supplied     numerical               RD'
#        print(s)
#
#    ns = ceil(log10(d.shape[0]))
#    counter = 0
#    fl_info_user = info_user.flatten()
#    fl_info_numerical = info_numerical.flatten()
#    if len(Diff.shape) == 1:
#        Diff = Diff.reshape(-1,1)
#        log10_RD = log10_RD.reshape(-1,1)
#    for i in xrange(Diff.shape[0]):
#        for j in xrange(Diff.shape[1]):
#            if abs(Diff[i,j]) < maxViolation: continue
#            counter += 1
#            k = Diff.shape[1]*i+j
#            nSpaces = ns - floor(log10(k+1))+2
#            if useDoubleColumn:  ss = str(i) + ' / ' + str(j)
#            else: ss = ''
#
#            if len(Diff.shape) == 1 or Diff.shape[1] == 1: n2 = 0
#            else: n2 = 15
#            s = '    ' + ('%d' % k).ljust(5) + ss.rjust(n2) + ('%+0.3e' % fl_info_user[k]).rjust(19) + ('%+0.3e' % fl_info_numerical[k]).rjust(15) + ('%d' % int(ceil(log10_RD[i,j]))).rjust(15)
#            print(s)
#
#    diff_d = abs(d[:,0]-d[:,1])
#    ind_max = argmax(diff_d)
#    val_max = diff_d[ind_max]
#    if not cond_same:
#        print('max(abs(d_user - d_numerical)) = ' + str(val_max))
#        print('(is registered in func number ' + str(ind_max) + ')')
#    else:
#        print('derivatives are equal')
#
#    print(64 * '*')
#    
#
#    
#    
#    
#    
#    
#    
