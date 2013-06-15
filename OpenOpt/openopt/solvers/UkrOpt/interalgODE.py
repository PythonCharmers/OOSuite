from numpy import hstack,  asarray, abs, atleast_1d, where, \
logical_not, argsort, vstack, sum, array, nan, all
import numpy as np
from FuncDesigner import oopoint
#from FuncDesigner.boundsurf import boundsurf


def interalg_ODE_routine(p, solver):
    isIP = p.probType == 'IP'
    isODE = p.probType == 'ODE'
    if isODE:
        f, y0, t, r30, ftol = p.equations, p.x0, p.timeVariable, p.times, p.ftol
        assert len(f) == 1, 'multiple ODE equations are unimplemented for FuncDesigner yet'
        f = list(f.values())[0]
    elif isIP:
        assert p.n == 1 and p.__isNoMoreThanBoxBounded__()
        f, y0, ftol = p.user.f[0], 0.0, p.ftol
        if p.fTol is not None: ftol = p.fTol
        t = list(f._getDep())[0]
        r30 = p.domain[t]
        p.iterfcn(p.point([nan]*p.n))
    else:
        p.err('incorrect prob type for interalg ODE routine') 
    
    eq_var = list(p._x0.keys())[0]

    dataType = solver.dataType
    if type(ftol) == int: 
        ftol = float(ftol) # e.g. someone set ftol = 1
    # Currently ftol is scalar, in future it can be array of same length as timeArray
    if len(r30) < 2:
        p.err('length ot time array must be at least 2')    
#    if any(r30[1:] < r30[:-1]):
#        p.err('currently interalg can handle only time arrays sorted is ascending order')  
#    if any(r30 < 0):
#        p.err('currently interalg can handle only time arrays with positive values')  
#    if p.times[0] != 0:
#        p.err('currently solver interalg requires times start from zero')  
    
    r37 = abs(r30[-1] - r30[0])
    if len(r30) == 2:
        r30 = np.linspace(r30[0], r30[-1], 150)
    r28 = asarray(atleast_1d(r30[:-1]), dataType)
    r29 = asarray(atleast_1d(r30[1:]), dataType)

    storedr28 = []
    r27 = []
    r31 = []
    r32 = []
    r33 = ftol
    F = 0.0
    p._Residual = 0
    
    # Main cycle
    for itn in range(p.maxIter+1):
        if r30[-1] > r30[0]:
            mp = oopoint({t: [r28, r29]}, skipArrayCast = True)
        else:
            mp = oopoint({t: [r29, r28]}, skipArrayCast = True)
        mp.isMultiPoint = True
        
        mp.dictOfFixedFuncs = p.dictOfFixedFuncs
        mp.surf_preference = True
        tmp = f.interval(mp, allowBoundSurf = True)
        if not all(tmp.definiteRange):
            p.err('''
            solving ODE and IP by interalg is implemented for definite (real) range only, 
            no NaN values in integrand are allowed''')
        # TODO: perform check on NaNs
        
        if hasattr(tmp, 'resolve'):# boundsurf:
            #adjustr4WithDiscreteVariables(wr4, p)
            cs = oopoint([(v, asarray(0.5*(val[0] + val[1]), dataType)) for v, val in mp.items()])
            cs.dictOfFixedFuncs = p.dictOfFixedFuncs
            r21, r22 = tmp.values(cs)
            if isIP:
                o, a = atleast_1d(r21), atleast_1d(r22)
                r20 = a-o
            elif isODE:
                l, u = tmp.l, tmp.u
                assert len(l.d) == len(u.d) == 1 # only time variable
                l_koeffs, u_koeffs = list(l.d.values())[0], list(u.d.values())[0]
                l_c, u_c = l.c, u.c
#                dT = r29 - r28 if r30[-1] > r30[0] else r28 - r29

                
                ends = oopoint([(v, asarray(val[1], dataType)) for v, val in mp.items()])
                ends.dictOfFixedFuncs = p.dictOfFixedFuncs
                ends_L, ends_U = tmp.values(ends)

#                o, a = atleast_1d(r21), atleast_1d(r22)

                o, a = tmp.resolve()[0]
#                r20 = 0.5 * u_koeffs * dT  + u_c  - (0.5 * l_koeffs * dT  + l_c)
                r20 = 0.5 * (ends_U - ends_L)
#                r20 = 0.5 * u_koeffs * dT ** 2 + u_c * dT - (0.5 * l_koeffs * dT ** 2 + l_c * dT)
#                r20 =  0.5*u_koeffs * dT  + u_c  - ( 0.5*l_koeffs * dT  + l_c)

#                o = 0.5*l_koeffs * dT + l_c
#                a = 0.5*u_koeffs * dT + u_c
                #assert 0, 'unimplemented'
            else:
                assert 0
        else:
            o, a = atleast_1d(tmp.lb), atleast_1d(tmp.ub)
            r20 = a - o

        
#        if isODE:
#            r36 = atleast_1d(r20 <= 0.95 * r33)
##            r36 = np.logical_and(r36, r20 < ftol)
##            r36 = np.logical_and(r36, a-o < ftol)
#        else:
#            r36 = atleast_1d(r20 <= 0.95 * r33 / r37)

        r36 = atleast_1d(r20 <= 0.95 * r33 / r37)
        if isODE:
#            r36 = np.logical_and(r36, r20 < ftol)
            r36 = np.logical_and(r36, a-o < ftol)

        ind = where(r36)[0]
        if isODE:
            storedr28.append(r28[ind])
            r27.append(r29[ind])
            r31.append(a[ind])
            r32.append(o[ind])
#            r31.append(a[ind])
#            r32.append(o[ind])
        else:
            assert isIP
            F += 0.5 * sum((r29[ind]-r28[ind])*(a[ind]+o[ind]))
        
        if ind.size != 0: 
            tmp = abs(r29[ind] - r28[ind])
            Tmp = sum(r20[ind] * tmp) #if isIP else sum(r20[ind])
            r33 -= Tmp
            if isIP: p._residual += Tmp
            r37 -= sum(tmp)
        ind = where(logical_not(r36))[0]
        if ind.size == 0:
            p.istop = 1000
            p.msg = 'problem has been solved according to required user-defined accuracy %0.1g' % ftol
            break
            
        # OLD
#        for i in ind:
#            t0, t1 = r28[i], r29[i]
#            t_m = 0.5 * (t0+t1)
#            newr28.append(t0)
#            newr28.append(t_m)
#            newr29.append(t_m)
#            newr29.append(t1)
        # NEW
        r38, r39 = r28[ind], r29[ind]
        r40 = 0.5 * (r38 + r39)
        r28 = vstack((r38, r40)).flatten()
        r29 = vstack((r40, r39)).flatten()
        
        # !!! unestablished !!!
        if isODE:
            p.iterfcn(fk = r33/ftol)
        elif isIP:
            p.iterfcn(xk=array(nan), fk=F, rk = ftol - r33)
        else:
            p.err('bug in interalgODE.py')
            
        if p.istop != 0 : 
            break
        
        #print(itn, r28.size)

    if isODE:
        
        t0, t1, lb, ub = hstack(storedr28), hstack(r27), hstack(r32), hstack(r31)
        ind = argsort(t0)
        if r30[0] > r30[-1]:
            ind = ind[::-1] # reverse
        t0, t1, lb, ub = t0[ind], t1[ind], lb[ind], ub[ind]
        lb, ub = hstack((y0, y0+(lb*(t1-t0)).cumsum())), hstack((y0, y0+(ub*(t1-t0)).cumsum()))
        #y_var = p._x0.keys()[0]
        #p.xf = p.xk = 0.5*(lb+ub)
        p.extras = {'startTimes': t0, 'endTimes': t1, eq_var:{'infinums': lb, 'supremums': ub}}
        return t0, t1, lb, ub
    elif isIP:
        P = p.point([nan]*p.n)
        P._f = F
        P._mr = ftol - r33
        P._mrName = 'None'
        P._mrInd = 0
#        p.xk = array([nan]*p.n)
#        p.rk = r33
#        p.fk = F
        #p._Residual = 
        p.iterfcn(asarray([nan]*p.n), fk=F, rk = ftol - r33)
    else:
        p.err('incorrect prob type in interalg ODE routine')
