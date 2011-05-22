from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, searchsorted, logical_or, any, nan, isinf, \
arange, vstack, inf, where, logical_not, take
from FuncDesigner import ooPoint

try:
    from bottleneck import nanargmin, nanmin, nanargmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax


    
def func10(y, e, n, ooVars):
    LB = [[] for i in range(n)]
    UB = [[] for i in range(n)]
    m = y.shape[0]
    Centers = 0.5 * (y + e)
    
    # TODO: remove the cycle
    #T1, T2 = tile(y, (2*n,1)), tile(e, (2*n,1))
    
    for i in range(n):
        t1, t2 = tile(y[:, i], 2*n), tile(e[:, i], 2*n)
        #t1, t2 = T1[:, i], T2[:, i]
        #T1[(n+i)*m:(n+i+1)*m, i] = T2[i*m:(i+1)*m, i] = Centers[:, i]
        t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = Centers[:, i]
        LB[i], UB[i] = t1, t2

####        LB[i], UB[i] = T1[:, i], T2[:, i]

#    sh1, sh2, inds = [], [], []
#    for i in range(n):
#        sh1+= arange((n+i)*m, (n+i+1)*m).tolist()
#        inds +=  [i]*m
#        sh2 += arange(i*m, (i+1)*m).tolist()

#    sh1, sh2, inds = asdf(m, n)
#    asdf2(T1, T2, Centers, sh1, sh2, inds)
    
    #domain = dict([(v, (T1[:, i], T2[:, i])) for i, v in enumerate(ooVars)])
    domain = dict([(v, (LB[i], UB[i])) for i, v in enumerate(ooVars)])
    
    domain = ooPoint(domain, skipArrayCast = True)
    domain.isMultiPoint = True
    return domain

def func8(domain, fd_obj, dataType):

    TMP = fd_obj.interval(domain, dataType)
    
    #TODO: remove 0.5*(val[0]+val[1]) from cycle
    centers = dict([(key, 0.5*(val[0]+val[1])) for key, val in domain.items()])
    centers = ooPoint(centers, skipArrayCast = True)
    centers.isMultiPoint = True
    F = fd_obj(centers)
    bestCenterInd = nanargmin(F)
    if isnan(bestCenterInd):
        bestCenterInd = 0
        bestCenterObjective = inf
    
    # TODO: check it , maybe it can be improved
    #bestCenter = centers[bestCenterInd]
    bestCenter = array([0.5*(val[0][bestCenterInd]+val[1][bestCenterInd]) for val in domain.values()], dtype=dataType)
    bestCenterObjective = atleast_1d(F)[bestCenterInd]
    #assert TMP.lb.dtype == dataType
    return asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType), bestCenter, bestCenterObjective

def func7(y, e, o, a):
    IND = logical_and(all(isnan(o), 1), all(isnan(a), 1))
    if any(IND):
        j = where(logical_not(IND))[0]
        lj = j.size
        y = take(y, j, axis=0, out=y[:lj])
        e = take(e, j, axis=0, out=e[:lj])
        o = take(o, j, axis=0, out=o[:lj])
        a = take(a, j, axis=0, out=a[:lj])
    return y, e, o, a 

def func9(an, fo, g):
    maxo = [node.key for node in an]
    ind = searchsorted(maxo, fo, side='right')
    if ind == len(maxo):
        return an, g
    else:
        g = nanmin((g, nanmin(maxo[ind])))
        return an[:ind], g


def func5(an, nCut, g):
    m = len(an)
    if m > nCut:
        maxo = [node.key for node in an]
        g = nanmin((maxo[nCut], g))
        an = an[:nCut]
    return an, g

def func4(y, e, o, a, n, fo):
    centers = 0.5*(y + e)
    o_modL, o_modU = o[:, 0:n], o[:, n:2*n]
    ind = logical_or(o_modL > fo, isnan(o_modL)) # TODO: assert isnan(o_modL) is same to isnan(a_modL)
    if any(ind):
        y[ind] = centers[ind]
    ind = logical_or(o_modU > fo, isnan(o_modU))# TODO: assert isnan(o_modU) is same to isnan(a_modU)
    if any(ind):
        e[ind] = centers[ind]
    return y, e

def func3(an, maxActiveNodes):
    m = len(an)
    if m > maxActiveNodes:
        an1, _in = an[:maxActiveNodes], an[maxActiveNodes:]
    else:
        an1, _in = an, []
    return an1, _in

def func1(y, e, o, a, n, varTols):
    Case = 1 # TODO: check other
    if Case == -3:
        t = argmin(a, 1) % n
    elif Case == -2:
        t = asarray([itn % n]*m)
    elif Case == -1:
        tmp = a - o
        tmp1, tmp2 = tmp[:, 0:n], tmp[:, n:2*n]
        tmp = tmp1
        ind = where(tmp2>tmp1)
        tmp[ind] = tmp2[ind]
        #tmp = tmp[:, 0:n] + tmp[:, n:2*n]
        t = argmin(tmp, 1) 
    elif Case == 0:
        t = argmin(a - o, 1) % n
    elif Case == 1:
#                a1, a2 = a[:, 0:n], a[:, n:]
#                ind = a1 < a2
#                a1[ind] = a2[ind]
#                t = argmin(a1, 1)

        a = a.copy() # to remain it unchanged in higher stack level funcs
        a[e-y<varTols] = nan
        
        t = nanargmin(a, 1) % n
        if any(isinf(a)):
            # new
#                    a1, a2 = a[:, 0:n], a[:, n:]
#                    
#                    #ind1, ind2 = isinf(a1), isinf(a2)
#                    #ind_any_infinite = logical_or(ind1, ind2)
#                    ind1, ind2 = isinf(a1), isinf(a2)
#                    ind_any_infinite = logical_or(ind1, ind2)
#                    
#                    a_ = where(a1 < a2, a1, a2)
#                    a_[ind_any_infinite] = inf
#                    t = nanargmin(a_, 1) 
#                    ind = isinf(a_[w, t])

##                    #old
            ###t = argmin(a, 1) % n
            #ind = logical_or(all(isinf(a), 1), all(isinf(o), 1))
            ind = all(isinf(a), 1)
            if any(ind):
                boxShapes = e[ind] - y[ind]
                t[ind] = nanargmax(boxShapes, 1)
                
    elif Case == 2:
        o1, o2 = o[:, 0:n], o[:, n:]
        ind = o1 > o2
        o1[ind] = o2[ind]                
        t = argmax(o1, 1)
    elif Case == 3:
        # WORST
        t = argmin(o, 1) % n
    elif Case == 4:
        # WORST
        t = argmax(a, 1) % n
    elif Case == 5:
        tmp = where(o[:, 0:n]<o[:, n:], o[:, 0:n], o[:, n:])
        t = argmax(tmp, 1)
    return t


def func2(y, e, t):
    new_y, new_e = y.copy(), e.copy()
    m = y.shape[0]
    w = arange(m)
    th = 0.5 * (new_y[w, t] + new_e[w, t])
    new_y[w, t] = th
    new_e[w, t] = th
    
    new_y = vstack((y, new_y))
    new_e = vstack((new_e, e))
    
    return new_y, new_e


func11 = lambda y, e, o, a, maxo: [si(maxo[i], y[i], e[i], o[i], a[i]) for i in range(len(maxo))]

class si:
    def __init__(self, key, *data):
        self.key = key
        self.data = data
    
    __lt__ = lambda self, other: self.key < other.key
    __le__ = lambda self, other: self.key <= other.key
    __gt__ = lambda self, other: self.key > other.key
    __ge__ = lambda self, other: self.key >= other.key
    __eq__ = lambda self, other: self.key == other.key
