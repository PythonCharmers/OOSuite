from openopt import QP
from numpy import *
from numpy.linalg import norm

def PolytopProjection(data, T = 1.0):
    n, m = data.shape
    H = dot(data, data.T)
    #print H.shape
    #print 'PolytopProjection: n=%d, m=%d, H.shape[0]= %d, H.shape[1]= %d ' %(n, m, H.shape[0], H.shape[1])
    #T = abs(dot(H, ones(n)))
    f = -T *ones(n)
    p = QP(H, f, lb = zeros(n), iprint = 1, maxIter = 20)

    r = p.solve('cvxopt_qp', iprint = -1)#, ftol = 1e-13, xtol = 1e-13)
    sol = r.xf

    s = (data * r.xf.reshape(-1, 1)).sum(0)
    
    #print norm(s) / sum(sol)
    return s
