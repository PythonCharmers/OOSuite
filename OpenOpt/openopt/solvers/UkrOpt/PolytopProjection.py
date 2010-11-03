from openopt import QP
from numpy import *
from numpy.linalg import norm

def PolytopProjection(data, T = 1.0):
    n, m = data.shape
    #data = float128(data)
    H = dot(data, data.T)
    #print H.shape
    #print 'PolytopProjection: n=%d, m=%d, H.shape[0]= %d, H.shape[1]= %d ' %(n, m, H.shape[0], H.shape[1])
    #T = abs(dot(H, ones(n)))
    f = -asfarray(T) *ones(n)
    p = QP(H, f, lb = zeros(n), iprint = -1, maxIter = 150)

    solver = 'cvxopt_qp'
    #solver = 'nlp:scipy_lbfgsb'
    #solver = 'nlp:scipy_tnc'
    #solver = 'nlp:ralg'
#    solver = 'nlp:algencan'
#    solver = 'nlp:ipopt'
    r = p.solve(solver, ftol = 1e-16, xtol = 1e-16, maxIter = 10000)
    sol = r.xf

    s = dot(data.T, r.xf)
    return s.flatten(), r.xf
