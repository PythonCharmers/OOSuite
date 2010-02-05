from numpy.linalg import norm
from numpy import dot, asfarray, atleast_1d,  zeros, ones, int, float64, where, inf, ndarray
from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.nonOptMisc import isspmatrix, scipyInstalled, scipyAbsentMsg

try:
    from lsqr import lsqr as LSQR
except:
    from openopt.kernel.oologfcn import OpenOptException
    raise OpenOptException('You should have lsqr installed, see OO LLSP doc webpage for details')

try:
    from scipy.sparse import csc_matrix, csr_matrix
except:
    pass

class lsqr(baseSolver):
    __name__ = 'lsqr'
    __license__ = "GPL?"
    __authors__ = 'Michael P. Friedlander (University of British Columbia), Dominique Orban (Ecole Polytechnique de Montreal)'
    __alg__ = 'an iterative (conjugate-gradient-like) method'
    __info__ = """    
    Parameters: atol (default 1e-9), btol (1e-9), conlim ('autoselect', default 1e8 for LLSP and 1e12 for SLE)
    
    For further information, see 

    1. C. C. Paige and M. A. Saunders (1982a).
       LSQR: An algorithm for sparse linear equations and sparse least squares,
       ACM TOMS 8(1), 43-71.
    2. C. C. Paige and M. A. Saunders (1982b).
       Algorithm 583.  LSQR: Sparse linear equations and least squares problems,
       ACM TOMS 8(2), 195-209.
    3. M. A. Saunders (1995).  Solution of sparse rectangular systems using
       LSQR and CRAIG, BIT 35, 588-604."""
       
    __optionalDataThatCanBeHandled__ = ['damp', 'X']
    _canHandleScipySparse = True
    atol = 1e-9
    btol = 1e-9
    conlim = 'autoselect'
    
    def __init__(self): pass

    def __solver__(self, p):
        condX = hasattr(p, 'X') and any(p.X)
        if condX: 
            p.err("sorry, the solver can't handle non-zero X data yet, but you can easily handle it by yourself")
        C, d = p.C, p.d
        m, n = C.shape[0], p.n

        if isinstance(C, ndarray) and 0.25* C.size > C.nonzero()[0].size:
            if not scipyInstalled: 
                p.pWarn(scipyAbsentMsg)
            else:
                C = csc_matrix(C)
                
        CT = C.T
        
        def aprod(mode, m, n, x):
            if mode == 1:
                r = dot(C, x).flatten() if not isspmatrix(C) else C._mul_sparse_matrix(csr_matrix(x.reshape(x.size, 1))).A.flatten()
                
                # It doesn't implemented properly yet
#                f = p.norm(r-d)
#                assert damp == 0
#                if damp != 0: 
#                assert not condX
#                p.iterfcn(x,f)
                return r
            elif mode == 2:
                return dot(CT, x).flatten() if not isspmatrix(C) else CT._mul_sparse_matrix(csr_matrix(x.reshape(x.size, 1))).A.flatten()
            
        if self.conlim == 'autoselect':
            conlim = 1e12 if m == n else 1e8
            
        damp = self.damp if hasattr(self, 'damp') and self.damp is not None else 0
        show = False
        
        [ x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var ] = \
        LSQR(m, n, aprod, d, damp, 1e-9, 1e-9, conlim, p.maxIter, show, wantvar = False)
        # ( m, n, aprod, b, damp, atol, btol, conlim, itnlim, show, wantvar = False )

        #p.istop, p.msg, p.iter = istop, msg.rstrip(), iter
        p.istop = 1000
        p.iter = 1 # itn
        p.xf = x
        #p.ff = p.fk = w[0]

