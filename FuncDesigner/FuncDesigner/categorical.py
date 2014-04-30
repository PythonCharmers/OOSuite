#PythonSum = sum
import numpy as np
#from ooFun import oofun#, BooleanOOFun
#from FDmisc import FuncDesignerException, raise_except
#from baseClasses import OOFun


def categoricalAttribute(oof, attr):
    from ooFun import oofun
    L = len(oof.domain)
    ind = oof.fields.index(attr)
    dom = [oof.domain[j][ind] for j in range(L)]
    f = lambda x: dom[int(x)] if type(x) != np.ndarray else np.array([dom[i] for i in np.asarray(x, int)])
    r = oofun(f, oof, engine = attr, vectorized = True, domain = dom)
    r._interval_ = lambda domain, dtype: categorical_interval(r, oof, domain, dtype)
    return r

def categorical_interval(r, oof, domain, dtype):
    l, u = domain[oof]
    l_ind, u_ind = np.asarray(l, int), np.asarray(u, int) +1
    s = l_ind.size
    vals = np.zeros((2, s), dtype)
    for j in range(s):
        tmp = np.asarray(r.domain[l_ind[j]:u_ind[j]])
        vals[:, j] = (tmp.min(), tmp.max())
    definiteRange = True
    return vals, definiteRange
