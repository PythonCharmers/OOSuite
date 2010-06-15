try:
    from solverPaths import solverPaths
except:
    pass

from oologfcn import OpenOptException
from numpy import zeros, bmat, hstack, vstack, ndarray, copy, where, prod, asarray, atleast_1d
try:
    import scipy
    scipyInstalled = True
    scipyAbsentMsg = ''
    from scipy.sparse import isspmatrix, csr_matrix, coo_matrix
    from scipy.sparse import bmat as Bmat
    from scipy.sparse import hstack as HstackSP, vstack as VstackSP, find as Find
    def Hstack(Tuple):
        #elems = asarray(Tuple, dtype=object)
        ind = where([prod(elem.shape)!=0 for elem in Tuple])[0].tolist()
        elems = [Tuple[i] for i in ind]
        # [elem if prod(elem.shape)!=0 for elem in Tuple]
        return HstackSP(elems) if any([isspmatrix(elem) for elem in elems]) else hstack(elems)
    def Vstack(Tuple):
        ind = where([prod(elem.shape)!=0 for elem in Tuple])[0].tolist()
        elems = [Tuple[i] for i in ind]
        return VstackSP(elems) if any([isspmatrix(elem) for elem in elems]) else hstack(elems)
    #Hstack = lambda Tuple: HstackSP(Tuple) if any([isspmatrix(elem) for elem in Tuple]) else hstack(Tuple)
    #Vstack = lambda Tuple: VstackSP(Tuple) if any([isspmatrix(elem) for elem in Tuple]) else vstack(Tuple)
    SparseMatrixConstructor = lambda *args, **kwargs: scipy.sparse.lil_matrix(*args, **kwargs)
except:
    scipyInstalled = False
    csr_matrix = None
    coo_matrix = None
    scipyAbsentMsg = 'Probably scipy installation could speed up running the code involved'
    isspmatrix = lambda *args,  **kwargs:  False
    Hstack = hstack
    Vstack = vstack
    Bmat = bmat
    def SparseMatrixConstructor(*args, **kwargs): 
        raise OpenOptException('error in OpenOpt kernel, inform developers')
    def Find(*args, **kwargs): 
        raise OpenOptException('error in OpenOpt kernel, inform developers')
DenseMatrixConstructor = lambda *args, **kwargs: zeros(*args, **kwargs)

##################################################################
def getSolverFromStringName(p, solver_str):
    if solver_str not in solverPaths:
        p.err('incorrect solver is called, maybe the solver "' + solver_str +'" is not installed. Also, maybe you have forgot to use "python setup.py install" after updating OpenOpt from subversion repository')
    if p.debug:
        solverClass =  getattr(my_import('openopt.solvers.'+solverPaths[solver_str]), solver_str)
    else:
        try:
            solverClass = getattr(my_import('openopt.solvers.'+solverPaths[solver_str]), solver_str)
        except:
            p.err('incorrect solver is called, maybe the solver "' + solver_str +'" require its installation, check http://www.openopt.org/%s or try p._solve() for more details' % p.probType)
    return solverClass()

##################################################################
def my_import(name):
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def oosolver(solverName, *args,  **kwargs):
    if args != ():
        raise OpenOptException("Error: oosolver() doesn't consume any *args, use **kwargs only")
    try:
        if ':' in solverName:
            # TODO: make it more properly
            # currently it's used for to get filed isInstalled value
            # from ooSystem
            solverName = solverName.split(':')[1]
            
        solverClass = getattr(my_import('openopt.solvers.'+solverPaths[solverName]), solverName)
        solverClassInstance = solverClass()
        solverClassInstance.fieldsForProbInstance = {}
        for key, value in kwargs.iteritems():
            if hasattr(solverClassInstance, key):
                setattr(solverClassInstance, key, value)
            else:
                solverClassInstance.fieldsForProbInstance[key] = value
        solverClassInstance.isInstalled = True
    except:
        from baseSolver import baseSolver
        solverClassInstance = baseSolver()
        solverClassInstance.__name__ = solverName
        solverClassInstance.isInstalled = False
    return solverClassInstance

def Copy(arg): 
    return arg.copy() if isinstance(arg, ndarray) or isspmatrix(arg) else copy(arg)
