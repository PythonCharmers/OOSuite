try:
    from solverPaths import solverPaths
except:
    pass
from baseSolver import baseSolver
from oologfcn import OpenOptException
from numpy import zeros, bmat, hstack, vstack
try:
    import scipy
    scipyInstalled = True
    scipyAbsentMsg = ''
    from scipy.sparse import isspmatrix, csr_matrix
    from scipy.sparse import bmat as Bmat
    from scipy.sparse import hstack as Hstack, vstack as Vstack, find as Find
    SparseMatrixConstructor = lambda *args, **kwargs: scipy.sparse.lil_matrix(*args, **kwargs)
except:
    scipyInstalled = False
    csr_matrix = None
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
        solverClassInstance = baseSolver()
        solverClassInstance.__name__ = solverName
        solverClassInstance.isInstalled = False
    return solverClassInstance
