__docformat__ = "restructuredtext en"
from time import time, clock
from numpy import asfarray, copy, inf, nan, isfinite, ones, ndim, all, atleast_1d, any, isnan, \
array_equiv, asscalar, asarray, where, ndarray, isscalar, matrix
from setDefaultIterFuncs import stopcase,  SMALL_DELTA_X,  SMALL_DELTA_F
from check import check
import copy
import os, string
from ooMisc import isSolved, killThread
from string import lower
from baseProblem import ProbDefaults
from baseSolver import baseSolver
from nonOptMisc import getSolverFromStringName
hasMultiprocessingModule = True
try:
    from multiprocessing import Pool
except:
    hasMultiprocessingModule = False
#from openopt.kernel.ooMisc import __solverPaths__
ConTolMultiplier = 0.8

#if __solverPaths__ is None:
#    __solverPaths__ = {}
#    file = string.join(__file__.split(os.sep)[:-1], os.sep)
#    for root, dirs, files in os.walk(os.path.dirname(file)+os.sep+'solvers'):
#        rd = root.split(os.sep)
#        if '.svn' in rd: continue
#        rd = rd[rd.index('solvers')+1:]
#        for file in files:
#            print file
#            if len(file)>6 and file[-6:] == '_oo.py':
#                __solverPaths__[file[:-6]] = 'openopt.solvers.' + string.join(rd,'.') + '.'+file[:-3]

#import pickle
#f = open('solverPaths.py', 'w')
#solverPaths = pickle.load(f)


def runProbSolver(p_, solver_str_or_instance=None, *args, **kwargs):
    #p = copy.deepcopy(p_, memo=None, _nil=[])
    p = p_
    if args is not (): p.err('unexpected args for p.solve()')
    if hasattr(p, 'was_involved'): p.err("""You can't run same prob instance for twice. 
    Please reassign prob struct. 
    You can avoid it via using FuncDesigner oosystem.""")
    else: p.was_involved = True

    if solver_str_or_instance is None:
        if hasattr(p, 'solver'): solver_str_or_instance = p.solver
        elif 'solver' in kwargs.keys(): solver_str_or_instance = kwargs['solver']

    if type(solver_str_or_instance) is str and ':' in solver_str_or_instance:
        isConverter = True
        probTypeToConvert,  solverName = solver_str_or_instance.split(':', 1)
        converterName = lower(p.probType)+'2'+probTypeToConvert
        converter = getattr(p, converterName)
        p.solver = getSolverFromStringName(p, solverName)
        solver_params = {}
        #return converter(solverName, *args, **kwargs)
    else:
        isConverter = False
        if solver_str_or_instance is None:
            p.err('you should provide name of solver')
        elif type(solver_str_or_instance) is str:
            p.solver = getSolverFromStringName(p, solver_str_or_instance)
        else: # solver_str_or_instance is oosolver
            p.solver = solver_str_or_instance
            for key, value  in solver_str_or_instance.fieldsForProbInstance.iteritems():
                setattr(p, key, value)
    p.isConverterInvolved = isConverter


    if 'debug' in kwargs.keys():
       p.debug =  kwargs['debug']
    
    #p.solver = solverClass()
#    p.solverName = p.solver.__name__
#    setattr(p, p.solverName, EmptyClass())
    solver = p.solver.__solver__

    for key, value in kwargs.iteritems():
        if hasattr(p.solver, key):
            if isConverter:
                solver_params[key] = value
            else:
                setattr(p.solver, key, value)
        elif hasattr(p, key):
            setattr(p, key, value)
        else: p.warn('incorrect parameter for prob.solve(): "' + str(key) + '" - will be ignored (this one has been not found in neither prob nor ' + p.solver.__name__ + ' solver parameters)')

    p.iterValues = EmptyClass()

    p.iterCPUTime = []
    p.iterTime = []
    p.iterValues.x = [] # iter points
    p.iterValues.f = [] # iter ObjFunc Values
    p.iterValues.r = [] # iter MaxResidual
    p.iterValues.rt = [] # iter MaxResidual Type: 'c', 'h', 'lb' etc
    p.iterValues.ri = [] # iter MaxResidual Index
    if p._baseClassName == 'NonLin':p.iterValues.nNaNs = [] # number of constraints equal to numpy.nan



    if p.goal in ['max','maximum']: p.invertObjFunc = True

    #TODO: remove it!
    p.advanced = EmptyClass()

    p.istop = 0
    p.iter = 0
    p.graphics.nPointsPlotted = 0
    p.finalIterFcnFinished = False
    #for fn in p.nEvals.keys(): p.nEvals[fn] = 0 # NB! f num is used in LP/QP/MILP/etc stop criteria check

    p.msg = ''
    if not type(p.callback) in (list,  tuple): p.callback = [p.callback]
    if hasattr(p, 'xlabel'): p.graphics.xlabel = p.xlabel
    if p.graphics.xlabel == 'nf': p.iterValues.nf = [] # iter ObjFunc evaluation number

    p._Prepare()
    for fn in ['FunEvals', 'Iter', 'Time', 'CPUTime']:
        if hasattr(p,'min'+fn) and hasattr(p,'max'+fn) and getattr(p,'max'+fn) < getattr(p,'min'+fn):
            p.warn('min' + fn + ' (' + str(getattr(p,'min'+fn)) +') exceeds ' + 'max' + fn + '(' + str(getattr(p,'max'+fn)) +'), setting latter to former')
            setattr(p,'max'+fn, getattr(p,'min'+fn))

    for fn in ['maxFunEvals', 'maxIter']: setattr(p, fn, int(getattr(p, fn)))# to prevent warnings from numbers like 1e7

    if hasattr(p, 'x0'): p.x0 = atleast_1d(asfarray(p.x0).copy())
    for fn in ['lb', 'ub', 'b', 'beq']:
        if hasattr(p, fn):
            fv = getattr(p, fn)
            if fv != None:# and fv != []:
                setattr(p, fn, asfarray(fv, dtype='float').flatten())
            else:
                setattr(p, fn, asfarray([]))


#    if p.lb.size == 0:
#        p.lb = -inf * ones(p.n)
#    if p.ub.size == 0:
#        p.ub = inf * ones(p.n)

    p.stopdict = {}

    for s in ['b','beq']:
        if hasattr(p, s): setattr(p, 'n'+s, len(getattr(p, s)))

    #if p.probType not in ['LP', 'QP', 'MILP', 'LLSP']: p.objFunc(p.x0)

    p.isUC = p._isUnconstrained()
    if p.solver.__isIterPointAlwaysFeasible__ is True or \
    (not p.solver.__isIterPointAlwaysFeasible__ is False and p.solver.__isIterPointAlwaysFeasible__(p)):
        assert p.data4TextOutput[-1] == 'log10(maxResidual)'
        p.data4TextOutput = p.data4TextOutput[:-1]
    elif p.useScaledResidualOutput:
        p.data4TextOutput[-1] = 'log10(MaxResidual/ConTol)'

    if p.showFeas and p.data4TextOutput[-1] != 'isFeasible': p.data4TextOutput.append('isFeasible')

    if not p.solver.iterfcnConnected:
        if SMALL_DELTA_X in p.kernelIterFuncs: p.kernelIterFuncs.pop(SMALL_DELTA_X)
        if SMALL_DELTA_F in p.kernelIterFuncs: p.kernelIterFuncs.pop(SMALL_DELTA_F)

    if not p.solver._canHandleScipySparse:
        if hasattr(p.A, 'toarray'): p.A = p.A.toarray()
        if hasattr(p.Aeq, 'toarray'): p.Aeq = p.Aeq.toarray()
    
    if isinstance(p.A, ndarray) and type(p.A) != ndarray: # numpy matrix
        p.A = p.A.A 
    if isinstance(p.Aeq, ndarray) and type(p.Aeq) != ndarray: # numpy matrix
        p.Aeq = p.Aeq.A 

    if hasattr(p, 'optVars'):
        p.err('"optVars" is deprecated, use "freeVars" instead ("optVars" is not appropriate for some prob types, e.g. systems of (non)linear equations)')

#    p.xf = nan * ones([p.n, 1])
#    p.ff = nan
    #todo : add scaling, etc
    p.primalConTol = p.contol
    p.contol *= ConTolMultiplier

    p.timeStart = time()
    p.cpuTimeStart = clock()


    ############################
    # Start solving problem:

    if p.iprint >= 0:
        p.disp('-'*50)
        s = 'solver: ' +  p.solver.__name__ +  '   problem: ' + p.name + '    type: %s' % p.probType
        if p.showGoal: s += '   goal: ' + p.goal
        p.disp(s)

    p.extras = {}

    try:
        if isConverter:
            # TODO: will R be somewhere used?
            R = converter(solverName, **solver_params)
        else:
            nErr = check(p)
            if nErr: p.err("prob check results: " +str(nErr) + "ERRORS!")#however, I guess this line will be never reached.
            p.iterfcn(p.x0)
            solver(p)
#    except killThread:
#        if p.plot:
#            print 'exiting pylab'
#            import pylab
#            if hasattr(p, 'figure'):
#                print 'closing figure'
#                #p.figure.canvas.draw_drawable = lambda: None
#                pylab.ioff()
#                pylab.close()
#                #pylab.draw()
#            #pylab.close()
#            print 'pylab exited'
#        return None
    except isSolved:
#        p.fk = p.f(p.xk)
#        p.xf = p.xk
#        p.ff = p.objFuncMultiple2Single(p.fk)

        if p.istop == 0: p.istop = 1000
    ############################
    p.contol = p.primalConTol

    # Solving finished
    p.isFinished = True
    if not hasattr(p, 'xf') and not hasattr(p, 'xk'): p.xf = p.xk = ones(p.n)*nan
    if hasattr(p, 'xf') and (not hasattr(p, 'xk') or array_equiv(p.xk, p.x0)): p.xk = p.xf
    if not hasattr(p,  'xf') or all(p.xf==nan): p.xf = p.xk
    
    if p.isFeas(p.xf) and (not p.probType=='MINLP' or p.discreteConstraintsAreSatisfied(p.xf)):
        p.isFeasible = True
    else: p.isFeasible = False
    p.fk = p.objFunc(p.xk)
    if not hasattr(p,  'ff') or any(p.ff==nan): 
        p.iterfcn, tmp_iterfcn = lambda *args: None, p.iterfcn
        p.ff = p.objFunc(p.xf)
        p.iterfcn = tmp_iterfcn

    if not hasattr(p, 'fk'): p.fk = p.ff
    if p.invertObjFunc:  p.fk, p.ff = -p.fk, -p.ff

    if asfarray(p.ff).size > 1: p.ff = p.objFuncMultiple2Single(p.fk)

    #p.ff = p.objFuncMultiple2Single(p.ff)
    #if not hasattr(p, 'xf'): p.xf = p.xk
    if type(p.xf) in (list, tuple) or isscalar(p.xf): p.xf = asarray(p.xf)
    p.xf = p.xf.flatten()
    p.rf = p.getMaxResidual(p.xf)

    if not p.isFeasible and p.istop > 0: p.istop = -100-p.istop/1000.0
    p.stopcase = stopcase(p)

    p.xk, p.rk = p.xf, p.rf
    if p.invertObjFunc: 
        p.fk = -p.ff
        p.iterfcn(p.xf, -p.ff, p.rf)
    else: 
        p.fk = p.ff
        p.iterfcn(p.xf, p.ff, p.rf)

    p.__finalize__()
    if not p.storeIterPoints: delattr(p.iterValues, 'x')

    r = OpenOptResult(p)

    #TODO: add scaling handling!!!!!!!
#    for fn in ('df', 'dc', 'dh', 'd2f', 'd2c', 'd2h'):
#        if hasattr(p, '_' + fn): setattr(r, fn, getattr(p, '_'+fn))

    p.invertObjFunc = False
    
    if p.isFDmodel:
        p.x0 = p._x0

    finalTextOutput(p, r)
    if not hasattr(p, 'isManagerUsed') or p.isManagerUsed == False: 
        finalShow(p)
    return r

##################################################################
def finalTextOutput(p, r):
    if p.iprint >= 0:
        if p.msg is not '':  
            p.disp("istop: " + str(r.istop) + ' (' + p.msg +')')
        else: 
            p.disp("istop: " + str(r.istop))

        p.disp('Solver:   Time Elapsed = ' + str(r.elapsed['solver_time']) + ' \tCPU Time Elapsed = ' + str(r.elapsed['solver_cputime']))
        if p.plot:
            p.disp('Plotting: Time Elapsed = '+ str(r.elapsed['plot_time'])+ ' \tCPU Time Elapsed = ' + str(r.elapsed['plot_cputime']))
        
        # TODO: add output of NaNs number in constraints (if presernt)
        if p.useScaledResidualOutput: 
            rMsg = 'max(residuals/requiredTolerances) = %g' % (r.rf / p.contol)
        else:
            rMsg = 'MaxResidual = %g' % r.rf
        if not p.isFeasible:
            nNaNs = (len(where(isnan(p.c(p.xf)))[0]) if hasattr(p, 'c') else 0) + (len(where(isnan(p.h(p.xf)))[0]) if hasattr(p, 'h') else 0)
            if nNaNs == 0:
                nNaNsMsg = ''
            elif nNaNs == 1:
                nNaNsMsg = '1 constraint is equal to NaN, '
            else:
                nNaNsMsg = ('%d constraints are equal to NaN, ' % nNaNs)
            p.disp('NO FEASIBLE SOLUTION is obtained (%s%s, objFunc = %0.8g)' % (nNaNsMsg,  rMsg, r.ff))
        else:
            msg = "objFunValue: " + (p.finalObjFunTextFormat % r.ff)
            if not p.isUC: msg += ' (feasible, %s)' % rMsg
            p.disp(msg)

##################################################################
def finalShow(p):
    if not p.plot: return
    pylab = __import__('pylab')
    pylab.ioff()
    if p.show:
        pylab.show()

class OpenOptResult: 
    # TODO: implement it
    #extras = EmptyClass() # used for some optional output
    def __init__(self, p):
        self.rf = asscalar(asarray(p.rf))
        self.ff = asscalar(asarray(p.ff))
        if p.isFDmodel:
            self.xf = dict([(v, asscalar(val) if isinstance(val, ndarray) and val.size ==1 else val) for v, val in p.xf.items()])
            if not hasattr(self, '_xf'):
                self._xf = dict([(v.name, asscalar(val) if isinstance(val, ndarray) and val.size ==1 else val) for v, val in p.xf.items()])
            def c(*args):
                r = []
                for arg in args:
                    tmp = [(self._xf[elem] if isinstance(elem,  str) else self.xf[elem]) for elem in (arg.tolist() if isinstance(arg, ndarray) else arg if type(arg) in (tuple, list) else [arg])]
                    tmp = [asscalar(item) if type(item) in (ndarray, matrix) and item.size == 1 else item for item in tmp]
                    r.append(tmp)
                r = r[0] if len(args) == 1 else r
                if len(args) == 1 and type(r) in (list, tuple) and len(r) >1: r = asfarray(r)
                return r
                
#                condIterable = len(args) == 1 and isinstance(args[0], (list, tuple))# may be tuple, list, oolist
#                Args = args[0] if condIterable else args
#                r = [(self._xf[arg] if isinstance(arg,  str) else self.xf[arg]) for arg in (Args.tolist() if isinstance(Args, ndarray) else Args)]
#                r = [asscalar(item) if type(item) in (ndarray, matrix) and item.size == 1 else item for item in r]
#                return r if condIterable else r if len(args) > 1 else r[0] # if len(args)==1 else r
            self.__call__ = c
        else:
            self.xf = p.xf

        self.elapsed = dict()
        self.elapsed['solver_time'] = round(100.0*(time() - p.timeStart))/100.0
        self.elapsed['solver_cputime'] = clock() - p.cpuTimeStart

        for fn in ('ff', 'istop', 'duals', 'isFeasible', 'msg', 'stopcase', 'iterValues',  'special', 'extras'):
            if hasattr(p, fn):  setattr(self, fn, getattr(p, fn))

        if hasattr(p.solver, 'innerState'):
            self.extras['innerState'] = p.solver.innerState

        self.solverInfo = dict()
        for fn in ('homepage',  'alg',  'authors',  'license',  'info', 'name'):
            self.solverInfo[fn] =  getattr(p.solver,  '__' + fn + '__')

            # note - it doesn't work for len(args)>1 for current Python ver  2.6
            #self.__getitem__ = c # = self.__call__
            
        if p.plot:
            #for df in p.graphics.drawFuncs: df(p)    #TODO: include time spent here to (/cpu)timeElapsedForPlotting
            self.elapsed['plot_time'] = round(100*p.timeElapsedForPlotting[-1])/100 # seconds
            self.elapsed['plot_cputime'] = p.cpuTimeElapsedForPlotting[-1]
        else:
            self.elapsed['plot_time'] = 0
            self.elapsed['plot_cputime'] = 0

        self.elapsed['solver_time'] -= self.elapsed['plot_time']
        self.elapsed['solver_cputime'] -= self.elapsed['plot_cputime']

        self.evals = dict([(key, val if type(val) == int else round(val *10) /10.0) for key, val in p.nEvals.items()])
        self.evals['iter'] = p.iter
        
class EmptyClass: pass
