# Handling of FuncDesigner probs

from numpy import hstack, atleast_1d, cumsum, asfarray, asarray, zeros, \
ndarray, prod, isscalar, nan, array_equal, copy, array
#from nonOptMisc import scipyInstalled, isspmatrix, SparseMatrixConstructor#, DenseMatrixConstructor

from FDmisc import FuncDesignerException
from ooPoint import ooPoint
DenseMatrixConstructor = lambda shape: zeros(shape)
#isspmatrix = lambda *args: False

class FuncDesignerTranslator:
#    freeVars = []
#    fixedVars = []
    def __init__(self, PointOrVariables, **kwargs): #, freeVars=None, fixedVars=None
        #assert freeVars is not None or fixedVars is not None, 'at most one parameter of "fixedVars" and "freeVars" is allowed'
        #assert 'freeVars' not in kwargs, 'only "fixedVars" and "freeVars" arguments are allowed, not "freeVars"'
        
        if isinstance(PointOrVariables, dict):
            Point = PointOrVariables
            Variables = Point.keys()
            self._sizeDict = dict((v, asarray(PointOrVariables[v]).size) for v in PointOrVariables)
            self._shapeDict = dict((v, asarray(PointOrVariables[v]).shape) for v in PointOrVariables)
            # TODO: assert v.size (if provided) == PointOrVariables[v]).size
            # and same with shapes
        else:
            assert type(PointOrVariables) in [list, tuple, set]
            Variables = PointOrVariables
            self._sizeDict = dict((v, (v.size if hasattr(v, 'size') and isinstance(v.size, int) else 1)) for v in Variables)
            self._shapeDict = dict((v, (v.shape if hasattr(v, 'shape') else ())) for v in Variables)
            
        self._variables = Variables
        self.n = sum(self._sizeDict.values())
        
        oovar_sizes = list(self._sizeDict.values()) # FD: for opt oovars only
        oovar_indexes = cumsum([0] + oovar_sizes)

        self.oovarsIndDict = dict((v, (oovar_indexes[i], oovar_indexes[i+1])) for i, v in enumerate(Variables))
        
        # TODO: mb use oovarsIndDict here as well (as for derivatives?)
        
        #startDictData = [] #if fixedVars is None else [(v, startPoint[v]) for v in fixedVars]
        # TODO: involve fixed variables
        self._SavedValues = {'prevX':nan}
        def vector2point(x):
            isComplexArray = isinstance(x, ndarray) and str(x.dtype).startswith('complex')
            if isComplexArray:
                x = atleast_1d(array(x, copy=True))
            else:
                x = atleast_1d(array(x, copy=True, dtype=float)) 
            if array_equal(x, self._SavedValues['prevX']):
                return self._SavedValues['prevVal']
                
            
            # without copy() ipopt and probably others can replace it by noise after closing
            kw = {'skipArrayCast':True} if isComplexArray else {}
            r = ooPoint((v, x[oovar_indexes[i]:oovar_indexes[i+1]]) for i, v in enumerate(self._variables), **kw)
            
            self._SavedValues['prevVal'] = r
            self._SavedValues['prevX'] = copy(x)
            return r
        self.vector2point = vector2point
        
    point2vector = lambda self, point: asfarray(atleast_1d(hstack([(point[v] if v in point else zeros(self._shapeDict[v])) for v in self._variables])))
        
    def pointDerivative2array(self, pointDerivarive, useSparse = False,  func=None, point=None): 
        # useSparse can be True, False, 'auto'
        # !!!!!!!!!!! TODO: implement useSparse = 'auto' properly
        assert useSparse is False, 'sparsity is not implemented in FD translator yet'
#        if useSparse == 'auto' and not scipyInstalled:
#            useSparse = False
#        if useSparse is not False and not scipyInstalled:
#            raise FuncDesignerException('to handle sparse matrices you should have module scipy installed') 

        # however, this check is performed in other function (before this one)
        # and those constraints are excluded automaticvally

        n = self.n
        if len(pointDerivarive) == 0: 
            if func is not None:
                assert point is not None
                funcLen = func(point).size
#                if useSparse:
#                    return SparseMatrixConstructor((funcLen, n))
#                else:
                return DenseMatrixConstructor((funcLen, n))
            else:
                raise FuncDesignerException('unclear error, maybe you have constraint independend on any optimization variables') 

        key, val = list(pointDerivarive.items())[0]
        
        if isscalar(val) or (isinstance(val, ndarray) and val.shape == ()):
            val = atleast_1d(val)
        var_inds = self.oovarsIndDict[key]
        # val.size works in other way (as nnz) for scipy.sparse matrices
        funcLen = int(round(prod(val.shape) / (var_inds[1] - var_inds[0]))) 
        
        newStyle = 1
        
        # TODO: remove "useSparse = False", replace by code from FDmisc
        
        if useSparse is not False and newStyle:
            assert 0, 'unimplemented yet'
#            r2 = []
#            hasSparse = False
#            zeros_start_ind = 0
#            zeros_end_ind = 0
#            for i, var in enumerate(freeVars):
#                if var in pointDerivarive:#i.e. one of its keys
#                    
#                    if zeros_end_ind != zeros_start_ind:
#                        r2.append(SparseMatrixConstructor((funcLen, zeros_end_ind - zeros_start_ind)))
#                        zeros_start_ind = zeros_end_ind
#                    
#                    tmp = pointDerivarive[var]
#                    if isspmatrix(tmp): 
#                        hasSparse = True
#                    else:
#                        tmp = asarray(tmp) # else bug with scipy sparse hstack
#                    if tmp.ndim < 2:
#                        tmp = tmp.reshape(funcLen, prod(tmp.shape) // funcLen)
#                    r2.append(tmp)
#                else:
#                    zeros_end_ind  += oovar_sizes[i]
#                    hasSparse = True
#                    
#            if zeros_end_ind != zeros_start_ind:
#                r2.append(SparseMatrixConstructor((funcLen, zeros_end_ind - zeros_start_ind)))
#                
#            r3 = Hstack(r2) if hasSparse else hstack(r2)
#            if isspmatrix(r3) and r3.nnz > 0.25 * prod(r3.shape): r3 = r3.A
#            return r3            
            
#            r2 = []
#            hasSparse = False
#            for i, var in enumerate(freeVars):
#                if var in pointDerivarive:#i.e. one of its keys
#                    tmp = pointDerivarive[var]
#                    if isspmatrix(tmp): hasSparse = True
#                    if isinstance(tmp, float) or (isinstance(tmp, ndarray) and tmp.shape == ()):
#                        tmp = atleast_1d(tmp)
#                    if tmp.ndim < 2:
#                        tmp = tmp.reshape(funcLen, prod(tmp.shape) // funcLen)
#                    r2.append(tmp)
#                else:
#                    r2.append(SparseMatrixConstructor((funcLen, oovar_sizes[i])))
#                    hasSparse = True
#            r3 = Hstack(r2) if hasSparse else hstack(r2)
#            if isspmatrix(r3) and r3.nnz > 0.25 * prod(r3.shape): r3 = r3.A
#            return r3
        else:
            if funcLen == 1:
                r = DenseMatrixConstructor(n)
            else:
#                if useSparse:
#                    r = SparseMatrixConstructor((n, funcLen))
#                else:
                r = DenseMatrixConstructor((n, funcLen))            
            for key, val in pointDerivarive.items():
                # TODO: remove indexes, do as above for sparse 
                indexes = self.oovarsIndDict[key]
#                if not useSparse and isspmatrix(val): val = val.A
                if r.ndim == 1:
                    r[indexes[0]:indexes[1]] = val if isscalar(val) else val.flatten()
                else:
                    r[indexes[0]:indexes[1], :] = val.T
#            if useSparse is True and funcLen == 1: 
#                return SparseMatrixConstructor(r)
            else: 
                return r.T if r.ndim > 1 else r.reshape(1, -1)
                
