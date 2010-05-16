# Handling of FuncDesigner probs

from numpy import empty, hstack, vstack, asfarray, all, atleast_1d, cumsum, asarray, zeros,  ndarray, prod, ones, isscalar
#from nonOptMisc import scipyInstalled, Hstack, Vstack, Find, isspmatrix, SparseMatrixConstructor, DenseMatrixConstructor, Bmat

from misc import FuncDesignerException
from ooPoint import ooPoint
DenseMatrixConstructor = lambda shape: zeros(shape)
isspmatrix = lambda *args: False

class FuncDesignerTranslator:
#    freeVars = []
#    fixedVars = []
    def __init__(self, PointOrVariables, **kwargs): #, freeVars=None, fixedVars=None
        #assert freeVars is not None or fixedVars is not None, 'at most one parameter of "fixedVars" and "freeVars" is allowed'
        #assert 'optVars' not in kwargs, 'only "fixedVars" and "freeVars" arguments are allowed, not "optVars"'
        
        if isinstance(PointOrVariables, dict):
            Point = PointOrVariables
            Variables = Point.keys()
            self._sizeDict = dict([(v, asarray(PointOrVariables[v]).size) for v in PointOrVariables])
            self._shapeDict = dict([(v, asarray(PointOrVariables[v]).shape) for v in PointOrVariables])
            # TODO: assert v.size (if provided) == PointOrVariables[v]).size
            # and same with shapes
        else:
            assert type(pointOrVariables) in [list, tuple, set]
            Variables = PointOrVariables
            self._sizeDict = dict([(v, (v.size if hasattr(v, 'size') else 1)) for v in Variables])
            self._shapeDict = dict([(v, (v.shape if hasattr(v, 'shape') else ())) for v in Variables])
            
        self.n = sum(self._sizeDict.values())
        
        oovar_sizes = self._sizeDict.values() # FD: for opt oovars only
        oovar_indexes = cumsum([0] + oovar_sizes)

        self.oovarsIndDict = dict([(v, (oovar_indexes[i], oovar_indexes[i+1])) for i, v in enumerate(Variables)])
        
        # TODO: mb use oovarsIndDict here as well (as for derivatives?)
        
        #startDictData = [] #if fixedVars is None else [(v, startPoint[v]) for v in fixedVars]
        #self.vector2point = lambda x: ooPoint(startDictData + [(v, x[oovar_indexes[i]:oovar_indexes[i+1]]) for i, v in enumerate(Variables)])
        self.vector2point = lambda x: ooPoint([(v, atleast_1d(x)[oovar_indexes[i]:oovar_indexes[i+1]]) for i, v in enumerate(Variables)])
        
    point2vector = lambda point: atleast_1d(hstack([(point[v] if v in point else zeros(self._shapeDict[v])) for v in self.freeVars]))
        
    def pointDerivative2array(self, pointDerivarive, asSparse = False,  func=None, point=None): 
        # asSparse can be True, False, 'auto'
        # !!!!!!!!!!! TODO: implement asSparse = 'auto' properly
        if asSparse == 'auto' and not scipyInstalled:
            asSparse = False
        if asSparse is not False and not scipyInstalled:
            raise FuncDesignerException('to handle sparse matrices you should have module scipy installed') 

        # however, this check is performed in other function (before this one)
        # and those constraints are excluded automaticvally

        n = self.n
        if len(pointDerivarive) == 0: 
            if func is not None:
                assert point is not None
                funcLen = func(point).size
                if asSparse:
                    return SparseMatrixConstructor((funcLen, n))
                else:
                    return DenseMatrixConstructor((funcLen, n))
            else:
                raise FuncDesignerException('unclear error, maybe you have constraint independend on any optimization variables') 

        key, val = pointDerivarive.items()[0]
        
        if isscalar(val) or (isinstance(val, ndarray) and val.shape == ()):
            val = atleast_1d(val)
        var_inds = self.oovarsIndDict[key]
        # val.size works in other way (as nnz) for scipy.sparse matrices
        funcLen = int(round(prod(val.shape) / (var_inds[1] - var_inds[0]))) 
        
        newStyle = 1
        
        if asSparse is not False and newStyle:
            r2 = []
            hasSparse = False
            for i, var in enumerate(optVars):
                if var in pointDerivarive:#i.e. one of its keys
                    tmp = pointDerivarive[var]
                    if isspmatrix(tmp): hasSparse = True
                    if isinstance(tmp, float) or (isinstance(tmp, ndarray) and tmp.shape == ()):
                        tmp = atleast_1d(tmp)
                    if tmp.ndim < 2:
                        tmp = tmp.reshape(funcLen, prod(tmp.shape) // funcLen)
                    r2.append(tmp)
                else:
                    r2.append(SparseMatrixConstructor((funcLen, oovar_sizes[i])))
                    hasSparse = True
            r3 = Hstack(r2) if hasSparse else hstack(r2)
            if isspmatrix(r3) and r3.nnz > 0.25 * prod(r3.shape): r3 = r3.A
            return r3
        else:
            if funcLen == 1:
                r = DenseMatrixConstructor(n)
            else:
                if asSparse:
                    r = SparseMatrixConstructor((n, funcLen))
                else:
                    r = DenseMatrixConstructor((n, funcLen))            
            for key, val in pointDerivarive.items():
                # TODO: remove indexes, do as above for sparse 
                indexes = self.oovarsIndDict[key]
                if not asSparse and isspmatrix(val): val = val.A
                if r.ndim == 1:
                    r[indexes[0]:indexes[1]] = val.flatten()
                else:
                    r[indexes[0]:indexes[1], :] = val.T
            if asSparse and funcLen == 1: 
                return SparseMatrixConstructor(r)
            else: 
                return r.T if r.ndim > 1 else r.reshape(1, -1)
                