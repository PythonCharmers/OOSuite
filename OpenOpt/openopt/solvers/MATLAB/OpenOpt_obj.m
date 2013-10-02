function [r d]= OpenOpt_obj(x, W)
W.put('x', x);
W.execute('r = f(x)');
r = W.get('r')';

if nargout > 1
    W.execute('r = df(x); is_sparse = isspmatrix(r)');
    d = getDenseOrSparseArray(W, 'r');
end
end
