function r = OpenOpt_jac(x, W)
W.put('x', x);
W.execute('r = df(x)');
r = getDenseOrSparseArray(W, 'r');
end
