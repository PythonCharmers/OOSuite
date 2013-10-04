function r = OpenOpt_ode_jac(t, x, W)
W.put('x', x);
W.put('t', t);
W.execute('r = df(x, t)');
r = getDenseOrSparseArray(W, 'r');
end
