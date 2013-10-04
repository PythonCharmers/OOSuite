function r = OpenOpt_ode_obj(t, x, W)
W.put('x', x);
W.put('t', t);
W.execute('r = f(x, t)');
r = W.get('r')';

end
