function istop = OpenOpt_iter(x, optimValues, state, W)
W.put('xk', x);
W.put('fk', optimValues.fval);
W.put('rk', optimValues.constrviolation);
W.execute('p.iterfcn(xk.flatten(),fk,rk); istop = p.istop');
istop = W.get('istop') ~= 0;
