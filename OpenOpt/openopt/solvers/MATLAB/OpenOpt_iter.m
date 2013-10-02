function istop = OpenOpt_iter(x, optimValues, state, W, handleConstraints)
W.put('xk', x);
W.put('fk', optimValues.fval);
if handleConstraints
    W.put('rk', optimValues.constrviolation);
    W.execute('p.iterfcn(xk.flatten(),fk,rk); istop = p.istop');
else
    W.execute('p.iterfcn(xk.flatten(),fk); istop = p.istop');
end
istop = W.get('istop') ~= 0;
