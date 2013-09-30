W = Wormhole;

A = getDenseOrSparseArray(W, 'A');
b = W.get('b');
Aeq = getDenseOrSparseArray(W, 'Aeq');
beq = W.get('beq'); 
lb = W.get('lb'); ub = W.get('ub'); x0 = W.get('x0'); nc = W.get('nc'); nh = W.get('nh');
options = optimset('fmincon');
options.GradObj = 'on';
options.TolFun = W.get('TolFun');
options.TolCon = W.get('TolCon');
options.TolX = W.get('TolX');

options.OutputFcn = @(x, optimValues, state) OpenOpt_iter(x, optimValues, state, W);
if nc + nh > 0
    nonlcon = @(x) OpenOpt_nlc(x,W,nc,nh);
    options.GradConstr = 'on';
else
    nonlcon = [];
end
obj = @(x) OpenOpt_obj(x, W);
[x,fval,exitflag,output] = fmincon(obj,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
W.put('xf',x);
W.put('msg', output.message);
W.execute('p.xf=p.xk=xf.flatten(); p.msg = msg');
W.put('CycleCond', 0);
exit
