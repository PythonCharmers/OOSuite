W = Wormhole;
solver_id = W.get('solver_id');

is_fmincon = solver_id == 1;
is_fsolve = solver_id == 2;
if is_fmincon
    A = getDenseOrSparseArray(W, 'A');
    b = W.get('b');
    Aeq = getDenseOrSparseArray(W, 'Aeq');
    beq = W.get('beq'); 
    lb = W.get('lb'); ub = W.get('ub'); nc = W.get('nc'); nh = W.get('nh');
    options = optimset('fmincon');
    options.GradObj = 'on';
    has_nonlinear = nc + nh > 0;
    handleConstraints = true;
elseif is_fsolve
    options = optimset('fsolve');
    options.Jacobian = 'on';
    has_nonlinear = false;
    handleConstraints = false;
end

x0 = W.get('x0'); 

options.TolFun = W.get('TolFun');
options.TolCon = W.get('TolCon');
options.TolX = W.get('TolX');
options.Display = 'off';
options.OutputFcn = @(x, optimValues, state) OpenOpt_iter(x, optimValues, state, W, handleConstraints);

if has_nonlinear
    nonlcon = @(x) OpenOpt_nlc(x,W,nc,nh);
    options.GradConstr = 'on';
else
    nonlcon = [];
end

obj = @(x) OpenOpt_obj(x, W);

if is_fmincon
    [x,fval,exitflag,output] = fmincon(obj,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
elseif is_fsolve
    [x,fval,exitflag,output] = fsolve(obj,x0,options);
end

W.put('xf',x);
W.put('msg', output.message);
W.execute('p.xf=p.xk=xf.flatten(); p.msg = msg');
W.put('CycleCond', 0);
exit
