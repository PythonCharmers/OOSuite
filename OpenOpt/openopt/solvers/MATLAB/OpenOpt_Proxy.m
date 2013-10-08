W = Wormhole;
solver_id = W.get('solver_id');

%~is_fmincon = solver_id == 1;
%~is_fsolve = solver_id == 2;
is_ode = solver_id > 100 & solver_id < 200;
is_fmincon = false;
is_fsolve = false;
switch solver_id
    case 1
        is_fmincon = true;
    case 2
        is_fsolve = true;
    case 101
        solver = @ode15s;
    case 102
        solver = @ode45;
    case 103
        solver = @ode23;
    case 104
        solver = @ode23s;
    case 105
        solver = @ode23t;
    case 106
        solver = @ode23tb;
    case 107
        solver = @ode113;
end

obj = @(x) OpenOpt_obj(x, W);

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
elseif is_ode
    options = odeset;
    obj = @(x,t) OpenOpt_ode_obj(x, t, W);
    options.Jacobian = @(x,t) OpenOpt_ode_jac(x, t, W);
    options.RelTol = W.get('RelTol');
    options.AbsTol = W.get('AbsTol');
    times = W.get('times');
end

x0 = W.get('x0'); 

if ~is_ode
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
end

if is_fmincon
    [x,fval,exitflag,output] = fmincon(obj,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
elseif is_fsolve
    [x,fval,exitflag,output] = fsolve(obj,x0,options);
elseif is_ode
    [T,x] = solver(obj, times, x0, options);
end

W.put('xf',x);
if ~is_ode
    W.put('msg', output.message);
    W.execute('p.xf=p.xk=xf.flatten(); p.msg = msg');
else
    W.execute('p._xf=xf')
end

W.put('CycleCond', 0);
exit
