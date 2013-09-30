function [c, ceq, gradc, gradceq] = OpenOpt_nlc(x, W, nc, nh)
%~c(1) = x(1)^2/9 + x(2)^2/4 - 1;
%~c(2) = x(1)^2 - x(2) - 1;
%~ceq = [];
%~
%~if nargout > 2
%~    gradc = [2*x(1)/9, 2*x(1); ...
%~             x(2)/2, -1];
%~    gradceq = [];
%~end
%~disp('OpenOpt_nlc')
%~disp(nargout)
W.put('x', x);
c = []; ceq  = []; gradc = []; gradceq = [];
if nc > 0
    W.execute('r = c(x)');
    c = W.get('r');
    if nargout > 2
        W.execute('r = dc(x)');
        gradc = getDenseOrSparseArray(W, 'r')';%W.get('r')';
    end
end
if nh > 0
    W.execute('r = h(x)');
    ceq = W.get('r');
    if nargout > 2
        W.execute('r = dh(x)');
        gradceq = getDenseOrSparseArray(W, 'r')';%W.get('r')';
    end
end

%~if nargout > 2
%~    W.execute('r = dc(x)');
%~    %~d = W.get('r')';
%~    gradc = W.get('r')';
%~    W.execute('r = dh(x)');
%~    gradceq = W.get('r')';
%~    %~disp('gradient:')
%~    %~disp(d)
%~    end
%~end
