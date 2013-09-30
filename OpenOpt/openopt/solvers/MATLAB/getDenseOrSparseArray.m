function r = getDenseOrSparseArray(W, name)
W.execute(sprintf('is_sparse = isspmatrix(%s)',name));
is_sparse = W.get('is_sparse');
if is_sparse == 1
    W.execute(sprintf('I,J,Vals = find(%s); shape=%s.shape',name,name));
    I = W.get('I')+1;
    J = W.get('J')+1;
    Vals = W.get('Vals');
    shape = W.get('shape');
    r = sparse(I,J,Vals, shape(1),shape(2));
else
    r = W.get(name);
end
%~    disp(r)
end
