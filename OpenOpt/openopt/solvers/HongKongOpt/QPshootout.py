import sys, time
from numpy import *
from QPSparse import QPSparse
from QPSolve import QPSolve
from openopt import *

if len(sys.argv) <= 1:
    #print "usage: "+sys.argv[0]+" MPSfilename"
    #sys.exit()
    probName = 'cvxqp3_m.qps'
else:
    probName = sys.argv[1]
pname, e, Q, A, b, Aeq, beq, lb, ub, c0 = QPSparse(probName)
smallproblem = e.shape[0] <= 10
if smallproblem:    # print them only for small problems
    print "Problem's name: ", pname
    print "e:",e
    print "Q:\n",Q
    print "Aeq:\n",Aeq
    print "beq:",beq
    print "A:\n",A
    print "b:",b
    print "lb:",lb
    print "ub:",ub
    print "c0:",c0
else:
    print "n variables:", e.shape[0]
    if beq != None:
        print "n eq constr:", beq.shape[0] 
    if b != None:
        print "n ineq cons:", b.shape[0]
    if lb != None:
        print "n lower bounds:", lb.shape[0]
    if ub != None:
        print "n upper bounds:", lb.shape[0]
    if Q != None:
        print "problem is quadratic."

print "\n==================================================="
print "Starting minimization with QPSolve..."
t = time.clock()
x, retcode = QPSolve(e, Q, A=A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub)
t = time.clock() - t
print "time:", t, "seconds"
print "loop in LPCSolve iterated",retcode[2],"times"
if retcode[0] != 1:
    print "Ray termination, sorry: no solution"
else:
    if smallproblem:
        print "minimizer:", x 
    print "minimum value:", c0 + dot(e,x) + 0.5*dot(x,dot(Q,x)) 


# OpenOpt doesn't seem to like QP problems where lb == None or ub == None
nvars = e.shape[0]
if lb == None:
    lb = array([-Inf]*nvars) 
if ub == None:
    ub = array([Inf]*nvars) 
    
print "\n==================================================="
print "Starting minimization with Openopt CVX..."
t = time.clock()
p = QP(Q, e, A=A, b=b, Aeq=Aeq, beq=beq)#, lb=lb, ub=ub)
r = p.solve('cvxopt_qp', iprint = 0)
t = time.clock() - t
print "time:", t, "seconds"
x = p.xf
if smallproblem:
    print "minimizer:", x
print "minimum value:", c0 + dot(e,x) + 0.5*dot(x,dot(Q,x)), "(also:",p.ff,")"

print "\n==================================================="
print "Starting minimization with Openopt ralg..."
t = time.clock()
p = QP(Q, e, A=A, b=b, Aeq=Aeq, beq=beq)#, lb=lb, ub=ub)
r = p.solve('nlp:ralg', xtol=1e-10, alp=3.9, iprint=-10)
t = time.clock() - t
print "time:", t, "seconds"
x = p.xf
if smallproblem:
    print "minimizer:", x
print "minimum value:", c0 + dot(e,x) + 0.5*dot(x,dot(Q,x)), "(also:",p.ff,")"


