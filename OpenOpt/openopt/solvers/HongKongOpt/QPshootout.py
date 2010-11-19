import sys, time
from numpy import *
from QPSparse import QPSparse
from qlcp import qlcp
from openopt import *

if len(sys.argv) <= 1:
    print "usage: "+sys.argv[0]+" MPSfilename"
    sys.exit()
pname, e, Q, A, b, Aeq, beq, lb, ub, c0 = QPSparse(sys.argv[1])
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
print "Starting minimization with qlcp..."
t = time.clock()
if linalg.det(Q) != 0:
    print "Q is invertible, passing QI to qlcp..."
    QI = linalg.inv(Q)
else:
    QI = None
x = qlcp(Q, e, QI=QI, A=A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub)
t = time.clock() - t
print "time:", t, "seconds"
if x == None:
    print "Ray termination, sorry: no solution"
else:
    if smallproblem:
        print "minimizer:", x 
    print "minimum value:", c0 + dot(e,x) + 0.5*dot(x,dot(Q,x)) 
''''''
print "\n==================================================="
print "Starting minimization with Openopt CVX..."
t = time.clock()
p = QP(Q, e, A=A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub)
r = p.solve('cvxopt_qp', iprint = -1)
t = time.clock() - t
print "time:", t, "seconds"
x = p.xf
if smallproblem:
    print "minimizer:", x
print "minimum value:", c0 + dot(e,x) + 0.5*dot(x,dot(Q,x)), "(also:",p.ff,")"

print "\n==================================================="
print "Starting minimization with Openopt ralg..."
t = time.clock()
p = QP(Q, e, A=A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub)
r = p.solve('nlp:ralg', xtol=1e-10, alp=3.9, iprint=-10)
t = time.clock() - t
print "time:", t, "seconds"
x = p.xf
if smallproblem:
    print "minimizer:", x
print "minimum value:", c0 + dot(e,x) + 0.5*dot(x,dot(Q,x)), "(also:",p.ff,")"
''''''

