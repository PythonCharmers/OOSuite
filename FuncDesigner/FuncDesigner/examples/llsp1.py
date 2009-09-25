from FuncDesigner import *
from openopt import LLSP

# create some variables
a, b, c = oovars('a', 'b', 'c')
# or just a, b, c = oovars(3)

# start point is unused by lapack_dgelss and lapack_dgelss
# but it is required to set dimensions of variables
# also, it is used for some converters e.g. r = p.solve('nlp:ralg')
startPoint = {a:0, b:0, c:0} 

# overdetermined system of 4 linear equations with 3 variables
f = [2*a+3*b-4*c+5, 2*a+13*b+15, a+4*b+4*c-25, 20*a+30*b-4*c+50]

# assign prob
p = LLSP(f, startPoint)

# solve
r = p.solve('lapack_dgelss')

# print result
print r.xf

# Expected output:
# {a: array([-0.3091145]), b: array([-0.86376906]), c: array([ 4.03827441])}

