from FuncDesigner import *
from numpy import *
a = oovar('a')
b = oovar('b')
point = {a: 1, b: 2}

f = oofun(lambda x,y: hstack((x+10*y, 2*x+100*y, 3*x+1000*y)), d=lambda x,y:array([[1,10],[1,100], [1, 1000]]),input=[a,b])
#point = {a:[1.5], b:[4, 5, 6]}
#f = [1, 2, 3]**a
f = f[2:3]*4 + 5
print f.D(point)

f.check_d1(point)
