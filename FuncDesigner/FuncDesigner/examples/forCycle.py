from FuncDesigner import *
a, b, c = oovars('a', 'b', 'c')
f1, f2 = sin(a) + cos(b) - log2(c) + sqrt(b), sum(c) + c * cosh(b) / arctan(a) + c[0] * c[1] + c[-1] / (a * c.size)
f3 = f1*f2 + 2*a + sin(b) * (1+2*c.size + 3*f2.size) 
F = sin(f2)*f3 + 1
M = 15
for i in range(M):  F = 0.5*F + 0.4*f3*cos(f1+2*f2)
point = {a:1, b:2, c:[3, 4, 5]} # however, you'd better use numpy arrays instead of Python lists
print(F(point))
print(F.D(point))
print(F.D(point, a))
print(F.D(point, [b]))
print(F.D(point, fixedVars = [a, c])) 
"""
[ 4.63468686  0.30782902  1.21725266]
{a: array([-436.83015952,  204.25331181,  186.38788436]), b: array([ 562.63390316, -273.23484496, -256.32464645]), c: array([[ 618.96880254,  432.060652  ,  432.060652  ],
       [-154.59737546, -224.09819341, -154.59737546],
       [-118.67776819, -118.67776819, -169.66500367]])}
[-436.83015952  204.25331181  186.38788436]
{b: array([ 562.63390316, -273.23484496, -256.32464645])}
{b: array([ 562.63390316, -273.23484496, -256.32464645])}
"""
