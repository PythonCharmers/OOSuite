from FuncDesigner import *
a, b, c = oovars('a', 'b', 'c')
f1, f2 = sin(a) + cos(b) - log2(c) + sqrt(b), sum(c) + c * cosh(b) / arctan(a) + c[0] * c[1] + c[-1] / (a * c.size)
f3 = f1*f2 + 2*a + sin(b) * (1+2*c.size + 3*f2.size)
f = (2*a*b*c + f1*f2 + f3 + dot(a+c, b+c)).named('asdf')
point = {a:1, b:2, c:[3, 4, 5]} # however, you'd better use numpy arrays instead of Python lists
print(f(point))
print(f.D(point))
print(f.D(point, a))
print(f.D(point, [b]))
print(f.D(point, fixedVars = [a, c]))
""" Expected output: 
[ 140.9337138   110.16255336   80.67870244]
{a: array([  69.75779959,   88.89020412,  109.93551537]), b: array([-23.10565554, -39.41138045, -59.08378522]), c: array([[  6.19249888,  38.261221  ,  38.261221  ],
       [ 29.68377935,  -0.18961959,  29.68377935],
       [ 23.03059873,  23.03059873,  -6.22406763]])}
[  69.75779959   88.89020412  109.93551537]
{b: array([-23.10565554, -39.41138045, -59.08378522])}
{b: array([-23.10565554, -39.41138045, -59.08378522])}
"""
