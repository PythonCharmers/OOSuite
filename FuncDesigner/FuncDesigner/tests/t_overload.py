from FuncDesigner import *

a, b, c = oovars('a', 'b', 'c') # or just oovars(3)

point1 = {a:[1, 2, 3], b:[4, 5, 6]}
point2 = {a:[1.5], b:[4, 5, 6]}
point3 = {a:[1, 2, 3], b:[4.5]}

t1 = (-a, a+b, a-b, norm(a), dot(a, b), dot(a, 5), dot(5, a), a*b, b/a, a/b, sin(a),  cos(a), tan(a), a**b, \
                           a**a, b**a, a**3, 3**a, [1, 2, 3]**a, a**[1, 2, 3], sqrt(a), log10(a), log(a), 10*a, sin(a), \
                           cos(a), sinh(a), cosh(a), arcsin(a/10), arccos(a/20), arctan(a), 5-a, a-5, 5*a, a*5, \
                           a*[1, 2, 3], [1, 2, 3]*a, 1/a, a/5, a.sum(),  a.prod())

t2 = (dot(a, b), dot(a, 5), dot(5, a))
testSuite = t1
#testSuite = t2

for i, point in enumerate([point1, point2, point3]):
    print '=' * 80
    print('checking point ' + str(i))
    for j, f in enumerate(testSuite):
            print('checking function ' +str(j))
            f.check_d1(point)


#f = a+a
#f.check_d1(somePoint) doesn't work with current oofun / DerApproximator implementation
