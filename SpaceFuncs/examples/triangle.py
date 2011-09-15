from SpaceFuncs import Triangle, Plot

# Consider a triangle in 3D space
T = Triangle((1, 2, 5), (2, 8, 4), (4, 6.5, 7))
print('inscribed circle radius: '  + str(T.r)) # 1.38278208828
print('circum circle radius: '  + str(T.R)) # 3.16955464416
print('area: ' + str(T.S)) # 10.9487442202
print('perimeter: ' + str(T.P)) # 15.8358201383
print('semiperimeter: ' + str(T.p)) # 7.91791006913
print('vertices: ' + str(T.vertices)) # [point([1, 2, 5]), point([2, 8, 4]), point([ 4. ,  6.5,  7. ])]
print('centroid: ' + str(T.M)) # [ 2.33333333  5.5         5.33333333]
print('orthocenter: ' + str(T.H)) # [ 3.05839416  6.46715328  5.86131387]
print('circum circle center: ' + str(T.O)) # [ 1.97080292  5.01642336  5.06934307]
print('sides: ' + str(T.sides)) # [3.905124837953327, 5.7662812973353983, 6.164414002968976]
print('angles: ' + str(T.angles)) # [ 0.66370155  1.1424067   1.33548441]
# alternatively you can use complete names:
# T.area, T.perimeter,T.semiperimeter

# Let's solve some parametrized problems
from FuncDesigner import *
from openopt import NLP, SNLE

# let's create parameterized triangle :
a,b,c = oovars(3)
T = Triangle((1,2,a),(2,b,4),(c,6.5,7))

# let's create an initial estimation to the problems below
startValues = {a:1, b:0.5, c:0.1} # you could mere set any, but sometimes a good estimation matters

# let's find an a,b,c values wrt r = 1.5 with required tolerance 10^-5, R = 4.2 and tol 10^-4, a+c == 2.5 wrt tol 10^-7
# if no tol is provided, p.contol is used (default 10^-6)
equations = [(T.r == 1.5)(tol=1e-5) , (T.R == 4.2)(tol=1e-4), (a+c == 2.5)(tol=1e-7)]
prob = SNLE(equations, startValues)
result = prob.solve('nssolve', iprint = 0) # nssolve is name of the solver involved
print('\nsolution has%s been found' % ('' if result.stopcase > 0 else ' not'))
print('values:' + str(result(a, b, c))) # [1.5773327492140974, -1.2582702179532217, 0.92266725078590239]
print('triangle sides: '+str(T.sides(result))) # [8.387574299361475, 7.0470774415247455, 4.1815836020856336]
print('orthocenter of the triangle: ' + str(T.H(result))) # [ 0.90789867  2.15008869  1.15609611]

# let's find minimum inscribed radius subjected to the constraints a<1.5, a>-1, b<0, a+2*c<4,  log(1-b)<2] : 
objective = T.r
prob = NLP(objective, startValues, constraints = [a<1.5, a>-1, b<0, a+2*c<4,  log(1-b)<2])
result1 = prob.minimize('ralg', iprint = 0) # ralg is name of the solver involved, see http://openopt.org/ralg for details
print('\nminimal inscribed radius: %0.3f' % T.r(result1)) #  1.321
print('optimal values:' + str(result1(a, b, c))) # [1.4999968332804028, 2.7938728907900973e-07, 0.62272481283890913]

#let's find minimum outscribed radius subjected to the constraints a<1.5, a>-1, b<0, a+2*c<4,  log(1-b)<2] : 
prob = NLP(T.R, startValues, constraints = (a<1.5, a>-1, b<0, (a+2*c<4)(tol=1e-7),  log(1-b)<2))
result2 = prob.minimize('ralg', iprint = 0) 
print('\nminimal outscribed radius: %0.3f' % T.R(result2)) # 3.681
print('optimal values:' + str(result2(a, b, c))) # [1.499999901863762, -1.7546960034401648e-06, 1.2499958739399943]



