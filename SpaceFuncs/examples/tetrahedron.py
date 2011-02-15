from SpaceFuncs import *

T = Tetrahedron((1, 2, 15), (2, 8, 4), (4, 6.5, 7), (14, 15, 16))
print('vertices: ' + str(T.vertices)) # [Point([ 1,  2, 15]), Point([2, 8, 4]), Point([ 4. ,  6.5,  7. ]), Point([14, 15, 16])]
print('centroid: ' + str(T.M)) # [  5.25    7.875  10.5  ]
print('volume: ' + str(T.V)) # 53.1666666667
print('insphere center: ' + str(T.I)) # [ 3.60473533  7.02965971  7.46558431]
print('insphere radius: ' + str(T.r)) # 0.707155232274
print('circum sphere center: ' + str(T.O)) # [ -4.88910658  20.73510972  17.50195925]
print('circum sphere radius: ' + str(T.R)) # 19.797618861
# alternatively you can use complete names:
# T.volume,T.centroid, etc

## Let's solve some parametrized problems
#from FuncDesigner import *
#from openopt import NLP, NLSP
#
## let's create parameterized triangle :
#a,b,c, = oovars(3)
#T = Tetrahedron((1,2,a),(2,b,4),(c,6.5,7))
#
## let's create an initial estimation to the problems below
#startValues = {a:1, b:0.5, c:0.1} # you could mere set any, but sometimes a good estimation matters
#
#
