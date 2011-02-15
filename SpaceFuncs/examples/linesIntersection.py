from SpaceFuncs import *

# 2D example

Line1 = Line([0, 1], [1, 0])
Line2 = Line([0, 0], [1, 1])

intersectionPoint = Line1 & Line2
print('lines intersection: ' + str(intersectionPoint)) # [ 0.   0.5]

"""
Parametrized example in 4D space
!!! Pay attention that no check for intersection is performed !!!
Intersection point is obtained as middle of skewLinesNearestPoints(line1, line2)
"""
from FuncDesigner import *
a, b, c = oovars('a', 'b', 'c')
Line1 = Line([0, 1, a, b], [2*b+c, 1, a, 0])
Line2 = Line([b+a**2, 0, 3, 4], [a+3*c, exp(b/10), a, 4])
intersectionPoint = Line1 & Line2
params1 = {a:1, b:0.5, c:0.1}
params2 = {a:-1, b:-0.5, c:-0.1}
for params in [params1, params2]:
    print('for parameters %s intersection point coords are %s' %(params, intersectionPoint(params)))
    # [ 0.52035407  1.05178554  0.95025071  2.30666864]
    # [-2.29055032  1.0490853  -1.30895003  2.43251037]
    
