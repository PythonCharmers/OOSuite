from SpaceFuncs import *

# 3D example
p1, p2 = Point(0, 2, 1), Point(4, 0, 5)
#p3 = point(10, 3, 4)
line1, line2 = Line(p1, p2), Line(p1+0.5*p2+1, p2+3*p1)

P1, P2 = skewLinesNearestPoints(line1, line2)
print('nearest points of the skew lines: ' + str((P1, P2)))
# (point([ 1.91780822,  1.04109589,  2.91780822]), point([ 2.43835616,  1.31506849,  2.53424658]))
print('distance between the points: ' + str(P1.distance(P2)))# 0.702246883177


# example with parameters in 5D space
from FuncDesigner import *
a, b, c = oovars('a', 'b', 'c')
p1, p2 = Point(0, 2, 3, 4, a), Point(4, 0, 3, b, c)
line1, line2 = Line(p1, p2), Line(p1+1, (1, 2, 3, 4, 5))
np1, np2 = skewLinesNearestPoints(line1, line2)
Parameters = {a:1, b:2, c:0.5}
print('for parameters %s\nnearest skew lines points are\n%s\n%s' % (Parameters,  np1(Parameters), np2(Parameters)))
# for parameters {a: 1, b: 2, c: 0.5}
# nearest skew lines points are
# [-0.08428446  2.04214223  3.          4.04214223  1.01053556]
# [ 1.          3.00438982  4.00438982  5.00438982  1.98683055]
