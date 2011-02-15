from SpaceFuncs import *

# 3D example
p1, p2, p3 = Point(1, 0, 0), Point(0, 2, 0), Point(0, 0, 4)
plane = Plane(p1, p2, p3)

P1, P2 = Point(0, 0, 0), Point(0, 0, 1)

print('P1 projection: ' + str(P1.projection(plane))) # [ 0.76190476  0.38095238  0.19047619]
print('P2 projection: ' + str(P2.projection(plane))) # [ 0.57142857  0.28571429  1.14285714]

# get line through P1, perpendicular to the plane:
l = P1.perpendicular(plane)


# 4D example
p1, p2, p3 = Point(1, 0, 0, 0), Point(0, 2, 0, 1), Point(0, 0, 4, 5)
plane = Plane(p1, p2, p3)

P1, P2 = Point(0, 0, 0, 0), Point(0, 0, 1, 0)

print('P1 projection: ' + str(P1.projection(plane))) # [ 0.83333333  0.33333333  0.          0.16666667]
print('P2 projection: ' + str(P2.projection(plane))) # [ 0.83333333  0.11111111  0.44444444  0.61111111]

# get line through P1, perpendicular to the plane:
l = P1.perpendicular(plane)


## Let's form a line that includes a P3 and is perpendicular to the Line:
#line = P3.perpendicular(plane)
## currently to form the perpendicular line
## point can belong to the original line 
## if and only if 2D space is involved
#
#""" example of points projection with parameters in 5D space """
#from FuncDesigner import *
#a, b, c = oovars('a', 'b', 'c')
#p1 = Point(0, 2, 3, 4, a)
#p2 = Point(4, 0, 3, b, c)
#line = Line(p1, p2)
#P = Point(1, 2+b*c, 3*a, 4+exp(b), a*sqrt(c))
#proj = P.projection(line)
#Parameters = {a:1, b:2, c:0.5}
#print('for parameters %s projection is %s' % (Parameters,  proj(Parameters)))
## for parameters {a: 1, b: 2, c: 0.5} projection is [-2.08357371  3.04178685  3.  5.04178685   1.26044671]
#
