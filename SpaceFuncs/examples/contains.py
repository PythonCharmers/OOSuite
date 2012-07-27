from SpaceFuncs import *


p0, p1, p2, p3 = Point(0, 0, 0), Point(1, 0, 0), Point(0, 2, 0), Point(0, 0, 4)
line = Line(p0, p1)

print(line.contains(p0)) # True
print((p0 + 1e-7*p2) in line) # True, with default tolerance 10^-6
print(line.contains(p0 + 1e-100*p2, tol = 1e-10)) # True; tol is max allowed distance from point to line
print(line.contains(p0 + 1e-100*p2, tol = 1e-200)) # False

plane = Plane(p1, p2, p3)

print(plane.contains(p0)) # False
print(p0.projection(plane) in plane) # True
print((p0.projection(plane) + 1e-100*Point(1, 1, 1)) in plane) # True, with default tolerance 10^-6
print(plane.contains(p1 + 1e-100*Point(1, 1, 1), tol = 1e-10)) # True
print(plane.contains(p1 + 1e-10*Point(1, 1, 1), tol = 1e-200)) # False

sphere = Sphere(p0, radius = 1)
print(sphere.contains(p0)) # False
print(sphere.contains(p1)) # True
print((p1 + 1e-7 * Point(1, 1, 1)) in sphere) # True, with default tolerance 10^-6
print(sphere.contains(p1 + 1e-100*Point(1, 1, 1), tol = 1e-10)) # True
print(sphere.contains(p1 + 1e-10*Point(1, 1, 1), tol = 1e-200)) # False
