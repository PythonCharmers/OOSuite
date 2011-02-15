from SpaceFuncs import *
A, B, C = Point(0, 8)('A'), Point(3, 7)('B'), Point(3.5, 5)('C')
T = Triangle(A, B, C)

Plot(T, 
     T.InscribedCircle, 
     T.CircumCircle, 
     LineSegment(B, T.incenter('I')), 
     LineSegment(C, T.incenter), # you can use both T.incenter and T.InscribedCircle.center,
     LineSegment(A, T.InscribedCircle.center), # T.circumCircleCenter and T.CircumCircle.center
     LineSegment(T.circumCircleCenter('O'), T.incenter), 
     )

