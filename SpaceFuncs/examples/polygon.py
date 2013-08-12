from FuncDesigner import *
from SpaceFuncs import *
spaceDimension = 2
nVertices = 15
parametrizedVerticesCoords = oovars(nVertices, size=spaceDimension)# Python list of 15 oovars, each one of size spaceDimension
P = Polygon(parametrizedVerticesCoords)

parameterValues = dict([(parametrizedVerticesCoords[i], [sin(2 * pi * i / nVertices), cos(2 * pi  * i  / nVertices)]) for i in range(nVertices)])
#print(P.sides(parameterValues))
print(P.P(parameterValues))# perimeter: 6.23735072453
print(P.S(parameterValues))# area: 3.05052482307
areaPlusPerimeter = P.S + P.P

print('for these params area + perimeter equals to ' + str(areaPlusPerimeter(parameterValues)))# 9.2878755476 for nVertices = 15

Plot(P(parameterValues))
