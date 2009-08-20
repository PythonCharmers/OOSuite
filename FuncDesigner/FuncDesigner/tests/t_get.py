from FuncDesigner import *
N = 4
x = oovar()

h1 = 1e1*(x[N-1]-1)**4 - 0
point = {x:[2, 2, 4, 5]}

print h1.D(point)
