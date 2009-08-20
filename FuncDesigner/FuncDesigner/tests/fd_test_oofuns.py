from FuncDesigner import *
a = oovar('a')
b = oovar('b')
c = oovar('c')

point1 = {a: 1, b: 2, c: [3, 4, 5]}

# TODO: 
# a,b,c = oovars('a', 'b', 'c')

d = oofun(lambda y: y+1.0, input = a, name = 'd')
print 'd1:', d.D(point1)

f = oofun(lambda y: y-10, input = b)


func = oofun(lambda y, z: y+z, input=[a, b])
#func = oofun(lambda y, z: y+z, input=[d, f])


print func(point1)

print func.D(point1)

# scalar oofun:
func1 = oofun(lambda y, z: y**2+z**2 + y*z, d=(lambda y, z: 2*y+z, lambda y, z:2*z + y), input=[d, f])
func1 = oofun(lambda y, z: y**2+z**2 + y*z, d=(lambda y, z: 2*y+z, lambda y, z:2*z + y), input=[d, b], name = 'func1')
#func1 = oofun(lambda y, z: y**2+z**2 + y*z, d=(None, lambda y, z:2*z + y), input=[a, b])
#func1 = oofun(lambda y, z: y**2+z**2 + y*z, d=lambda y, z: [2*y+3*z, 2*z + y], input=[a, b])
#func1 = oofun(lambda y, z: [y**2+sum(z)**2+z[2] + 4*y, z[0], 20*z[1], 10*y, 100*y+z[0]], input=[d, c]) # R^(a.size+c.size) -> R^5
print func1.D(point1)




func1.check_d1(point1)

# vector oofun:
#func2 = oofun(lambda y, z: [y**2+z.sum()**2, y**2+z[1]**2,  y**2+z[2]**2], input=[a, c])
#func2.d = (lambda y, z: 2*y, #d_func_0/d_y
#           lambda y, z: 2*z, #d_func_0/d_z
#           )
#print func2(point1)
#print func2.D(point1)


print 'done'
