from FuncDesigner import *
a, b, c = oovars('a', 'b', 'c') 
f1,  f2 = sin(a)+cosh(b), 2*b+3*c.sum()
f3 = 2*a*b*prod(c) + f1*cos(f2)

point1 = {a:1, b:2, c:[3, 4, 5]}
point2 = {a: -10.4, b: 2.5,  c:[3.2, 4.8]}

# usage:
# myOOFun = integrate.scipy_quad(integration_oofun, lower_bound,  upper_bound, integration_oovar,  /{optionally: parameters for  scipy.integrate.quad}/) 

f4 = integrate.scipy_quad(f1+2*f2+3*f3, -1, 1, b) 
print f4(point1), f4(point2) # Expected output: 147.383792876, 102.143425528

# integral bounds can be oofuns as well:
f5 = integrate.scipy_quad(f1+2*f2+3*f3, 10.4, f2+5*sin(f1), a) 
f6 = integrate.scipy_quad(f1+2*f2+3*f3, cos(f2+2*f3), -9.3, a) 
f7 = integrate.scipy_quad(f1+2*f2+3*sqrt(abs(f3)), cos(f2+2*f3), f2+5*sin(f1), a) 
print f5(point1), f5(point2) # Expected output: 404683.969794 107576.397664
print f6(point1), f6(point2) # Expected output: 30394.156119 9498.22201774
print f7(point1), f7(point2) # Expected output: 9336.70442146 5259.53130904

#r_d = f4.D(point) # scipy_quad derivatives are not implemented yet
