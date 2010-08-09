from FuncDesigner import *
from scipy import fft
from numpy import arange
a, b, c = oovars('a', 'b', 'c')
FFT  = oofun(lambda x, y, z: (fft((x+2*y+4*z)*arange(16))).real, input=[a, b, c])
f = a**2+b**2+c**2 + FFT
point = {a:1, b:2, c:3} 
print f(point)
print f.D(point)
 
#  Another way, via creating oofun constructor:
 
from FuncDesigner import *
from scipy import fft
from numpy import arange, hstack
a, b, c = oovars('a', 'b', 'c')
myFFT = lambda INPUT: oofun(lambda *args:fft(hstack(args)).real, input=INPUT)
f = a**2+b**2+c**2 + myFFT([a+2*b, a+3*c, 2*b+4*c])
point = {a:1, b:2, c:3} 
print f(point)
print f.D(point)
