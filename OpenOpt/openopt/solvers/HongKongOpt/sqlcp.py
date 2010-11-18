'''
Copyright (c) 2010 Enzo Michelangeli and IT Vision Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''
import sys
from numpy import *
from qlcp import qlcp
from nearcorr import nearcorr

import numpy as np  # required by numdifftools (go figure)

def _simple_grad(f, x, delta = 1e-6):
    nvars = x.shape[0]
    I = eye(nvars)
    Z = zeros(nvars)
    grad = array([(f(x+I[i,:]*delta) - f(x-I[i,:]*delta))/delta/2. for i in xrange(nvars)])
    return grad
    
''''''
def _simple_hessian(f, x, delta = 1e-6):
    g = lambda x: _simple_grad(f, x, delta = delta) # g(x) is the gradient of f
    return _simple_grad(g, x, delta=delta)
''''''
def _simple_hessdiag(f, x, delta = 1e-6):
    nvars = x.shape[0]
    I = eye(nvars)
    Z = zeros(nvars)
    hd = array([(f(x+I[i,:]*delta) + f(x-I[i,:]*delta) - 2*f(x))/delta**2 for i in xrange(nvars)])
    return diag(hd)
    
def sqlcp(f, x0, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, minstep=1e-15, minrfchange=1e-15):
    '''
    SQP solver. Approximates f in x0 with paraboloid with same gradient an hessian,
    then finds its minimum with the quadratic solver qlcp and uses it as new point, 
    iterating till norm of change in x drops below minstep. 
    Requires the Hessian to be definite positive.
    The Hessian is initially approximated by its principal diagonal, and then
    updated at every step with the BFGS method.
    '''
    nvars = x0.shape[0]
    x = x0.copy()
    niter = 0
    deltah = 1e-4
    deltag = deltah**2
    twoI = 2.*eye(nvars)
    oldfx = f(x)
    gradfx = _simple_grad(f, x, deltag)  # return the gradient of f() at x
    #hessfx = _simple_hessian(f,x,delta=deltah)
    hessfx = _simple_hessdiag(f,x,delta=deltah) # good enough, and much faster, but only works if REAL Hessian is DP!
    invhessfx = linalg.inv(hessfx)
    while True:
        niter += 1
        #print "f(",x,"):",f(x)
        bb = b if b == None else b-dot(A,x)
        bbeq = beq if beq == None else beq-dot(Aeq,x)
        lbb = lb if lb == None else lb - x
        ubb = ub if ub == None else ub - x
        deltax = qlcp(gradfx, hessfx, QI=invhessfx, A=A, b=bb, Aeq=Aeq, beq=bbeq, lb=lbb, ub=ubb)
        ###
        if deltax == None:
            '''
            print "No solution from LCP solver: trying diag Hessian"
            hessfx = diag(diag(hessfx)) # set off-diagonal entries to zero
            deltax = qlcp(gradfx, hessfx, A=A, b=bb, Aeq=Aeq, beq=bbeq, lb=lbb, ub=ubb)
            if deltax == None:
            '''
            print "Cannot converge, sorry."
            x = None
            break
        
        x += deltax
        if linalg.norm(deltax) < minstep:
            break
        fx = f(x)
        if abs(fx-oldfx) < minrfchange*abs(fx):
            break
        oldfx = fx
        oldgradfx = gradfx.copy()
        gradfx = _simple_grad(f, x, deltag)  # return the gradient of f() at the new x
        # we might also put a termination test on the norm of grad...
        
        '''
        # recalc hessian afresh would be sloooow...
        hessfx = _simple_hessian(f,x,delta=deltah)  # return the hessian of f() at x
        invhessfx = linalg.inv(hessfx)
        '''
        # update Hessian and its inverse with BFGS based on current Hessian, deltax and deltagrad    
        # See http://en.wikipedia.org/wiki/BFGS
        deltagrad = gradfx - oldgradfx
        hdx = dot(hessfx, deltax)
        dgdx = dot(deltagrad,deltax)
        #if dgdx < 0.:
        #    print "deltagrad * deltax < 0!" # a bad sign
        hessfx += ( outer(deltagrad,deltagrad) / dgdx - 
                    outer(hdx, hdx) / dot(deltax, hdx) )
        # now update inverse of Hessian  
        '''
        invhessfx = linalg.inv(hessfx)
        '''
        hidg = dot(invhessfx,deltagrad)
        oIdgdeltax = outer(hidg,deltax)
        invhessfx += ( (dgdx+dot(deltagrad,hidg))*outer(deltax,deltax)/(dgdx**2) -
                (oIdgdeltax+oIdgdeltax.T)/dgdx ) # just because invhessfx is symmetric, or else:
                #(oIdgdeltax+outer(deltax,dot(invhessfx.T,deltagrad)))/dgdx )
    return x, niter
    
if __name__ == "__main__":

    set_printoptions(suppress=True) # no annoying auto-exp print format

    # upside-down multi-gaussian with circular contour, centered in [1, 1, 1...]
    f = lambda x: -exp(-dot(x-1.,x-1.)) # min in (1., 1., ....) == -1.     

    nvars = 2
    #x0 = array([0.5]*nvars)
    x0 = array([1.5, 0.6])
    
    x, niter = sqlcp(f, x0, lb=array([-2.]*nvars), ub=array([2.]*nvars) )
    
    print "after",niter,"iterations:" 
    if x != None:
        print "x:", x, ", f(x):", f(x)
    else:
        print "sqlcp() did not converge (encountered Hessian <= 0.)"
    