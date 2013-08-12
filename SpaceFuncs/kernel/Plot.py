# created by DmitreyPlot.py
from .misc import SpaceFuncsException
pylabInstalled = False
try:
    import pylab
    pylabInstalled = True
except ImportError:
    pass
    
def Plot(*args, **kw):
    if not pylabInstalled: 
        raise SpaceFuncsException('to plot you should have matplotlib installed')
    for arg in args: arg.plot(**kw)
    pylab.axis('equal')
    pylab.draw()
    if kw.get('grid') not in ('off', False, 0):
        pylab.grid('on')
    pylab.show()
