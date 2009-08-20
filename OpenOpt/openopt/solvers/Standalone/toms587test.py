#from toms587 import lsei
#from numpy import *
#
#W = array([8.94558004,  7.3286058, 3.0,  2.05011463,  0.43314039, 70.31767878, -12.29662251, -13.91359675,  75.74379415])
#me,ma,mg,n,prgopt,xf,rnorme,rnorml,mode,ws,ip = 0, 3, 0, 2, ravel(1.), ravel(0.), -15.0, -15.0, -15, ravel(-15.), ravel((-15, -15, -15))
#
#lsei(W,me,ma,mg,n,prgopt,xf,rnorme,rnorml,mode,ws,ip)

#StdErr: *** glibc detected *** /usr/bin/python: free(): invalid next size (fast): 0x0000000000cfb390 ***

from toms587 import lsei
from numpy import *

W = array([8.94558004,  7.3286058, 3.0,  2.05011463,  0.43314039, 70.31767878, -12.29662251, -13.91359675,  75.74379415])
me,ma,mg,n,prgopt,xf,rnorme,rnorml,mode,ws,ip = 0, 3, 0, 2, ravel(1.), ravel(0.), -15.0, -15.0, -15, ravel(-15.), array((-15, -15, -15))

lsei(W,me,ma,mg,n,prgopt,xf,rnorme,rnorml,mode,ws,ip)
