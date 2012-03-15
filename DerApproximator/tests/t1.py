from DerApproximator import *
import numpy as np
np.seterr(invalid='ignore')

print(get_d1(lambda x: (x**2).sum(), [1,2,3]))
print(get_d1(lambda x: x**2, [1,2,3]))
print(get_d1(lambda x: ((-x)**0.5)**2, [0], stencil=1))
'''
[ 1.99999993  3.99999998  6.0000002 ]
[[ 1.99999998  0.          0.        ]
 [ 0.          4.00000006  0.        ]
 [ 0.          0.          6.0000002 ]]
[-1.]
'''
