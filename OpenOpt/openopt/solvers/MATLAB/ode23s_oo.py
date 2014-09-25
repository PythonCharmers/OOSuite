from __future__ import absolute_import
from .matlab_ode import matlab_ode

class ode23s(matlab_ode):
    __name__ = 'ode23s'
    __alg__ = ""
    solver_id = 104

    def __init__(self): 
        pass

