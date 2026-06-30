'''
    Class Definitions
'''
import params

import numpy as np

class BoundSheet:
    def __init__(self, assign_x, assign_y, assign_dxdt, assign_dydt):
        self.alpha_chebyshev = np.zeros(params.Nb+1)
        self.alpha_collocation = np.zeros(params.Nb)
        for k in range(params.Nb+1):
            self.alpha_chebyshev[k] = -np.cos(k*np.pi/params.Nb)
        for l in range(self.Nb):
            self.alpha_collocation[l] = -np.cos((2*l+1)*np.pi / (2*params.Nb))
        
        self.x = np.zeros((params.Nb+1, params.Nt))
        self.y = np.zeros((params.Nb+1, params.Nt))
        self.dxdt = np.zeros((params.Nb+1, params.Nt))
        self.dydt = np.zeros((params.Nb+1, params.Nt))

