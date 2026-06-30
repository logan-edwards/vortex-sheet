'''
    Simulation Parameters
'''

import numpy as np

Nb = 100
delta = 0.2

T = 5
dt = 0.01
Nt = int(T/dt)

theta = np.zeros((Nb, Nt))
dthetadt = np.zeros((Nb, Nt))

L = np.zeros((Nb, Nt))
dLdt = np.zeros((Nb, Nt))

c_x = np.zeros((Nb, Nt))
c_y = np.zeros((Nb, Nt))
dcdt_x = np.zeros((Nb, Nt))
dcdt_y = np.zeros((Nb, Nt))