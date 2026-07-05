'''
    Simulation Parameters
'''

import numpy as np

Nb = 100
delta = 0.2

T = 5
dt = 0.01
Nt = int(T/dt) + 1

eps_shed = 1e-12

u = 1.0

pitching_amplitude = np.pi/9.0
pitching_phase = 0.0

heaving_amplitude = 0.5
heaving_phase = 0.0

lengthening_amplitude = 1.5
lengthening_phase = np.pi/4.0