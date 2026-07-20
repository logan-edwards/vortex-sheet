'''
    Simulation Parameters
'''

import numpy as np

Nb = 100
delta = 0.2

T = 5
dt = 0.05
Nt = int(T/dt) + 1

eps_shed = 1e-12

u = 1.0

'''
    Misc parameters
'''
enable_animation = True
scipy_iteration = 0
sim_start_time = 0