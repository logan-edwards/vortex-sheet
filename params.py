'''
params.py

This file contains the simulation parameters.

Dependencies: none
'''

Nb = 100
delta = 0.2

T = 5
dt = 0.01
Nt = int(T/dt) + 1

eps_shed = 1e-12

u = 1.0

'''
    Misc parameters
'''
enable_animation = True