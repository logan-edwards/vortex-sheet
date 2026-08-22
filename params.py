'''
params.py

This file contains the simulation parameters.

Dependencies: none
'''

#                           --- BODY PARAMETERS ---
# Nb:               Number of points discretizing the body. Machine precision is 
#                   already reached with ~100 points due to the spectral 
#                   accuracy of the solver.
#
# eps_shed:         Distance away from the body to place new points on the free
#                   sheet. This value should be kept small to best approximate
#                   the continuity of the combined vortex sheet, but it must be
#                   nonzero to avoid singularities in numerical integrals.
#
# delta:            Smoothing parameter. Small values of delta are more prone to
#                   numerical instability, although small delta more accurately
#                   represent the free sheet dynamics. In the literature, values
#                   in the range 0.05 <= delta <= 0.5 are acceptable in
#                   different contexts; a value of delta = 0.2 is generally a
#                   good choice for the problems studied here.
#
#                           --- SIM PARAMETERS ---
# T:                Total simulation time.
#
# dt:               Time resolution for the simulation, i.e. the timestep
#                   between each step in the simulation.
#
# Nt:               Number of timesteps to run the simulation for. This quantity
#                   is computed from T and dt and should NOT be set manually.
#
#                           --- SIM FLAGS ---
# enable_animation: Produces an animation file if set to True. This increases
#                   the memory cost slightly by storing the free sheet points at
#                   each timestep.

Nb = 100
eps_shed = 1e-12
delta = 0.2

T = 5
dt = 0.01
Nt = int(T/dt) + 1

u = 1.0

enable_animation = True
