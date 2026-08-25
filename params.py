'''
params.py

This file contains the simulation parameters.

Dependencies: numpy
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
# u:                Characteristic velocity of the body, i.e. the horizontal
#                   translation rate in absence of other motions. The Froude
#                   efficiency is nondimensionalized in terms of this parameter.
#
# h0:               Heaving amplitude.
#
# phi_h:            Heaving phase.
#
# theta0:           Pitching amplitude (radians).
#
# phi_theta:        Pitching phase.
#
# L0:               Lengthening amplitude, difference between maximum and
#                   minimum arclength. Due to numerical instability, L0 should
#                   be in the range 0 < L0 < 0.6, with L0 = 0 corresponding to
#                   no lengthening, and L0 = 2 corresponding to shrinking to a
#                   single point.
#
# phi_L:            Lengthening phase.
#
# alpha0:           Material coordinate of pitching, heaving, and lengthening.
#                   Should be in the range -1 < alpha0 < 1.
#
# freq_multiplier:  Should be either 1 or 2. Determines the frequency of
#                   lengthening; freq_multiplier = 1 corresponds to lengthening
#                   once on either the upstroke or downstroke, while
#                   freq_multiplier = 2 corresponds to lengthening once on each.
#
#                           --- POST-PROCESSING ---
# enable_animation: Produces an animation file if set to True. This increases
#                   the memory cost slightly by storing the free sheet points at
#                   each timestep.
#
# animation_name:   Name for the animation file.
#
# animation_format: File format for animation; also the file extension.

import numpy as np

Nb = 100
eps_shed = 1e-12
delta = 0.2

T = 5
dt = 0.003
Nt = int(T/dt) + 1

u = 1.0

h0 = 0.1
phi_h = 0
theta0 = np.pi / 9.0
phi_theta = 270.0 * np.pi / 360.0
L0 = 0.3
phi_L = 60.0 * np.pi / 180.0

alpha0 = 1.0

freq_multiplier = 1

enable_animation = True
animation_name = 'animation'
animation_format = 'mp4'