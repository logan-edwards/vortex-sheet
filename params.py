'''
params.py

This file contains global values used by the simulation. Data includes values
for the body motion parameterization, as well as other constants referenced
during simulation runtime.

All values must be assigned, or unexpected behavior may occur.

Dependencies: numpy (if np.pi is used to compute values in radians)
'''

import numpy as np

#               --- SIMULATION PARAMETERS ---
# Nb:           Number of points to use when discretizing the body; machine
#               precision is safely reached with a couple hundred points due to
#               spectral accuracy of the integration method utilized.
#
# eps_shed:     Distance away from the trailing edge to append new points to the
#               free sheet. A non-zero distance is necessary to avoid
#               division by 0, but this distance should be kept small to better
#               approximate the continuity of the combined vortex sheet.
#
# delta:        Smoothing parameter for the smoothed Cauchy kernel. In general,
#               values in the range 0.05 <= delta <= 0.5 have been used in the
#               literature for various applications; smaller delta better
#               approximate the true behavior but are prone to instability. A
#               value of delta = 0.2 is standard and recommended in this case.
#
# T:            Total simulation runtime.
#
# dt:           Simulation timestep.
#
# Nt:           Number of timesteps in the simulation. The vortex sheet method
#               implemented here experiences quadratic slowdown with increased
#               Nt; keep this in mind as T and dt are balanced.
#               !!! NT IS A COMPUTED QUANTITY AND SHOULD NOT BE SET MANUALLY !!!

Nb = 100
eps_shed = 1e-12
delta = 0.2

dt = 0.001
T = 5

Nt = int(T/dt)

#               --- BODY MOTION PARAMETERS ---
# h0:           Heaving amplitude, in length units.
#
# phi_h         Phase offset of heaving, in radians.
#
# theta0:       Pitching amplitude, in radians.
#
# phi_theta:    Phase offset of pitching, in radians.
#
# L0:           Magnitude of chordwise body lengthening, where L0 is the
#               difference between max and min plate lengths. The limiting cases
#               are L0 = 0, where the arclength of the plate does not vary, and
#               L0 = 2, where the plate shrinks to a point at the location of
#               the pivot.
#               !!! DUE TO NUMERICAL INSTABILITY, L0 SHOULD BE KEPT AS SMALL AS 
#               FEASIBLE, IDEALLY BELOW 0.5; LARGER VALUES MAY CAUSE NONPHYSICAL
#               ENTANGLEMENT BETWEEN THE FREE SHEET AND THE BODY !!!
#
# phi_L:        Phase offset of chordwise lengthening, in radians.
#
# alpha0:       Location of the pivot in material coordinates; alpha0 = 1.0
#               corresponds to the leading edge, while alpha0 = -1.0 corresponds
#               to the trailing edge.
#
# U:            Horizontal "free-stream" speed, in units length/time.
#
# f:            Frequency of the body motion, in units 1/time
#
# freq_mult:    Multiplier on the lengthening frequency, i.e. the frequency of
#               lengthening is f_L = freq_mult * f; sensible values are either
#               freq_mult = 1 (lengthening once per period) or freq_mult = 2
#               (lengthening on both the upstroke and downstroke).

h0 = 0.1
phi_h = 0.0

theta0 = np.pi / 9.0
phi_theta = 270.0 * np.pi / 360.0

L0 = 0.0
phi_L = 60.0 * np.pi / 180.0

alpha0 = 1.0
U = 1.0
f = 0.2
freq_mult = 1