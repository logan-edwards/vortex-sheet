'''
classes.py

This file contains the class definitions for the free and bound sheets.

Dependencies: numpy
'''
import params

import numpy as np

class BoundSheet:
    def __init__(self, 
                 heaving_amp, 
                 heaving_phase, 
                 pitching_amp, 
                 pitching_phase, 
                 length_amp, 
                 length_phase, 
                 pivot_loc, 
                 frequency
                 ):
        self.alpha_chebyshev = np.zeros(params.Nb+1)
        self.alpha_collocation = np.zeros(params.Nb)
        for k in range(params.Nb+1):
            self.alpha_chebyshev[k] = -np.cos(k*np.pi/params.Nb)
        for l in range(params.Nb):
            self.alpha_collocation[l] = -np.cos((2*l+1)*np.pi / (2*params.Nb))

        self.sigmas = np.zeros((params.Nb+1, params.Nt))
        self.pressures = np.zeros((params.Nb+1, params.Nt))
        self.dsigmadt = np.zeros((params.Nb+1, params.Nt))
        self.force_x = np.zeros(params.Nt)
        self.force_y = np.zeros(params.Nt)
        self.power_in = np.zeros(params.Nt)

        self.x_chebyshev = np.zeros((params.Nb+1, params.Nt))
        self.y_chebyshev = np.zeros((params.Nb+1, params.Nt))
        self.dxdt_chebyshev = np.zeros((params.Nb+1, params.Nt))
        self.dydt_chebyshev = np.zeros((params.Nb+1, params.Nt))
        self.dsdalpha = np.zeros(params.Nt)

        self.x_collocation = np.zeros((params.Nb, params.Nt))
        self.y_collocation = np.zeros((params.Nb, params.Nt))
        self.dxdt_collocation = np.zeros((params.Nb, params.Nt))
        self.dydt_collocation = np.zeros((params.Nb, params.Nt))

        self.normal_x = np.zeros(params.Nt)
        self.normal_y = np.zeros(params.Nt)
        self.tangent_x = np.zeros(params.Nt)
        self.tangent_y = np.zeros(params.Nt)

        self.L = np.zeros(params.Nt)
        c = np.zeros(params.Nt)
        theta = np.zeros(params.Nt)

        self.dLdt = np.zeros(params.Nt)
        dcdt = np.zeros(params.Nt)
        dthetadt = np.zeros(params.Nt)

        for t in range(params.Nt):
            freq_multiplier = 2
            self.L[t] = 1 + 0.5 * length_amp * np.sin(freq_multiplier * 2 * np.pi * frequency * t * params.dt + length_phase)
            c[t] = heaving_amp * np.sin(2 * np.pi * frequency * t * params.dt + heaving_phase)
            theta[t] = pitching_amp * np.sin(2 * np.pi * frequency * t * params.dt + pitching_phase)

            self.dLdt[t] = 0.5 * length_amp * freq_multiplier * 2 * np.pi * frequency * np.cos(freq_multiplier * 2 * np.pi * frequency * params.dt * t + length_phase)
            dcdt[t] = heaving_amp * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t * params.dt + heaving_phase)
            dthetadt[t] = pitching_amp * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t * params.dt + pitching_phase)

        for t in range(params.Nt):
            for k in range(params.Nb+1):
                self.x_chebyshev[k, t] = self.L[t] * (self.alpha_chebyshev[k] - pivot_loc) * np.cos(theta[t]) + params.dt * t * params.u
                self.y_chebyshev[k, t] = c[t] + self.L[t] * (self.alpha_chebyshev[k] - pivot_loc) * np.sin(theta[t])
                self.dxdt_chebyshev[k, t] = self.dLdt[t] * (self.alpha_chebyshev[k] - pivot_loc) * np.cos(theta[t]) + self.L[t] * (self.alpha_chebyshev[k] - pivot_loc) * -np.sin(theta[t]) * dthetadt[t] + params.u
                self.dydt_chebyshev[k, t] = dcdt[t] + self.dLdt[t] * (self.alpha_chebyshev[k] - pivot_loc) * np.sin(theta[t]) + self.L[t] * (self.alpha_chebyshev[k] - pivot_loc) *np.cos(theta[t]) * dthetadt[t]
            for l in range(params.Nb):
                self.x_collocation[l, t] = self.L[t] * (self.alpha_collocation[l] - pivot_loc) * np.cos(theta[t]) + params.dt * t * params.u
                self.y_collocation[l, t] = c[t] + self.L[t] * (self.alpha_collocation[l] - pivot_loc) * np.sin(theta[t])
                self.dxdt_collocation[l, t] = self.dLdt[t] * (self.alpha_collocation[l] - pivot_loc) * np.cos(theta[t]) + self.L[t] * (self.alpha_collocation[l] - pivot_loc) * -np.sin(theta[t]) * dthetadt[t] + params.u
                self.dydt_collocation[l, t] = dcdt[t] + self.dLdt[t] * (self.alpha_collocation[l] - pivot_loc) * np.sin(theta[t]) + self.L[t] * (self.alpha_collocation[l] - pivot_loc) *np.cos(theta[t]) * dthetadt[t]

        self.normal_x = -np.sin(theta)
        self.normal_y = np.cos(theta)
        self.tangent_x = np.cos(theta)
        self.tangent_y = np.sin(theta)

class FreeSheet:
    def __init__(self):
        # change this so that we append, as we did in the previous code.
        self.circulation = np.zeros(0)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.dxdt = np.zeros(0)
        self.dydt = np.zeros(0)
        self.dgammadt = 0
    def append_circulation(self,
                           circulation_new,
                           x_new,
                           y_new,
                           ):
        self.circulation = np.append(self.circulation, circulation_new)
        self.x = np.append(self.x, x_new)
        self.y = np.append(self.y, y_new)
        self.dxdt = np.append(self.dxdt, 0)
        self.dydt = np.append(self.dydt, 0)