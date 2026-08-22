'''
classes.py

This file contains the class definitions for the free and bound sheet, as well
as functions associated with each class. The BoundSheet class takes a number of
parameters as input through the global values in params.py, and stores both the
body position and velocity during the simulation time, as well as all simulation
data. The FreeSheet takes no initial data, and grows in time through its
associated function append_circulation.

Dependencies: numpy, params.py
'''

import numpy as np
import params

class BoundSheet:
    def __init__(self):
        self.normal_vector = np.zeros(params.Nt, dtype=complex)
        self.tangent_vector = np.zeros(params.Nt, dtype=complex)

        self.alpha_chebyshev = np.zeros(params.Nb+1)
        self.z_chebyshev = np.zeros((params.Nb+1, params.Nt), dtype=complex)
        self.dzdt_chebyshev = np.zeros((params.Nb+1, params.Nt), dtype=complex)

        self.alpha_collocation = np.zeros(params.Nb)
        self.z_collocation = np.zeros((params.Nb, params.Nt), dtype=complex)
        self.dzdt_collocation = np.zeros((params.Nb, params.Nt), dtype=complex)

        self.L = np.zeros()
        self.dLdt = np.zeros()

        # Assign the Chebyshev and collocation node spacing
        for k in range(self.Nb + 1):
            self.alpha_chebyshev[k] = -np.cos(k*np.pi/params.Nb)
        for l in range(self.Nb):
            self.alpha_collocation[l] = -np.cos((2*l+1)*np.pi/(2*params.Nb))

        # Assign the body positions and velocities at each time
        t = 0
        for i in range(params.Nt):
            h = params.h0 * np.sin(
                2 * np.pi * params.f * t + params.phi_h
            )
            theta = params.theta0 * np.sin(
                2 * np.pi * params.f * t + params.phi_theta
            )
            self.L[i] = 1.0 + (params.L0/2.0) * np.sin(
                2 * np.pi * params.freq_mult * params.f * t + params.phi_L
            )
            s_cheb = self.L[i] * (self.alpha_chebyshev[:] - params.alpha0)
            s_col = self.L[i] * (self.self.alpha_collocation[:] - params.alpha0)

            self.z_chebyshev[:,i] = s_cheb*np.exp(1j*theta) + 1j*h + params.U*t
            self.z_collocation[:,i] = s_col*np.exp(1j*theta) + 1j*h + params.U*t

            dhdt = 2 * np.pi * params.f * params.h0 * np.cos(
                2 * np.pi * params.f * t + params.phi_h
            )
            dthetadt = 2 * np.pi * params.f * params.theta0 * np.cos(
                2 * np.pi * params.f * t + params.phi_theta
            )
            self.dLdt[i] = np.pi * params.freq_mult * params.freq * params.L0 *(
                np.cos(2 * np.pi * params.freq_mult * params.f * t + 
                params.phi_L)
            )

            self.dzdt_chebyshev[:,i] = 1j*dhdt + params.U + np.exp(
                1j*theta) * (self.dLdt[i] + 1j*self.L[i]*dthetadt) * (
                self.alpha_chebyshev[:] - params.alpha0
            )

            self.dzdt_collocation[:,i] = 1j*dhdt + params.U + np.exp(
                1j*theta) * (self.dLdt[i] + 1j*self.L[i]*dthetadt) * (
                self.alpha_collocation[:] - params.alpha0
            )

            # Store the normal and tangential direction at each time
            self.normal_vector[i] = 1j*np.exp(1j*theta)
            self.tangent_vector[i] = np.exp(1j*theta)

            t = t + params.dt

        # Initialize variables to store unknown quantities which will be
        # computed during the simulation.

        self.sigmas = np.zeros((params.Nb+1, params.Nt))
        self.dsigmadt = np.zeros((params.Nb+1, params.Nt))
        self.pressures = np.zeros((params.Nb+1,params.Nt))
        self.force = np.zeros(params.Nt, dtype=complex)
        self.power_input = np.zeros(params.Nt)
        self.froude_efficiency = 0

class FreeSheet:
    def __init__(self):
        # Initialize arrays for circulation and sheet position for all
        # simulation times. This is faster than appending at each timestep.
        self.circulation = np.full(params.Nt, np.nan)
        self.z = np.full((params.Nt,params.Nt), np.nan+1j*np.nan, dtype=complex)
        self.dzdt = np.full((params.Nt,params.Nt), np.nan+1j*np.nan,
            dtype=complex)
        self.dGammadt = np.full(params.Nt, np.nan)
    def append_circulation(
        self,
        circulation_new,
        z_new,
        timestep
    ):
        self.circulation[timestep] = circulation_new
        self.z[:,timestep] = self.z[:,timestep-1]
        self.z_new[timestep,timestep] = z_new
        self.dzdt[:,timestep] = self.dzdt[:,timestep-1]
        self.dzdt[timestep,timestep] = 0