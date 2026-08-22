'''
functions.py

This file contains the functions necessary for running the simulation.

Functions:

K_delta:                Delta-smoothed Cauchy kernel. Takes the denominator of
                        the Birkhoff-Rott equation and a value for delta. The
                        unsmoothed Cauchy kernel is achieved by setting delta=0.

compute_bound_sheet:    Solves the system of equations resulting from enforcing
                        the kinematic boundary condition, the Kutta condition at
                        the trailing edge, and Kelvin's circulation theorem.
                        Returns a vector b = [sigma0 ... sigma_{Nb} Gamma-]^T

compute_sheet_velocity: Computes the velocity of the free sheet. Returns a new
                        velocity vector.

update_sheet_position:  Updates the position of the free sheet points using AB2.
                        Returns a vector of new sheet positions.

compute_pressures:      Computes the pressure distribution along the bound
                        sheet. Returns a vector with pressures at the Chebyshev
                        nodes.

compute_forces:         Computes the net force on the body. Returns a complex
                        number F where thrust/drag is given by Re(F) and
                        lift/downforce is given by Im(F).

compute_power_input:    Computes the input power corresponding to the body
                        motion. Returns a single scalar value.

compute_time_average:   Computes the time-average given a set of values sampled
                        over a given time interval. Returns either a real or
                        scalar value.

compute_efficiency:     Computes the time-averaged Froude efficiency given an
                        average input power and thrust force.

Dependencies: numpy, numba
'''

import numpy as np
from numba import njit, prange

@njit
def K_delta(z, delta):
    denominator = np.abs(z)**2 + delta**2
    if(denominator < 1e-12):
        return(0)
    else:
        return(
            (1/(2j*np.pi)) * np.conj(z) / denominator
        )

@njit(parallel=True)
def compute_bound_sheet(
    Nb,
    Nf,
    normal_vector,
    z_chebyshev,
    z_collocation,
    dzdt_collocation,
    z_free_sheet,
    circulation,
    L,
    delta
    ):

    if(Nf - 1 < 0):
        return(np.zeros(Nb+2, dtype=np.complex128))

    # --- Constructing and solving equation of the form Ax = b ---
    A = np.zeros((Nb+2,Nb+2), dtype=np.complex128)
    b = np.zeros(Nb+2, dtype=np.complex128)

    # --- Construct the b vector ---
    for l in prange(Nb):
        bsum = 0
        for i in range(Nf - 1):
            bsum = bsum - 0.5 * (circulation[i+1] - circulation[i]) * (
                K_delta(z_collocation[l] - z_free_sheet[i+1], delta)
                + K_delta(z_collocation[l] - z_free_sheet[i], delta)
            )
        bsum = bsum + 0.5 * circulation[Nf-1] * (
            K_delta(z_collocation[l] - z_chebyshev[0], delta)
            + K_delta(z_collocation[l] - z_free_sheet[Nf-1])
        )
        bsum = bsum + np.conjugate(dzdt_collocation[l])
        b[l] = np.real(normal_vector * bsum)

    # --- Construct the A matrix ---

    # Kinematic Boundary Condition
    for l in prange(Nb):
        for k in range(Nb+1):
            # FILL OUT THE KBC
            if(k==0 or k==Nb):
                w = np.pi / (2*Nb)
            else:
                w = np.pi / Nb
            A[l,k] = np.real(L * normal_vector * w * K_delta(
                z_collocation[l] - z_chebyshev[k], 0
            ))
        A[l,Nb+1] = np.real(0.5 * L * normal_vector * (
            K_delta(z_collocation[l] - z_chebyshev[0], delta) +
            K_delta(z_collocation[l] - z_free_sheet[Nf - 1], delta)
        ))
    
    # Kutta Condition
    A[Nb,0] = 1.0

    # Kelvin's circulation theorem
    A[Nb+1, 0] = L * np.pi / (2*Nb)
    A[Nb+1, Nb] = L * np.pi / (2*Nb)
    for k in range(1,Nb):
        A[Nb+1, k] = L * np.pi / Nb
    A[Nb+1, Nb+1] = 1.0

    return(np.linalg.solve(A,b))

def compute_sheet_velocity(
    Nb,
    Nf,
    z_chebyshev,
    z_free_sheet,
    circulation,
    sigma,
    L,
    delta
    ):

    vel = np.zeros(Nf, dtype=np.complex128)

    # Integration over the free sheet
    for i in prange(Nf):
        for j in range(Nf-1):
            vel[i] = vel[i] + 0.5 * (circulation[j+1] - circulation[j]) * (
                K_delta(z_free_sheet[i] - z_free_sheet[j+1], delta) +
                K_delta(z_free_sheet[i] - z_free_sheet[j], delta)
            )
    # Integration over the bound sheet
    for i in prange(Nf):
        for k in range(Nb):
            if(k==0 or k==Nb):
                w = np.pi/(2*Nb)
            else:
                w = np.pi/Nb
            vel[i] = vel[i] + L * w * sigma[k] * K_delta(
                z_free_sheet[i] - z_chebyshev[k], 0
            )
    return(vel)

@njit(parallel=True)
def update_sheet_position(
    Nf,
    z_free_sheet,
    dzdt_free_sheet,
    dzdt_free_sheet_prev,
    dt
    ):

    # Compute new free sheet position with AB2 wherever we can;
    # Use Euler's method on latest point where we don't have previous data
    z_new = np.zeros(Nf, dtype=np.complex128)
    for i in prange(Nf-1):
        z_new[i] = z_free_sheet[i] + dt * (
            1.5 * dzdt_free_sheet[i] - 0.5 * dzdt_free_sheet_prev[i]
        )
    z_new[Nf-1] = z_new[Nf-1] + dt * dzdt_free_sheet[Nf-1]

    return(z_new)

@njit(parallel=True)
def compute_pressure_distribution(
    Nb,
    Nf,
    tangent_vector,
    z_chebyshev,
    z_collocation,
    dzdt_chebyshev,
    z_free_sheet,
    circulation,
    sigma,
    dsigmadt,
    dGammadt,
    L,
    dLdt
    ):

    # Allocate space for each quantity necessary to find pressure
    p = np.zeros(Nb+1)
    mu = np.zeros(Nb+1)
    tau = np.zeros(Nb+1)
    gamma = np.zeros(Nb+1)
    I_collocation = np.zeros(Nb, dtype=np.complex128)
    X = np.zeros(Nb+1, dtype=np.complex128)
    I_chebyshev = np.zeros(Nb, dtype=np.complex128)
    I_free = np.zeros(Nb+1, dtype=np.complex128)
    I_dsigmadt = np.zeros(Nb+1)
    I_sigma = np.zeros(Nb+1)

    # Reconstruct the integral at the Chebyshev nodes using collocation data
    # and a discrete cosine transform.

    # Computing the integral at the collocation nodes
    for l in prange(Nb):
        for k in range(Nb+1):
            if(k==0 or k==Nb):
                w = np.pi/(2*Nb)
            else:
                w = np.pi/Nb
            I = K_delta(z_collocation[l] - z_chebyshev[k], 0)
            I_collocation[l] = I_collocation[l] + w * sigma[k] * L * I

    # Performing the discrete cosine transform.
    for j in prange(Nb):
        for l in range(Nb):
            X = X + 2 * I_collocation[l] * np.cos(j*np.pi*(2*l+1)/(2*Nb))

    for k in prange(Nb+1):
        I_chebyshev[k] = I_chebyshev[k] + X[0]/(2*Nb)
        for j in range(1,Nb):
            I_chebyshev[k] = I_chebyshev[k] + (X[j]/Nb) * np.cos(j*k*np.pi/Nb)

    # Integration over the free sheet
    for k in prange(Nb+1):
        for j in range(Nf-1):
            I_free[k] = I_free[k] + 0.5 * (circulation[j+1] - circulation[j]) *(
                K_delta(z_chebyshev[k] - z_free_sheet[j+1], 0) +
                K_delta(z_chebyshev[k] - z_free_sheet[j], 0)
            )
    mu = np.real(tangent_vector * np.conjugate(I_chebyshev + I_free))
    tau = np.real(tangent_vector * np.conjugate(dzdt_chebyshev))

    for k in prange(1,Nb):
        gamma[k] = sigma[k] / np.sin(k*np.pi/Nb)
    
    for k in range(1,Nb+1):
        I_dsigmadt[k] = I_dsigmadt[k-1] + (np.pi/(2*Nb)) * (
            dsigmadt[k] + dsigmadt[k-1]
        )

    for k in range(1,Nb+1):
        I_sigma[k] = I_sigma[k-1] + (np.pi/(2*Nb)) * (sigma[k] + sigma[k-1])

    p = (mu - tau) * gamma + L * I_dsigmadt + dLdt * I_sigma + dGammadt
    return(p)

@njit(parallel=True)
def compute_forces(
    Nb,
    Nt,
    normal_vector,
    tangent_vector,
    pressure_matrix,
    leading_edge_sigma,
    L_vector
    ):

    F = np.zeros(Nt)
    for i in prange(Nt):
        for k in range(1,Nb-1):
            pressure_force = -normal_vector * L_vector[i] * (np.pi/Nb) * (
                pressure_matrix[k,i] * np.sin(np.pi*k/Nb)
            )
        suction_force = tangent_vector * (np.pi/8) * (leading_edge_sigma[i])**2
        F[i] = pressure_force + suction_force

    return(F)

@njit(parallel=True)
def compute_power_input(
    Nb,
    Nt,
    normal_vector,
    dzdt_chebyshev_matrix,
    pressure_matrix,
    L_vector
    ):

    P = np.zeros(Nt)
    for i in prange(Nt):
        for k in range(1,Nb-1):
            P[i] = P[i] + pressure_matrix[k,i] * np.real(
                normal_vector[i] * np.conjugate(dzdt_chebyshev_matrix[k,i])
            ) * L_vector[i] * (np.pi / Nb) * np.sin(k*np.pi/Nb)
    
    return P

@njit(parallel=True)
def compute_time_average(
    time,
    quantity
    ):
    Nt = np.size(time)
    time_avg = 0
    for i in range(Nt-1):
        dt = time[i+1] - time[i]
        time_avg = time_avg + 0.5 * (quantity[i+1] + quantity[i]) * dt

    time_avg = time_avg / (time[Nt-1] - time[0])

    return(time_avg)

@njit
def compute_efficiency(
    thrust_avg,
    power_avg,
    characteristic_speed
    ):

    if(power_avg == 0):
        return(np.nan)
    else:
        return(characteristic_speed * thrust_avg / power_avg)
