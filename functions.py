import numpy as np
from numba import njit, prange

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import csv
import os


'''
    Vortex sheet method functions
'''

@njit
def K_delta_normal(dx, dy, nx, ny, delta):
    denominator = dx**2 + dy**2 + delta**2
    if(denominator < 1e-12):
        return(0)
    return(
        (0.5/np.pi) * (-dy * nx + dx * ny) / (dx**2 + dy**2 + delta**2)
    )

@njit
def K_delta(dx, dy, delta):
    denominator = dx**2 + dy**2 + delta**2
    if(denominator < 1e-12):
        return 0.0, 0.0
    u = (0.5/np.pi) * -dy / denominator
    v = (0.5/np.pi) * dx / denominator

    return u,v

@njit(parallel=True)
def compute_bound_sheet(nhat_x,
                        nhat_y,
                        body_x_cheb,
                        body_y_cheb,
                        body_x_coll,
                        body_y_coll,
                        body_dxdt_coll,
                        body_dydt_coll,
                        free_sheet_x,
                        free_sheet_y,
                        free_sheet_circulation,
                        delta
                        ):
    Nb = np.size(body_x_cheb) - 1
    Nf = np.size(free_sheet_circulation)

    if(Nf - 1 < 0):
        return(np.zeros(Nb+2))

    A = np.zeros((Nb+2, Nb+2))
    b = np.zeros(Nb+2)

    '''
    Construct the b vector
    '''
    for l in prange(Nb):
        bsum = 0
        bsum = bsum + nhat_x * body_dxdt_coll[l] + nhat_y * body_dydt_coll[l]
        for i in range(Nf - 1):
            bsum = bsum - 0.5 * (
                free_sheet_circulation[i+1] - free_sheet_circulation[i]) * (
                    K_delta_normal(
                        body_x_coll[l] - free_sheet_x[i+1], 
                        body_y_coll[l] - free_sheet_y[i+1], 
                        nhat_x, 
                        nhat_y, 
                        delta
                    ) + K_delta_normal(
                    body_x_coll[l] - free_sheet_x[i], 
                    body_y_coll[l] - free_sheet_y[i], 
                    nhat_x, 
                    nhat_y, 
                    delta
                    )
                )
        bsum = bsum + 0.5 * free_sheet_circulation[Nf - 1] * (K_delta_normal(
            body_x_coll[l] - body_x_cheb[0], 
            body_y_coll[l] - body_y_cheb[0], 
            nhat_x, 
            nhat_y, 
            delta
        ) + K_delta_normal(
            body_x_coll[l] - free_sheet_x[Nf - 1],
            body_y_coll[l] - free_sheet_y[Nf - 1],
            nhat_x,
            nhat_y,
            delta
            )
        )
        b[l] = bsum

    '''
    Construct the A matrix
        - 0, ..., Nb-1: enforce KBC
        Nb: Kutta condition
        Nb+1: Kelvin's circulation theorem
    '''

    '''
        Enforce KBC:
    '''
    for l in prange(Nb):
        for k in range(Nb+1):
            if(k==0 or k==Nb):
                A[l,k] = (np.pi/(2*Nb)) * K_delta_normal(
                    body_x_coll[l] - body_x_cheb[k], 
                    body_y_coll[l] - body_y_cheb[k], 
                    nhat_x, 
                    nhat_y,
                    0
                    )
            else:
                A[l,k] = (np.pi/Nb) * K_delta_normal(
                    body_x_coll[l] - body_x_cheb[k], 
                    body_y_coll[l] - body_y_cheb[k], 
                    nhat_x, 
                    nhat_y,
                    0
                    )
        A[l,Nb+1] = 0.5 * (
            K_delta_normal(
                body_x_coll[l] - body_x_cheb[0], 
                body_y_coll[l] - body_y_cheb[0], 
                nhat_x, 
                nhat_y, 
                delta
                ) + 
            K_delta_normal(
                body_x_coll[l] - free_sheet_x[Nf-1], 
                body_y_coll[l] - free_sheet_y[Nf-1], 
                nhat_x, 
                nhat_y, 
                delta
                )
            )
    '''
        Enforce Kutta condition:
    '''
    A[Nb, 0] = 1.0

    '''
        Enforce KCT:
    '''
    A[Nb+1, 0] = np.pi/(2*Nb)
    A[Nb+1, Nb] = np.pi/(2*Nb)
    for k in range (1,Nb):
        A[Nb+1, k] = np.pi/Nb
    A[Nb+1, Nb+1] = 1.0

    '''
        Solution is of the form:
        x = sigma0,
            sigma1,
            ...
            sigmaNb,
            Gamma^-
    '''
    return(np.linalg.solve(A,b))

@njit(parallel=True)
def compute_sheet_velocity(free_sheet_x,
                           free_sheet_y,
                           body_x_cheb,
                           body_y_cheb,
                           free_sheet_circulation,
                           sigma,
                           delta,
                           ):
    Nb = np.size(body_x_cheb)
    Nf = np.size(free_sheet_circulation)

    u_bound = np.zeros(Nf)
    u_free = np.zeros(Nf)
    v_bound = np.zeros(Nf)
    v_free = np.zeros(Nf)

    '''
        Integrate over the free sheet
    '''
    for i in prange(Nf):
        for j in range(Nf - 1):
            u1,v1 = K_delta(free_sheet_x[i] - free_sheet_x[j+1],
                            free_sheet_y[i] - free_sheet_y[j+1],
                            delta)
            u2,v2 = K_delta(free_sheet_x[i] - free_sheet_x[j],
                            free_sheet_y[i] - free_sheet_y[j],
                            delta)
            u_free[i] = u_free[i] + 0.5 * (
                free_sheet_circulation[j+1] - free_sheet_circulation[j]
            ) * (u1 + u2)
            v_free[i] = v_free[i] + 0.5 * (
                free_sheet_circulation[j+1] - free_sheet_circulation[j]
            ) * (v1 + v2)
    
    u_free = u_free
    v_free = v_free

    '''
        Integrate over the bound sheet
    '''
    for i in prange(Nf):
        for k in range(Nb):
            u1,v1 = K_delta(free_sheet_x[i] - body_x_cheb[k],
                            free_sheet_y[i] - body_y_cheb[k],
                            0)
            if(k==0 or k==Nb):
                u_bound[i] = u_bound[i] + (np.pi / (2*Nb)) * sigma[k] * u1
                v_bound[i] = v_bound[i] + (np.pi / (2*Nb)) * sigma[k] * v1
            else:
                u_bound[i] = u_bound[i] + (np.pi / Nb) * sigma[k] * u1
                v_bound[i] = v_bound[i] + (np.pi / Nb) * sigma[k] * v1
    u_bound = u_bound
    v_bound = v_bound

    return u_bound + u_free, v_bound + v_free

@njit(parallel=True)
def update_sheet_position(free_sheet_x,
                          free_sheet_y,
                          free_sheet_dxdt,
                          free_sheet_dydt,
                          free_sheet_dxdt_prev,
                          free_sheet_dydt_prev,
                          dt
                          ):
    Nf = np.size(free_sheet_x)
    x_new = np.zeros(Nf)
    y_new = np.zeros(Nf)

    for i in prange(Nf-1):
        x_new[i] = free_sheet_x[i] + 1.5 * dt * free_sheet_dxdt[i] - 0.5 * dt * free_sheet_dxdt_prev[i]
        y_new[i] = free_sheet_y[i] + 1.5 * dt * free_sheet_dydt[i] - 0.5 * dt * free_sheet_dydt_prev[i]
    x_new[Nf-1] = free_sheet_x[Nf-1] + dt * free_sheet_dxdt[Nf-1]
    y_new[Nf-1] = free_sheet_y[Nf-1] + dt * free_sheet_dydt[Nf-1]
    
    return x_new, y_new

@njit(parallel=True)
def compute_pressure_distribution(tan_x,
                                  tan_y,
                                  body_x_coll,
                                  body_y_coll,
                                  body_x_cheb,
                                  body_y_cheb,
                                  body_dxdt_cheb,
                                  body_dydt_cheb,
                                  free_sheet_x,
                                  free_sheet_y,
                                  free_sheet_circulation,
                                  sigma,
                                  dsigmadt,
                                  dgammadt,
                                  L
                                  ):
    Nb = np.size(body_x_coll)
    Nf = np.size(free_sheet_x)

    p = np.zeros(Nb + 1)
    mu = np.zeros(Nb + 1)
    tau = np.zeros(Nb + 1)
    gamma = np.zeros(Nb + 1)
    I_coll_x = np.zeros(Nb)
    I_coll_y = np.zeros(Nb)
    X_x = np.zeros(Nb + 1)
    X_y = np.zeros(Nb + 1)
    I_cheb_x = np.zeros(Nb + 1)
    I_cheb_y = np.zeros(Nb + 1)
    I_free_x = np.zeros(Nb + 1)
    I_free_y = np.zeros(Nb + 1)
    I_dsigmadt = np.zeros(Nb + 1)

    '''
        Reconstruct the integral at the chebyshev nodes using collocation data
    '''

    for l in prange(Nb):
        for k in range(Nb + 1):
            Ix,Iy = K_delta(body_x_coll[l] - body_x_cheb[k],
                            body_y_coll[l] - body_y_cheb[k],
                            0
                            )
            if(k == 0 or k == Nb):
                I_coll_x[l] = I_coll_x[l] + (np.pi/(2*Nb)) * sigma[k] * Ix
                I_coll_y[l] = I_coll_y[l] + (np.pi/(2*Nb)) * sigma[k] * Iy
            else:
                I_coll_x[l] = I_coll_x[l] + (np.pi/Nb) * sigma[k] * Ix
                I_coll_y[l] = I_coll_y[l] + (np.pi/Nb) * sigma[k] * Iy
    for j in prange(Nb):
        for l in range(Nb):
            X_x[j] = X_x[j] + 2 * I_coll_x[l] * np.cos(j*np.pi*(2*l+1)/(2*Nb))
            X_y[j] = X_y[j] + 2 * I_coll_y[l] * np.cos(j*np.pi*(2*l+1)/(2*Nb))
    for k in range(Nb + 1):
        I_cheb_x[k] = I_cheb_x[k] + X_x[0] / (2*Nb)
        I_cheb_y[k] = I_cheb_y[k] + X_y[0] / (2*Nb)
        for j in range(1, Nb):
            I_cheb_x[k] = I_cheb_x[k] + (X_x[j]/Nb) * np.cos(j*k*np.pi/Nb)
            I_cheb_y[k] = I_cheb_y[k] + (X_y[j]/Nb) * np.cos(j*k*np.pi/Nb)
    
    '''
        Compute the integral over the free sheet
    '''

    for k in prange(Nb + 1):
        for j in range(Nf-1):
            ux1,uy1 = K_delta(body_x_cheb[k] - free_sheet_x[j+1],
                              body_y_cheb[k] - free_sheet_y[j+1],
                              0
                              )
            ux2,uy2 = K_delta(body_x_cheb[k] - free_sheet_x[j],
                              body_y_cheb[k] - free_sheet_y[j],
                              0
                              )
            I_free_x[k] = I_free_x[k] + 0.5 * (
                free_sheet_circulation[j+1] - free_sheet_circulation[j]
            ) * (ux1 + ux2)
            I_free_y[k] = I_free_y[k] + 0.5 * (
                free_sheet_circulation[j+1] - free_sheet_circulation[j]
            ) * (uy1 + uy2)
    
    mu = tan_x * (I_cheb_x + I_free_x) + tan_y * (I_cheb_y + I_free_y)
    tau = tan_x * body_dxdt_cheb + tan_y * body_dydt_cheb
    
    for k in range(1, Nb):
        gamma[k] = sigma[k] / (L * np.sin(k*np.pi/Nb))
    
    for k in range(1, Nb + 1):
        I_dsigmadt[k] = I_dsigmadt[k-1] + (np.pi / (2*Nb)) * (
            dsigmadt[k] + dsigmadt[k-1]
        )

    p = (mu - tau) * gamma + I_dsigmadt + dgammadt

    return(p)

@njit(parallel=True)
def compute_forces(nhat_x,
                   nhat_y,
                   tan_x,
                   tan_y,
                   leading_edge_sigmas,
                   pressures,
                   L
                   ):
    Nb = np.size(pressures[:,0]) - 1
    Nt = np.size(pressures[0,:])

    Fx = np.zeros(Nt)
    Fy = np.zeros(Nt)

    for t in prange(Nt):
        for k in range(1, Nb-1):
            Fx[t] = Fx[t] - nhat_x[t] * L[t] * (np.pi / Nb) * pressures[k,t] * np.sin(np.pi * k / Nb)
            Fy[t] = Fy[t] - nhat_y[t] * L[t] * (np.pi / Nb) * pressures[k,t] * np.sin(np.pi * k / Nb)
        
        suction_x = tan_x[t] * (np.pi / 8) * (leading_edge_sigmas[t]/L[t])**2 # does this need to scale by length?
        suction_y = tan_y[t] * (np.pi / 8) * (leading_edge_sigmas[t]/L[t])**2 # does this need to scale by length?

        Fx[t] = Fx[t] + suction_x
        Fy[t] = Fy[t] + suction_y

    return Fx,Fy

@njit(parallel=True)
def compute_power_in(nhat_x,
                     nhat_y,
                     body_dxdt,
                     body_dydt,
                     pressures,
                     L
                  ):
    Nb = np.size(pressures[:,0]) - 1
    Nt = np.size(pressures[0,:])

    P = np.zeros(Nt)

    for t in prange(Nt):
        for k in range(1, Nb):
            P[t] = P[t] - pressures[k,t] * (
                body_dxdt[k,t] * nhat_x[t] + body_dydt[k,t] * nhat_y[t]
            ) * L[t] * (np.pi / Nb) * np.sin(k * np.pi / Nb)
    
    return P

@njit
def compute_time_average(time, quantity):
    Nt = np.size(time)
    time_avg = 0

    for i in range(Nt-1):
        time_avg = time_avg + 0.5 * (quantity[i+1] + quantity[i]) * (time[i+1] - time[i])
    
    time_avg = time_avg / (time[Nt-1] - time[0])
    
    return(time_avg)

'''
    matplotlib helper functions
'''

def animate_motion(x, y, x_sheet, y_sheet, t, anim_name):
    fig, ax = plt.subplots()

    body_line, = ax.plot(
        x[:, 0],
        y[:, 0],
        'k-'
    )

    sheet_line, = ax.plot(
        x_sheet[:, 0],
        y_sheet[:, 0],
        'o',
        markersize=1
    )
    
    ax.set_xlim(
        min(np.nanmin(x), np.nanmin(x_sheet)) - 0.2,
        max(np.nanmax(x), np.nanmax(x_sheet)) + 0.2
    )
    ax.set_ylim(
        min(np.nanmin(y), np.nanmin(y_sheet)) - 0.2,
        max(np.nanmax(y), np.nanmax(y_sheet)) + 0.2
    )
    
    ax.set_aspect('equal')

    def update(frame):
        body_line.set_ydata(y[:, frame])
        body_line.set_xdata(x[:, frame])
        
        sheet_line.set_ydata(y_sheet[:, frame])
        sheet_line.set_xdata(x_sheet[:, frame])
        
        ax.set_title(f"t={t[frame]:.2f}")
        return body_line, sheet_line

    desired_time = 10  # seconds
    fps_desired = x.shape[1] / desired_time
    interval_desired = desired_time * 1000 / x.shape[1]

    print(f"framerate = {fps_desired} fps")

    sheet_animation = animation.FuncAnimation(
        fig,
        update,
        frames=x.shape[1],
        interval=interval_desired,
        blit=True
    )

    sheet_animation.save(anim_name, writer='ffmpeg', fps=fps_desired)

def plot_time_dependent_quantity(time, quantity, ylabel):
    fig, ax = plt.subplots()
    ax.plot(time, quantity)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(ylabel)
    plt.show()

def plot_polar(angle, radius, quantity, label):
    # 1. Generate sample data (theta in radians, r as radius)
    # 2. Create figure and polar axis

    theta_grid, r_grid = np.meshgrid(angle, radius)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    pcm = ax.contourf(theta_grid, r_grid, quantity, label, levels=50, cmap="viridis")

    cbar = fig.colorbar(pcm, ax=ax, pad=0.1)
    cbar.set_label(label)

    ax.set_title(label, va="bottom")
    ax.grid(True)

    plt.legend()
    plt.show()

def write_report(data, data_labels, filename):

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new or empty
        if ((not os.path.exists(filename)) or (os.path.getsize(filename) == 0)):
            writer.writerow(data_labels)

        # Handle dict data
        if isinstance(data, dict):
            row = [data.get(label, "") for label in data_labels]
        else:
            row = list(np.asarray(data).ravel())

        writer.writerow(row)
    
    return(0)