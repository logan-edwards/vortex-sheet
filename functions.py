import params
import classes

import numpy as np
from numba import njit, prange

import matplotlib.pyplot as plt
import matplotlib.animation as animation

@njit
def K_delta_normal(dx, dy, nx, ny, delta):
    return(
        (0.5/np.pi) * (-dy * nx + dx * ny) / (dx**2 + dy**2 + delta**2)
    )

@njit
def K_normal(dx, dy, nx, ny, delta):
    if(dx < 1e-13 and dy < 1e-13):
        return(0)
    return(
        (0.5/np.pi) * (-dy * nx + dx * ny) / (dx**2 + dy**2)
    )

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
    Nb = np.size(body_x_cheb, axis=0)
    Nf = np.size(free_sheet_circulation, axis=1)

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
        bsum = bsum + 0.5 * free_sheet_circulation[Nf - 1] * K_delta_normal(
            body_x_coll[l] - body_x_cheb[0], 
            body_y_coll[l] - body_y_cheb[0], 
            nhat_x, 
            nhat_y, 
            delta
        )
        bsum = bsum + nhat_x * body_dxdt_coll[l] + nhat_y * body_dydt_coll[l]

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
                A[l,k] = 0.5 * np.pi * K_normal(
                    body_x_coll[l] - body_x_cheb[k], 
                    body_y_coll[l] - body_y_cheb[k], 
                    nhat_x, 
                    nhat_y, 
                    delta
                    )
            else:
                A[l,k] = np.pi * K_normal(
                    body_x_coll[l] - body_x_cheb[k], 
                    body_y_coll[l] - body_y_cheb[k], 
                    nhat_x, 
                    nhat_y, 
                    delta
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
    A[Nb+1, 0] = np.pi/2
    A[Nb+1, Nb] = np.pi/2
    for k in range (1,Nb):
        A[Nb+1, k] = np.pi
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
def advance_free_sheet():
    return(0)

@njit(parallel=True)
def compute_forces(nhat_x,
                   nhat_y,
                   body_x_coll,
                   body_y_coll,
                   body_x_cheb,
                   body_y_cheb,
                   body_dxdt_cheb,
                   body_dydt_cheb,
                   free_sheet_x,
                   free_sheet_y,
                   free_sheet_circulation,
                   dsigmadt,
                   dgammadt,
                   ):
    return(0)

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
        np.nanmin(x) - 0.2,
        np.nanmax(x) + 0.2
    )
    ax.set_ylim(
        np.nanmin(y) - 0.2,
        np.nanmax(y) + 0.2
    )
    
    ax.set_aspect('equal')

    def update(frame):
        body_line.set_ydata(y[:, frame])
        body_line.set_xdata(x[:, frame])
        '''
        sheet_line.set_ydata(np.imag(z_anim_sheet[:, frame]))
        sheet_line.set_xdata(np.real(z_anim_sheet[:, frame]))
        '''
        ax.set_title(f"t={t[frame]:.2f}")
        return body_line,#, sheet_line

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