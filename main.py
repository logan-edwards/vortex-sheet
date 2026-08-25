'''
main.py

This file implements the simulation.

Functions:

run_sim: Runs the simulation subject to parameters h0, phi_h, theta0, phi_theta,
L0, phi_L, alpha0, and f.

main: Gets parameters from the command line and runs the simulation.

Dependencies: numpy, time
'''

import params
import classes
import functions

import numpy as np
from time import time_ns

def run_sim(
    heaving_amplitude,
    heaving_phase,
    pitching_amplitude,
    pitching_phase,
    lengthening_amplitude,
    lengthening_phase,
    pivot_location,
    frequency
    ):

    if(params.enable_animation == True):
        x_anim_sheet = np.full((params.Nt,params.Nt), np.nan)
        y_anim_sheet = np.full((params.Nt,params.Nt), np.nan)

    body = classes.BoundSheet(
        heaving_amplitude,
        heaving_phase,
        pitching_amplitude,
        pitching_phase,
        lengthening_amplitude,
        lengthening_phase,
        pivot_location,
        frequency
    )

    free_sheet = classes.FreeSheet()
    free_sheet.append_circulation(0,
        body.x_chebyshev[0,0] - params.eps_shed * body.tangent_x[0],
        body.y_chebyshev[0,0] - params.eps_shed * body.tangent_y[0]
    )
    
    free_sheet.circulation[0] = 0
    free_sheet.x[0] = (
        body.x_chebyshev[0,0] - params.eps_shed * body.tangent_x[0]
    )
    free_sheet.y[0] = (
        body.y_chebyshev[0,0] - params.eps_shed * body.tangent_y[0]
    )

    for timestep in range(1, params.Nt):
        if(params.enable_animation == True):
            x_anim_sheet[:timestep,timestep] = free_sheet.x
            y_anim_sheet[:timestep,timestep] = free_sheet.y

        x = functions.compute_bound_sheet(
            body.normal_x[timestep],
            body.normal_y[timestep],
            body.x_chebyshev[:,timestep],
            body.y_chebyshev[:,timestep],
            body.x_collocation[:,timestep],
            body.y_collocation[:,timestep],
            body.dxdt_collocation[:,timestep],
            body.dydt_collocation[:,timestep],
            free_sheet.x,
            free_sheet.y,
            free_sheet.circulation,
            body.L[timestep],
            params.delta
        )
        body.sigmas[:,timestep] = x[0:params.Nb+1]
        
        # Append circulation to the free sheet
        free_sheet.append_circulation(
            x[params.Nb+1],
            body.x_chebyshev[0,timestep] - (
                params.eps_shed * body.tangent_x[timestep]
            ),
            body.y_chebyshev[0,timestep] - (
                params.eps_shed * body.tangent_y[timestep]
            )
        )

        dxdt_prev = np.copy(free_sheet.dxdt)
        dydt_prev = np.copy(free_sheet.dydt)

        free_sheet.dxdt, free_sheet.dydt = functions.compute_sheet_velocity(
            free_sheet.x,
            free_sheet.y,
            body.x_chebyshev[:,timestep],
            body.y_chebyshev[:,timestep],
            free_sheet.circulation,
            body.sigmas[:,timestep],
            body.L[timestep],
            params.delta
        )
        
        free_sheet.x, free_sheet.y = functions.update_sheet_position(
            free_sheet.x,
            free_sheet.y,
            free_sheet.dxdt,
            free_sheet.dydt,
            dxdt_prev,
            dydt_prev,
            params.dt
        )

        if(timestep > 2):
            body.dsigmadt[:,timestep] = (3 * body.sigmas[:,timestep]
                                         - 4 * body.sigmas[:,timestep-1]
                                         + body.sigmas[:,timestep-2]
                                         ) / (2 * params.dt)
            free_sheet.dgammadt = (3 * free_sheet.circulation[timestep]
                                   - 4 * free_sheet.circulation[timestep-1]
                                   + free_sheet.circulation[timestep - 2]
                                   ) / (2 * params.dt)

        body.pressures[:,timestep] = functions.compute_pressure_distribution(
            body.tangent_x[timestep],
            body.tangent_y[timestep],
            body.x_collocation[:,timestep],
            body.y_collocation[:,timestep],
            body.x_chebyshev[:,timestep],
            body.y_chebyshev[:,timestep],
            body.dxdt_chebyshev[:,timestep],
            body.dydt_chebyshev[:,timestep],
            free_sheet.x,
            free_sheet.y,
            free_sheet.circulation,
            body.sigmas[:,timestep],
            body.dsigmadt[:,timestep],
            free_sheet.dgammadt,
            body.L[timestep],
            body.dLdt[timestep]
        )

    body.force_x, body.force_y = functions.compute_forces(
        body.normal_x,
        body.normal_y,
        body.tangent_x,
        body.tangent_y,
        body.sigmas[params.Nb,:],
        body.pressures,
        body.L
    )

    body.power_in = functions.compute_power_in(
        body.normal_x,
        body.normal_y,
        body.dxdt_chebyshev,
        body.dydt_chebyshev,
        body.pressures,
        body.L
    )

    if(params.enable_animation == True):
        anim_string = params.animation_name + params.animation_format
        functions.animate_motion(
            body.x_chebyshev, 
            body.y_chebyshev,
            x_anim_sheet,
            y_anim_sheet,
            np.linspace(0, params.T, params.Nt), 
            anim_string
        )
    
    return body, free_sheet

def main():
    t_cutoff = 2 / params.f
    params.T = 2 * t_cutoff
    params.Nt = int(params.T / params.dt) + 1

    body,sheet = run_sim(
        params.h0,
        params.phi_h,
        params.theta0,
        params.phi_theta,
        params.L0,
        params.phi_L,
        params.alpha0,
        params.f
    )

    time = np.linspace(0,params.T,params.Nt)
    functions.plot_time_dependent_quantity(time,body.force_x,'Thrust')
    functions.plot_time_dependent_quantity(time,body.power_in,'Power Input')
    start_index = np.searchsorted(time, t_cutoff, side='right')

    time_steady = time[start_index:]
    force_x_steady = body.force_x[start_index:]
    power_steady = body.power_in[start_index:]

    avg_thrust_steady = functions.compute_time_average(
        time_steady, 
        force_x_steady
    )

    avg_power_steady = functions.compute_time_average(
        time_steady, 
        power_steady
    )

    avg_sigma = functions.compute_time_average(
        time_steady, 
        np.abs(body.sigmas[params.Nb,start_index:])
    )

    print(f"Efficiency = {avg_thrust_steady * params.u / avg_power_steady}")
    print(f"Leading Edge Sigma (avg) = {avg_sigma}")
main()