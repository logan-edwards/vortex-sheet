'''
main.py

This file is the entry of the program. It contains three functions:

    main: Entrypoint when main.py is run from the command line.

    set_params: Runs on startup to set the simulation parameters

    run_sim: Runs the simulation using the values in params.py. Returns an
    instance of the BoundSheet and FreeSheet class, populated with data from the
    simulation. 
'''

import params
import classes
import functions

def set_params():
    return(0)

def run_sim():
    body = classes.BoundSheet()
    free_sheet = classes.FreeSheet()

    free_sheet.append_circulation(
        body.z_chebyshev[0,0] - body.tangent_vector[0] * params.eps_shed,
        0,
        0
    )

    for timestep in range(1, params.Nt):
        x = functions.compute_bound_sheet(
            params.Nb,
            timestep,
            body.normal_vector[timestep],
            body.z_chebyshev[:,timestep],
            body.z_collocation[:,timestep],
            body.dzdt_collocation[:,timestep],
            free_sheet.z[:,timestep],
            free_sheet.circulation[:,timestep],
            body.L[timestep],
            params.delta
        )

        body.sigmas[:,timestep] = x[0:params.Nb+1]

        free_sheet.append_circulation(
            body.z_chebyshev[0,timestep] - (
                body.tangent_vector[timestep] * params.eps_shed
            ),
            x[params.Nb+1],
            timestep
        )

        free_sheet.dzdt[:,timestep] = functions.compute_sheet_velocity(
            params.Nb,
            timestep,
            body.z_chebyshev[:,timestep],
            free_sheet.z[:,timestep],
            free_sheet.circulation,
            body.sigmas[:,timestep],
            body.L[timestep],
            params.delta
        )

        free_sheet.z[0:timestep,timestep] = functions.update_sheet_position(
            timestep,
            free_sheet.z[:,timestep],
            free_sheet.dzdt[:,timestep],
            free_sheet.dzdt[:,timestep-1],
            params.dt
        )

        if(timestep > 2):
            body.dsigmadt[:,timestep] = (
                3 * body.sigmas[:,timestep]
                - 4 * body.sigmas[:,timestep-1]
                + body.sigmas[:,timestep-2]
            ) / (2*params.dt)

            free_sheet.dGammadt[timestep] = (
                3 * free_sheet.circulation[timestep]
                - 4 * free_sheet.circulation[timestep-1]
                + free_sheet.circulation[timestep-2]
            )

        body.pressures[:,timestep] = functions.compute_pressure_distribution(
            params.Nb,
            timestep,
            body.tangent_vector[timestep],
            body.z_chebyshev[:,timestep],
            body.z_collocation[:,timestep],
            body.dzdt_chebyshev[:,timestep],
            free_sheet.z[:,timestep],
            free_sheet.circulation[:timestep],
            body.sigmas[:,timestep],
            body.dsigmadt[:,timestep],
            body.dGammadt[timestep],
            body.L[timestep],
            body.dLdt[timestep]
        )

    functions.compute_forces(
        params.Nb,
        params.Nt,
        body.normal_vector,
        body.tangent_vector,
        body.pressures,
        body.sigmas[params.Nb,:],
        body.L
    )

    functions.compute_power_input(
        params.Nb,
        params.Nt,
        body.normal_vector,
        body.dzdt_chebyshev,
        body.pressures,
        body.L
    )

    return body, free_sheet

def main():
    run_sim()
    return(0)