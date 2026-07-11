import functions
import classes
import params

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def objective(parameters):
    '''
    Bound sheet attributes:
        heaving amplitude
        heaving phase
        pitching amplitude
        pitching phase
        lengthening amplitude
        lengthening phase
        pivot location
        frequency
    '''

    frequency = 0.4

    t_cutoff = 2/frequency
    params.T = 5 * t_cutoff
    params.Nt = int(params.T/params.dt) + 1

    if(params.enable_animation == True):
        x_anim_sheet = np.full((params.Nt,params.Nt), np.nan)
        y_anim_sheet = np.full((params.Nt,params.Nt), np.nan)

    body = classes.BoundSheet(
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
        parameters[6],
        frequency
    )



    free_sheet = classes.FreeSheet()
    free_sheet.append_circulation(0,
                                  body.x_chebyshev[0,0] - params.eps_shed * body.tangent_x[0],
                                  body.y_chebyshev[0,0] - params.eps_shed * body.tangent_y[0]
                                  )
    
    free_sheet.circulation[0] = 0
    free_sheet.x[0] = body.x_chebyshev[0,0] - params.eps_shed * body.tangent_x[0]
    free_sheet.y[0] = body.y_chebyshev[0,0] - params.eps_shed * body.tangent_y[0]

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
            params.delta
        )
        body.sigmas[:,timestep] = x[0:params.Nb+1]
        
        '''
            Append circulation to the free sheet
        '''
        free_sheet.append_circulation(
            x[params.Nb+1],
            body.x_chebyshev[0,timestep] - params.eps_shed * body.tangent_x[timestep],
            body.y_chebyshev[0,timestep] - params.eps_shed * body.tangent_y[timestep]
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
                                         params.delta
                                         )
        
        free_sheet.x, free_sheet.y = functions.update_sheet_position(free_sheet.x,
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
            body.dsdalpha[timestep]
        )

    body.force_x, body.force_y = functions.compute_forces(
        body.normal_x,
        body.normal_y,
        body.tangent_x,
        body.tangent_y,
        body.sigmas[params.Nb,:],
        body.pressures,
        body.dsdalpha
    )

    body.power_in = functions.compute_power_in(
        body.normal_x,
        body.normal_y,
        body.dxdt_chebyshev,
        body.dydt_chebyshev,
        body.pressures,
        body.dsdalpha
    )

    time = np.linspace(0, params.T, params.Nt)
    start_index = np.searchsorted(time, t_cutoff, side='right')
    time_truncated = time[start_index:]
    force_x_truncated = body.force_x[start_index:]
    force_y_truncated = body.force_y[start_index:]
    power_truncated = body.power_in[start_index:]

    thrust_avg = functions.compute_time_average(
        time_truncated,
        force_x_truncated
    )
    lift_avg = functions.compute_time_average(
        time_truncated,
        force_y_truncated
    )
    power_avg = functions.compute_time_average(
        time_truncated,
        power_truncated
    )

    sigma_lead_truncated = body.sigmas[params.Nb,start_index:]

    params.scipy_iteration = params.scipy_iteration + 1

    if abs(power_avg) < 1e-6:
        print(f"ITER {params.scipy_iteration}\t penalized infinite eff")
        return 1e9
    else:
        froude_eff = np.abs(thrust_avg) * params.u / np.abs(power_avg)
        print(f"ITER {params.scipy_iteration}\t eff = {froude_eff}")
        max_sigma = np.max(np.abs(sigma_lead_truncated))
        return -froude_eff + 10 * max_sigma**2