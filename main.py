import params
import classes
import functions
import optimizer

import numpy as np
from scipy.optimize import minimize

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

    if(params.enable_animation == True):
        functions.animate_motion(
            body.x_chebyshev, 
            body.y_chebyshev,
            x_anim_sheet,
            y_anim_sheet,
            np.linspace(0, params.T, params.Nt), 
            'test_animation.mp4'
        )
    
    return body, free_sheet

def main():
    '''
    Optimization parameters:
        heaving amplitude
        heaving phase
        pitching amplitude
        pitching phase
        lengthening amplitude
        lengthening phase
        pivot location
        frequency
    '''

    params.enable_animation = False
    t_cutoff = 2 / 0.25
    params.T = 5 * t_cutoff
    params.Nt = int(params.T/params.dt) + 1

    phi_params = np.linspace(0, 2*np.pi, 12)
    length_params = np.zeros(4)
    length_params[0] = -0.5
    length_params[1] = -0.25
    length_params[2] = 0.25
    length_params[3] = 0.5

    saved_efficiencies = np.zeros((4, 12))
    saved_LES = np.zeros((4,12))

    for phi_index in range(np.size(phi_params)):
        for length_index in range(np.size(length_params)):
            print(f"--- Running with phi = {phi_params[phi_index]}, % lengthening = {length_params[length_index]}")
            body,free = run_sim(
                0.1,
                0,
                np.pi / 9.0,
                9.0 * np.pi / 4.0,
                length_params[length_index],
                phi_params[phi_index],
                1,
                0.25
            )
            time = np.linspace(0,params.T,params.Nt)
            start_index = np.searchsorted(time, t_cutoff, side='right')

            time_steady = time[start_index:]
            force_x_steady = body.force_x[start_index:]
            power_steady = body.power_in[start_index:]

            avg_thrust_steady = functions.compute_time_average(time_steady,
                                                            force_x_steady
                                                            )

            avg_power_steady = functions.compute_time_average(time_steady,
                                                            power_steady)

            print(f"Efficiency = {avg_thrust_steady * params.u / avg_power_steady}")

            functions.write_report([
                phi_params[phi_index],
                length_params[length_index],
                avg_thrust_steady,
                avg_power_steady,
                np.abs(avg_thrust_steady) * params.u / np.abs(avg_power_steady),
                np.max(np.abs(body.sigmas[params.Nb,start_index:]))
            ],
            [
                'phi',
                'length ratio',
                'thrust',
                'power',
                'efficiency',
                'LE sigma magnitude'
            ],
            'lengthening_report.csv')

            saved_efficiencies[phi_index, length_index] = np.abs(avg_thrust_steady) * params.u / np.abs(avg_power_steady)
            saved_LES[phi_index, length_index] = np.max(np.abs(body.sigmas[params.Nb,start_index:]))

    functions.plot_polar(
        phi_params,
        length_params,
        saved_efficiencies,
        'Thrust Efficiency'
    )

    function.plot_polar(
        phi_params,
        length_params,
        saved_LES,
        'Leading Edge Sigma'
    )


            
            
            



 
main()