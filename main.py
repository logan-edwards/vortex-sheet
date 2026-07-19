import params
import classes
import functions
import optimizer

import numpy as np
from scipy.optimize import minimize
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

    params.enable_animation = True
    run_sweep = False
    run_optimizer = False

    '''
        Temp code to get some animations
    '''

    
    freq = 0.4
    t_cutoff = 2 / freq
    #params.T = 1 * t_cutoff
    params.T = 3*t_cutoff
    params.Nt = int(params.T/params.dt) + 1
    body,sheet = run_sim(
        0.1,
        0,
        np.pi / 9,
        3*np.pi/2,
        1.1,
        np.pi/2,
        1,
        freq
    )
    time = np.linspace(0,params.T,params.Nt)
    functions.plot_time_dependent_quantity(time,body.force_x,'thrust')
    functions.plot_time_dependent_quantity(time,body.power_in,'power input')
    start_index = np.searchsorted(time, t_cutoff, side='right')

    time_steady = time[start_index:]
    force_x_steady = body.force_x[start_index:]
    power_steady = body.power_in[start_index:]

    avg_thrust_steady = functions.compute_time_average(time_steady, np.maximum(0,force_x_steady))

    avg_power_steady = functions.compute_time_average(time_steady, np.maximum(0, power_steady))

    print(f"Efficiency = {avg_thrust_steady * params.u / avg_power_steady}")
    
    
    if run_sweep == True:
        t_cutoff = 2 / 0.4
        params.T = 3 * t_cutoff
        params.Nt = int(params.T/params.dt) + 1

        phi_params = np.linspace(0, 2*np.pi, 36)
        #frequency_params = np.linspace(0.2, 0.8, 8)
        length_params = np.linspace(0,1.5,4)

        saved_efficiencies = np.zeros((np.size(phi_params), np.size(length_params)))
        saved_LES = np.zeros((np.size(phi_params),np.size(length_params)))
        saved_thrust = np.zeros((np.size(phi_params),np.size(length_params)))
        saved_power = np.zeros((np.size(phi_params),np.size(length_params)))
        saved_St = np.zeros((np.size(phi_params),np.size(length_params)))

        for phi_index in range(np.size(phi_params)):
            for length_index in range(np.size(length_params)):
                print(f"--- Running with phi = {phi_params[phi_index]/np.pi}pi, L0 = {length_params[length_index]}")
                #t_cutoff = 2 / frequency_params[frequency_index]
                #params.T = 5 * t_cutoff
                #params.Nt = int(params.T/params.dt) + 1
                body,free = run_sim(
                    0.1,
                    0.0,
                    np.pi / 9.0,
                    270.0 * np.pi / 180.0,
                    length_params[length_index],
                    phi_params[phi_index],
                    1,
                    0.4
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
                                                                np.maximum(0,power_steady)
                )

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
                    'L0',
                    'thrust',
                    'power',
                    'efficiency',
                    'LE sigma magnitude'
                ],
                'new_param_sweep.csv')

                if(avg_thrust_steady < 0):
                    avg_thrust_steady = 0

                saved_efficiencies[phi_index, length_index] = avg_thrust_steady * params.u / avg_power_steady
                saved_LES[phi_index, length_index] = np.max(np.abs(body.sigmas[params.Nb,start_index:]))
                saved_thrust[phi_index, length_index] = avg_thrust_steady
                saved_power[phi_index, length_index] = avg_power_steady

        functions.plot_polar(
            phi_params,
            length_params,
            np.transpose(saved_efficiencies),
            'Efficiency'
        )

        functions.plot_polar(
            phi_params,
            length_params,
            np.transpose(saved_LES),
            'Leading Edge Sigma'
        )

        functions.plot_polar(
            phi_params,
            length_params,
            np.transpose(saved_thrust),
            'Thrust'
        )

        functions.plot_polar(
            phi_params,
            length_params,
            np.transpose(saved_power),
            'Power'
        )
    
    if run_optimizer == True:
        params.sim_start_time = time_ns()
        initial_guess = np.zeros(6)
        # heaving amplitude, phase
        initial_guess[0] = 0.1
        initial_guess[1] = 0
        # pitching amplitude, phase
        initial_guess[2] = np.pi / 9.0
        initial_guess[3] = 3.0 * np.pi / 2.0
        # lengthening %, phase
        initial_guess[4] = -0.5
        initial_guess[5] = np.pi
        # pivot location
        #initial_guess[6] = 1.0
        result = minimize(
            optimizer.objective,
            initial_guess,
            method='L-BFGS-B',
            jac='cs',
            bounds=[
                (0,1),
                (0,2*np.pi),
                (0,np.pi/4),
                (0,2*np.pi),
                (-0.95,0.95),
                (0,2*np.pi),
            ],
            tol=1e-6,
            options={
                'maxiter': 500
            }
        )
        print(f"Success status = {result.success} in iterations {result.nfev}, message = {result.message}")
        print(f"Gradient at optimal solution = {result.jac}")
        print(f"Optimal efficiency (WITH PENALTY) = {-result.fun}")
        print(f"Obtained with parameter set: {result.x}")

    
    


                
                
                



 
main()