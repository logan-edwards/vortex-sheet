import params
import classes
import functions

import numpy as np
from scipy.optimize import minimize

def run_sim(parameters):
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

    frequency = 1

    t_cutoff = 2/frequency
    params.T = 2.5 * t_cutoff
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
            free_sheet.dgammadt
        )

    body.force_x, body.force_y = functions.compute_forces(
        body.normal_x,
        body.normal_y,
        body.tangent_x,
        body.tangent_y,
        body.sigmas[params.Nb,:],
        body.pressures
    )

    body.power_in = functions.compute_power_in(
        body.normal_x,
        body.normal_y,
        body.dxdt_chebyshev,
        body.dydt_chebyshev,
        body.pressures
    )

    time = np.linspace(0, params.T, params.Nt)
    start_index = np.searchsorted(time, t_cutoff, side='right')
    print(f"Start index = {start_index}, final index = {params.Nt}")
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

    if abs(power_avg) < 1e-12:
        return -1e6, 1


    print(f"Froude eff = {thrust_avg * params.u / power_avg}")
    print(f"Lifting eff = {lift_avg * params.u / power_avg}")

    #return(thrust_avg * params.u / power_avg, np.max(np.abs(sigma_lead_truncated)))

    if(params.enable_animation == True):
        functions.animate_motion(
            body.x_chebyshev, 
            body.y_chebyshev,
            x_anim_sheet,
            y_anim_sheet,
            np.linspace(0, params.T, params.Nt), 
            'test_animation.mp4'
        )

def objective(parameters):
    froude_eff, max_sigma = run_sim(parameters)
    
    return -froude_eff + 10 * max_sigma**2

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
    '''x0 = np.zeros(7)
    x0[0] = 0.1
    x0[1] = 0
    x0[2] = np.pi / 8
    x0[3] = 0
    x0[4] = 1
    x0[5] = 0
    x0[6] = 1
    #x0[7] = 1
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac='2-point',
        bounds=[
            (0.0, 2.0),
            (0.0, 2*np.pi),
            (0,np.pi/2),
            (0,2*np.pi),
            (-0.99,2.0),
            (0,2*np.pi),
            (0,1.0)
        ],
        options={
            'maxiter': 50,
            'ftol': 1e-6,
            'gtol': 1e-5,
            'finite_diff_rel_step': 1e-3,
            'disp': True
        }
    )

    print(f"Best parameters = {result.x}")
    print(f"Optimum efficiency = {-result.fun}")'''

    x = np.zeros(7)
    x[0] = 8.42440635e-03
    x[1] = 2.38731930e-07
    x[2] = 3.30827433e-02
    x[3] = 4.65665430e-07
    x[4] = -8.22348598e-01
    x[5] = 1.20512527e-06
    x[6] = 1.00000000e+00
 
main()