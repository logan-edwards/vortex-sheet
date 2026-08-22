'''
main.py

This file implements the simulation.

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
            body.L[timestep],
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
                                         body.L[timestep],
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

    params.enable_animation = False
    run_single_sim = False
    run_freq_sweep = False
    run_length_sweep = False
    run_alpha_sweep = True

    '''
        Temp code to get some animations
    '''

    if run_single_sim == True:
        freq = 0.2
        t_cutoff = 2 / freq
        #params.T = 1 * t_cutoff
        params.T = 2 * t_cutoff
        #params.T = 3*t_cutoff
        params.Nt = int(params.T/params.dt) + 1
        body,sheet = run_sim(
            0.1,
            0,
            np.pi / 9,
            270 * np.pi / 180,
            0.5,
            45 * np.pi / 180,
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

        avg_thrust_steady = functions.compute_time_average(time_steady, force_x_steady)

        avg_power_steady = functions.compute_time_average(time_steady, power_steady)

        avg_sigma = functions.compute_time_average(time_steady, np.abs(body.sigmas[params.Nb,start_index:]))

        print(f"Efficiency = {avg_thrust_steady * params.u / avg_power_steady}")
        print(f"Leading Edge Sigma (avg) = {avg_sigma}")

    if run_freq_sweep == True:
        '''
            Tweak these parameters to effect sampling size and fixed parameters
        '''
        frequency_samples = 8
        phase_samples = 12
        trial_name = 'Trial000'
        
        c0 = 0.125
        theta0 = 15.0 * np.pi / 180.0
        phi_theta = 270.0 * np.pi / 180.0
        pivot = 1.0
        L0 = 1.5


        '''
            Parameters determined by sampled data
        '''
        reduced_freq = np.linspace(0.2, 0.8, frequency_samples)
        phase_angle = np.linspace(30*np.pi/180, 120*np.pi/180, phase_samples)

        saved_thrust = np.zeros((frequency_samples, phase_samples))
        saved_power = np.zeros((frequency_samples, phase_samples))
        saved_LES = np.zeros((frequency_samples, phase_samples))
        saved_efficiency = np.zeros((frequency_samples, phase_samples))
        saved_St = np.zeros((frequency_samples, phase_samples))

        for i in range(frequency_samples):
            for j in range(phase_samples):
                '''
                    Calculate frequency from reduced frequency,
                    then set up appropriate cutoff times and total sim times
                '''
                frequency = params.u * reduced_freq[i] / 2.0
                t_cutoff = 2.0 / frequency
                params.T = 2 * t_cutoff
                params.Nt = int(params.T/params.dt) + 1
                body,sheet = run_sim(
                    c0,
                    0,
                    theta0,
                    phi_theta,
                    L0,
                    phase_angle[j],
                    pivot,
                    frequency
                )
                time = np.linspace(0,params.T,params.Nt)
                start_index = np.searchsorted(time, t_cutoff, side='right')

                avg_thrust_steady = functions.compute_time_average(time[start_index:], body.force_x[start_index:])
                avg_power_steady = functions.compute_time_average(time[start_index:], body.power_in[start_index:])
                efficiency = avg_thrust_steady * params.u / avg_power_steady
                avg_sigma = functions.compute_time_average(time[start_index:], np.abs(body.sigmas[params.Nb,start_index:]))
                St = 2 * frequency * (c0 + 2.0 * theta0) / params.u

                saved_thrust[i,j] = avg_thrust_steady
                saved_power[i,j] = avg_power_steady
                saved_efficiency[i,j] = efficiency
                saved_LES[i,j] = avg_sigma
                saved_St[i,j] = St

                print(f"Efficiency = {efficiency}")
                print(f"St = {St}")
                print(f"Leading Edge Sigma (avg) = {avg_sigma}")

                functions.write_report([
                    phase_angle[j],
                    reduced_freq[i],
                    avg_thrust_steady,
                    avg_power_steady,
                    efficiency,
                    avg_sigma,
                    St
                    ],
                    [
                        'phase angle (phi_L)',
                        'reduced frequency f*',
                        '<thrust>',
                        '<power,input>',
                        'efficiency',
                        '<|LES|>',
                        'Strouhal Number (St)'
                    ],
                    trial_name+'.csv'
                )

        '''
        functions.plot_polar(phase_angle, reduced_freq, saved_thrust, 'Thrust')
        functions.plot_polar(phase_angle, reduced_freq, saved_power, 'Input Power')
        functions.plot_polar(phase_angle, reduced_freq, saved_efficiency, 'Froude Efficiency')
        functions.plot_polar(phase_angle, reduced_freq, saved_LES, 'Average |LES|')
        functions.plot_polar(phase_angle, reduced_freq, saved_St, 'Strouhal Number (St)')
        '''

        functions.plot_polar_rawdata(phase_angle, reduced_freq, saved_thrust, 'Thrust')
        functions.plot_polar_rawdata(phase_angle, reduced_freq, saved_power, 'Input Power')
        functions.plot_polar_rawdata(phase_angle, reduced_freq, saved_efficiency, 'Froude Efficiency')
        functions.plot_polar_rawdata(phase_angle, reduced_freq, saved_LES, 'Average |LES|')
        functions.plot_polar_rawdata(phase_angle, reduced_freq, saved_St, 'Strouhal Number (St)')
    
    if run_length_sweep == True:
        '''
            Tweak these parameters to effect sampling size and fixed parameters
        '''
        length_samples = 9
        phase_samples = 12
        trial_name = 'Trial_f000'
        
        c0 = 0.125
        theta0 = 15.0 * np.pi / 180.0
        phi_theta = 270.0 * np.pi / 180.0
        pivot = 1.0
        reduced_freq = 0.2

        '''
            Parameters determined by sampled data
        '''
        lengths = np.linspace(0, 1.8, length_samples)
        phase_angle = np.linspace(30*np.pi/180, 120*np.pi/180, phase_samples)

        saved_thrust = np.zeros((length_samples, phase_samples))
        saved_power = np.zeros((length_samples, phase_samples))
        saved_LES = np.zeros((length_samples, phase_samples))
        saved_efficiency = np.zeros((length_samples, phase_samples))
        saved_St = np.zeros((length_samples, phase_samples))

        for i in range(length_samples):
            for j in range(phase_samples):
                '''
                    Calculate frequency from reduced frequency,
                    then set up appropriate cutoff times and total sim times
                '''
                frequency = params.u * reduced_freq / 2.0
                t_cutoff = 2.0 / frequency
                params.T = 2 * t_cutoff
                params.Nt = int(params.T/params.dt) + 1
                body,sheet = run_sim(
                    c0,
                    0,
                    theta0,
                    phi_theta,
                    lengths[i],
                    phase_angle[j],
                    pivot,
                    frequency
                )
                time = np.linspace(0,params.T,params.Nt)
                start_index = np.searchsorted(time, t_cutoff, side='right')

                avg_thrust_steady = functions.compute_time_average(time[start_index:], body.force_x[start_index:])
                avg_power_steady = functions.compute_time_average(time[start_index:], body.power_in[start_index:])
                efficiency = avg_thrust_steady * params.u / avg_power_steady
                avg_sigma = functions.compute_time_average(time[start_index:], np.abs(body.sigmas[params.Nb,start_index:]))
                St = 2 * frequency * (c0 + 2.0 * theta0) / params.u

                saved_thrust[i,j] = avg_thrust_steady
                saved_power[i,j] = avg_power_steady
                saved_efficiency[i,j] = efficiency
                saved_LES[i,j] = avg_sigma
                saved_St[i,j] = St

                print(f"Efficiency = {efficiency}")
                print(f"St = {St}")
                print(f"Leading Edge Sigma (avg) = {avg_sigma}")

                functions.write_report([
                    phase_angle[j],
                    lengths[i],
                    avg_thrust_steady,
                    avg_power_steady,
                    efficiency,
                    avg_sigma,
                    St
                    ],
                    [
                        'phase angle (phi_L)',
                        'L0',
                        '<thrust>',
                        '<power,input>',
                        'efficiency',
                        '<|LES|>',
                        'Strouhal Number (St)'
                    ],
                    trial_name+'.csv'
                )

        '''
        functions.plot_polar(phase_angle, reduced_freq, saved_thrust, 'Thrust')
        functions.plot_polar(phase_angle, reduced_freq, saved_power, 'Input Power')
        functions.plot_polar(phase_angle, reduced_freq, saved_efficiency, 'Froude Efficiency')
        functions.plot_polar(phase_angle, reduced_freq, saved_LES, 'Average |LES|')
        functions.plot_polar(phase_angle, reduced_freq, saved_St, 'Strouhal Number (St)')
        '''

        functions.plot_polar_rawdata(phase_angle, lengths, saved_thrust, 'Thrust')
        functions.plot_polar_rawdata(phase_angle, lengths, saved_power, 'Input Power')
        functions.plot_polar_rawdata(phase_angle, lengths, saved_efficiency, 'Froude Efficiency')
        functions.plot_polar_rawdata(phase_angle, lengths, saved_LES, 'Average |LES|')
        functions.plot_polar_rawdata(phase_angle, lengths, saved_St, 'Strouhal Number (St)')

    if run_alpha_sweep == True:
        '''
            Tweak these parameters to effect sampling size and fixed parameters
        '''
        alpha_samples = 5
        phase_samples = 16
        trial_name = 'Trial_alpha000'
        
        c0 = 0.125
        theta0 = 15.0 * np.pi / 180.0
        phi_theta = 270.0 * np.pi / 180.0
        pivot = 1.0
        L0 = 0.5
        reduced_freq = 0.3
        phi_L = 75 * np.pi / 180


        '''
            Parameters determined by sampled data
        '''
        alpha0 = np.linspace(-1.0, 1.0, alpha_samples)
        phase_angle = np.linspace(0, 2*np.pi, phase_samples)

        saved_thrust = np.zeros((alpha_samples, phase_samples))
        saved_power = np.zeros((alpha_samples, phase_samples))
        saved_LES = np.zeros((alpha_samples, phase_samples))
        saved_efficiency = np.zeros((alpha_samples, phase_samples))
        saved_St = np.zeros((alpha_samples, phase_samples))

        for i in range(alpha_samples):
            for j in range(phase_samples):
                '''
                    Calculate frequency from reduced frequency,
                    then set up appropriate cutoff times and total sim times
                '''
                frequency = params.u * reduced_freq / 2.0
                t_cutoff = 2.0 / frequency
                params.T = 2 * t_cutoff
                params.Nt = int(params.T/params.dt) + 1
                body,sheet = run_sim(
                    c0,
                    0,
                    theta0,
                    phi_theta,
                    L0,
                    phase_angle[j],
                    alpha0[i],
                    frequency
                )
                time = np.linspace(0,params.T,params.Nt)
                start_index = np.searchsorted(time, t_cutoff, side='right')

                avg_thrust_steady = functions.compute_time_average(time[start_index:], body.force_x[start_index:])
                avg_power_steady = functions.compute_time_average(time[start_index:], body.power_in[start_index:])
                efficiency = avg_thrust_steady * params.u / avg_power_steady
                avg_sigma = functions.compute_time_average(time[start_index:], np.abs(body.sigmas[params.Nb,start_index:]))
                St = 2 * frequency * (c0 + (2-alpha0[i])*theta0) / params.u

                saved_thrust[i,j] = avg_thrust_steady
                saved_power[i,j] = avg_power_steady
                saved_efficiency[i,j] = efficiency
                saved_LES[i,j] = avg_sigma
                saved_St[i,j] = St

                print(f"Efficiency = {efficiency}")
                print(f"St = {St}")
                print(f"Leading Edge Sigma (avg) = {avg_sigma}")

                functions.write_report([
                    phase_angle[j],
                    alpha0[i],
                    avg_thrust_steady,
                    avg_power_steady,
                    efficiency,
                    avg_sigma,
                    St
                    ],
                    [
                        'phase angle (phi_L)',
                        'pivot loc. alpha0',
                        '<thrust>',
                        '<power,input>',
                        'efficiency',
                        '<|LES|>',
                        'Strouhal Number (St)'
                    ],
                    trial_name+'.csv'
                )

        '''
        functions.plot_polar(phase_angle, reduced_freq, saved_thrust, 'Thrust')
        functions.plot_polar(phase_angle, reduced_freq, saved_power, 'Input Power')
        functions.plot_polar(phase_angle, reduced_freq, saved_efficiency, 'Froude Efficiency')
        functions.plot_polar(phase_angle, reduced_freq, saved_LES, 'Average |LES|')
        functions.plot_polar(phase_angle, reduced_freq, saved_St, 'Strouhal Number (St)')
        '''

        functions.plot_polar_rawdata(phase_angle, alpha0, saved_thrust, 'Thrust')
        functions.plot_polar_rawdata(phase_angle, alpha0, saved_power, 'Input Power')
        functions.plot_polar_rawdata(phase_angle, alpha0, saved_efficiency, 'Froude Efficiency')
        functions.plot_polar_rawdata(phase_angle, alpha0, saved_LES, 'Average |LES|')
        functions.plot_polar_rawdata(phase_angle, alpha0, saved_St, 'Strouhal Number (St)')

main()