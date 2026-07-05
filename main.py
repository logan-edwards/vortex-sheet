import params
import classes
import functions

import numpy as np

def main():
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
    body = classes.BoundSheet(
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1
    )

    free_sheet = classes.FreeSheet()
    free_sheet.circulation[0] = 0
    free_sheet.x[0,0] = body.x_chebyshev[0,0] - params.eps_shed * body.tangent_x
    free_sheet.y[0,0] = body.y_chebyshev[0,0] - params.eps_shed * body.tangent_y

    for timestep in range(1, params.Nt):
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
        free_sheet.circulation[timestep] = x[params.Nb+1]
        free_sheet.x[timestep, timestep] = body.x_chebyshev[0, timestep] - params.eps_shed * body.tangent_x
        free_sheet.y[timestep, timestep] = body.y_chebyshev[0, timestep] - params.eps_shed * body.tangent_y

        '''
            Save circulation and sigma time derivatives for pressures, forces
        '''
        if(timestep > 2):
            body.dgammadt[timestep] = (
                3 * free_sheet.circulation[timestep]
                - 4 * free_sheet.circulation[timestep-1]
                + free_sheet.circulation[timestep-2]
            ) / (2 * params.dt)
            body.dsigmadt[:,timestep] = (
                3 * body.sigmas[:,timestep]
                - 4 * body.sigmas[:,timestep]
                + body.sigmas[:,timestep]
            ) / (2 * params.dt)
        
                   




    functions.animate_motion(
        body.x_chebyshev, 
        body.y_chebyshev, 
        np.linspace(0, params.T, params.Nt), 
        'test_animation.mp4'
    )


main()