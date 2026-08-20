# vortex_sheet_methods
Vortex sheet method for modeling the aerodynamics of thin airfoil.

At its core, this code solves the Birkhoff-Rott equation, a singular integro-differential equation given by
$$
\frac{\partial \overline{z}}{\partial t} = \frac{1}{2\pi i}\int_C \frac{\gamma(s',t)}{z-z(s',t)}\,\mathrm{d}s'
$$
where $C\subseteq\mathbb{C}$ is the collection of points along the vortex sheet, and the Cauchy principal value of the integral is taken. The combined vortex sheet $C$ consists of the *bound vortex sheet*, $C_b$, which is the boundary layer along the body in the limit of inviscid flow around an infinitely thin plate, and the *free vortex sheet*, $C_f$, which is the wake behind the body. 

The algorithm we utilize is as follows; at each discrete time $t_i$, we:

1. solve a system of equations representing physical constraints for each point on the bound sheet
2. evaluate the (de-singularized version of the) Birkhoff-Rott equations for all points on the free sheet
3. advance the free sheet position in time using velocity data from (2)

This work is a refactoring of code I produced during the 2026 U(M) REU in Mathematics.