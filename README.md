# vortex_sheet_methods
Vortex sheet method for modeling the aerodynamics of thin airfoil.

At its core, this code solves the Birkhoff-Rott equation, a singular integro-differential equation given by
$$
\frac{\partial \overline{z}}{\partial t} = \frac{1}{2\pi i}\int_C \frac{\gamma(s',t)}{z-z(s',t)}\,\mathrm{d}s'
$$
where $C$ is the collection of points along the vortex sheet.

This work is a refactoring of code I produced during the 2026 U(M) REU in Mathematics.
