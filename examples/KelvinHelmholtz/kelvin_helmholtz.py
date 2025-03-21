#!/usr/bin/env python3

# -----------------------------------------------------------------
# Create Kelvin-Helmholtz instability initial
# conditions following McNally et al. 2012
# (https://ui.adsabs.harvard.edu/abs/2012ApJS..201...18M/abstract)
# -----------------------------------------------------------------


import numpy as np
from mesh_hydro_utils import write_ic


nx = 256

dx = 1.0 / nx
dy = 0.025  # not grid related

rho1 = 1
rho2 = 2
delta_rho = 0.5 * (rho2 - rho1)
P = 5 / 2
v0 = 0.01


rho = np.empty((nx, nx), dtype=float)
u = np.empty((nx, nx, 2), dtype=float)
p = np.ones((nx, nx), dtype=float) * P


for i in range(nx):
    x = (i + 0.5) * dx
    u[i, :, 1] = v0 * np.sin(4 * np.pi * x)

    for j in range(nx):
        y = (j + 0.5) * dx
        if y < 0.25:
            rho[i, j] = rho2 - delta_rho * np.exp((y - 0.25) / dy)
            u[i, j, 0] = -0.5 + 0.5 * np.exp((y - 0.25) / dy)
        elif y < 0.5:
            rho[i, j] = rho1 + delta_rho * np.exp((0.25 - y) / dy)
            u[i, j, 0] = 0.5 - 0.5 * np.exp((0.25 - y) / dy)
        elif y < 0.75:
            rho[i, j] = rho1 + delta_rho * np.exp((y - 0.75) / dy)
            u[i, j, 0] = 0.5 - 0.5 * np.exp((y - 0.75) / dy)
        else:
            rho[i, j] = rho2 - delta_rho * np.exp((0.75 - y) / dy)
            u[i, j, 0] = -0.5 + 0.5 * np.exp((0.75 - y) / dy)


write_ic("kelvin_helmholtz.dat", 2, rho, u, p)
