#!/usr/bin/env python3

# ---------------------------------------------------
# Create IC conditions for an explosion in the lower
# center of the box.
# ---------------------------------------------------


import numpy as np
from mesh_hydro_utils import write_ic


nx = 200

rho0 = 1.0
v0 = 0.0
p0 = 1e-5
e1 = 1.0


def p(rho, e, gamma):
    return rho * e * (gamma - 1)


pblast = p(rho0, e1, 5.0 / 3.0)

rho = np.ones((nx, nx)) * rho0
v = np.ones((nx, nx, 2)) * v0
p = np.ones((nx, nx)) * p0


dx = 1.0 / nx
print(1.0 / dx, pblast)

if nx % 2 == 0:
    c = int(nx / 2) - 1
    p[c, c] = 1.0 / dx
    p[c + 1, c] = 1.0 / dx
    p[c, c + 1] = 1.0 / dx
    p[c + 1, c + 1] = 1.0 / dx
else:
    c = int(nx / 2)
    p[c, c] = 1.0 / dx

print("c=", c, "nx", nx)

write_ic("sedov.dat", 2, rho, v, p)
