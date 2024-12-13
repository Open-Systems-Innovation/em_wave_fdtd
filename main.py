"""
2D FDTD electromagnetic wave simulation
Simulates Transverse Magnetic (TM) mode field components with
PML boundary conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

# physics parameters
c = 3e8                     # [m/s] speed of light
mu = np.pi*4e-7             # [H/m] vacuum permeability
epsilon = 1 / (mu * c**2)   # [F/m]

# wave source parameters
frequency = 24000  # 5e9 [Hz]
amplitude = 1.4*1450            # [?]
wavelength = c / frequency  # [m]
ps1 = 50  # point source location

# interior grid parameters (not PML)
domain_nx = 200                     # number of points in x direction
domain_ny = 200                     # number of points in the y direction
size_x = 0.5                 # total domin size in x direction [m]
size_y = 0.5                 # total domin size in x direction [m]
dx = size_x / domain_nx             # grid spacing in x direction [m]
dy = size_y / domain_ny             # grid spacing in y direction [m]

# PML boundary parameters
pml_thickness = 5
R_0 = 1e-8  # PML refelction Coefficient
pml_order = 2

# overall grid parameters (interior + pml)
nx = domain_nx + 2*pml_thickness
ny = domain_ny + 2*pml_thickness
nxm1 = nx - 1
nym1 = ny - 1
nxp1 = nx + 1
nyp1 = ny + 1

# boundaries of the non-pml region
pis = pml_thickness       # PML i index start
pie = nx - pml_thickness  # PML i index end
pjs = pml_thickness       # PML j index start
pje = ny - pml_thickness  # PML j index end

# time stepping parameters
courant_factor = 0.9
dt = courant_factor * 1/(c*np.sqrt(1/(dx**2) + 1/(dy**2)))   # based on CFL
t_final = dt*350                  # final time
t = 0                        # starting time

# log information to the consol
print("\n PROBLEM INFORMATION ")
print(" -------------------")
print(f"number of grid points in x direction = {nx}")
print(f"number of grid points in y direction = {ny}")
print(f"domain width  = {size_x}[m]")
print(f"domain height = {size_y}[m]")
print(f"dx = {dx}[m] \ndy = {dy}[m] \ndt = {dt}[s]")
print(f"wavelength = {wavelength}[m] \nfrequency = {frequency}[Hz]")
print(f"epsilon = {epsilon}[?] \nmu = {mu}[?]")
print(f"c = {c}[m/s]\n")

# Initialize Fields
Hx = np.zeros((nxp1, ny))
Hy = np.zeros((nx, nyp1))
Ez = np.zeros((nxp1, nyp1))

# Initialize relative material parameters
eps_rz = np.ones((nxp1, nyp1))
mu_rx = np.ones((nxp1, ny))
mu_ry = np.ones((nx, nyp1))
sigma_ez = np.zeros((nxp1, nyp1))
sigma_mx = np.zeros((nxp1, ny))
sigma_my = np.zeros((nx, nyp1))

# Initialize Sources


# Create update coefficients
# for updating Ez
Ceze = (2 * eps_rz * epsilon - dt * sigma_ez) / (2 * eps_rz * epsilon + dt * sigma_ez)
Cezhy = (2 * dt / dx) / (2 * eps_rz * epsilon + dt * sigma_ez)
Cezhx = -(2 * dt / dy) / (2 * eps_rz * epsilon + dt * sigma_ez)
# for updating Hx
Chxh = (2 * mu_rx * mu - dt * sigma_mx) / (2 * mu_rx * mu + dt * sigma_mx)
Chxez = -(2 * dt / dy) / (2 * mu_rx * mu + dt * sigma_mx)
# for updating Hy
Chyh = (2 * mu_ry * mu - dt * sigma_my) / (2 * mu_ry * mu + dt * sigma_my)
Chyez = (2 * dt / dx) / (2 * mu_ry * mu + dt * sigma_my)

# Initialize PML fields
Ezx_xn = np.zeros((pml_thickness, nym1))
Ezy_xn = np.zeros((pml_thickness, nym1 - 2*pml_thickness))
Ezx_xp = np.zeros((pml_thickness, nym1))
Ezy_xp = np.zeros((pml_thickness, nym1 - 2*pml_thickness))
Ezx_yn = np.zeros((nxm1 - 2*pml_thickness, pml_thickness))
Ezy_yn = np.zeros((nxm1, pml_thickness))
Ezx_yp = np.zeros((nxm1 - 2*pml_thickness, pml_thickness))
Ezy_yp = np.zeros((nxm1, pml_thickness))

# Initialize PML material parameters
print('Initializing PML coefficients')

# Assume pml_thickness is the same for all PML layers
sigma_pex_xn = np.zeros((pml_thickness, nym1))
sigma_pmx_xn = np.zeros((pml_thickness, nym1))

sigma_max = -(pml_order + 1) * epsilon * c * np.log(R_0) / (2 * dx * pml_thickness)
rho_e = (np.arange(pml_thickness, 0, -1) - 0.75) / pml_thickness
rho_m = (np.arange(pml_thickness, 0, -1) - 0.25) / pml_thickness
for i in range(pml_thickness):
    sigma_pex_xn[i, :] = sigma_max * rho_e[i] ** pml_order
    sigma_pmx_xn[i, :] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order

# Coefficients updating Hy
Chyh_xn = (2 * mu - dt * sigma_pmx_xn) / (2 * mu + dt * sigma_pmx_xn)
Chyez_xn = (2 * dt / dx) / (2 * mu + dt * sigma_pmx_xn)

# Coefficients updating Ezx
Cezxe_xn = (2 * epsilon - dt * sigma_pex_xn) / (2 * epsilon + dt * sigma_pex_xn)
Cezxhy_xn = (2 * dt / dx) / (2 * epsilon + dt * sigma_pex_xn)

# Coefficients updating Ezy
Cezye_xn = 1
Cezyhx_xn = -dt / (dy * epsilon)

# For the xp region (PML layer in the positive x direction)
sigma_pex_xp = np.zeros((pml_thickness, nym1))
sigma_pmx_xp = np.zeros((pml_thickness, nym1))

sigma_max = -(pml_order + 1) * epsilon * c * np.log(R_0) / (2 * dx * pml_thickness)
rho_e = (np.arange(1, pml_thickness + 1) - 0.75) / pml_thickness
rho_m = (np.arange(1, pml_thickness + 1) - 0.25) / pml_thickness
for i in range(pml_thickness):
    sigma_pex_xp[i, :] = sigma_max * rho_e[i] ** pml_order
    sigma_pmx_xp[i, :] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order

# Coefficients updating Hy
Chyh_xp = (2 * mu - dt * sigma_pmx_xp) / (2 * mu + dt * sigma_pmx_xp)
Chyez_xp = (2 * dt / dx) / (2 * mu + dt * sigma_pmx_xp)

# Coefficients updating Ezx
Cezxe_xp = (2 * epsilon - dt * sigma_pex_xp) / (2 * epsilon + dt * sigma_pex_xp)
Cezxhy_xp = (2 * dt / dx) / (2 * epsilon + dt * sigma_pex_xp)

# Coefficients updating Ezy
Cezye_xp = 1
Cezyhx_xp = -dt / (dy * epsilon)

# For the yn region (PML layer in the negative y direction)
sigma_pey_yn = np.zeros((nxm1, pml_thickness))
sigma_pmy_yn = np.zeros((nxm1, pml_thickness))

sigma_max = -(pml_order + 1) * epsilon * c * np.log(R_0) / (2 * dy * pml_thickness)
rho_e = (np.arange(pml_thickness, 0, -1) - 0.75) / pml_thickness
rho_m = (np.arange(pml_thickness, 0, -1) - 0.25) / pml_thickness
for i in range(pml_thickness):
    sigma_pey_yn[:, i] = sigma_max * rho_e[i] ** pml_order
    sigma_pmy_yn[:, i] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order

# Coefficients updating Hx
Chxh_yn = (2 * mu - dt * sigma_pmy_yn) / (2 * mu + dt * sigma_pmy_yn)
Chxez_yn = -(2 * dt / dy) / (2 * mu + dt * sigma_pmy_yn)

# Coefficients updating Ezx
Cezxe_yn = 1
Cezxhy_yn = dt / (dx * epsilon)

# Coefficients updating Ezy
Cezye_yn = (2 * epsilon - dt * sigma_pey_yn) / (2 * epsilon + dt * sigma_pey_yn)
Cezyhx_yn = -(2 * dt / dy) / (2 * epsilon + dt * sigma_pey_yn)

# For the yp region (PML layer in the positive y direction)
sigma_pey_yp = np.zeros((nxm1, pml_thickness))
sigma_pmy_yp = np.zeros((nxm1, pml_thickness))

sigma_max = -(pml_order + 1) * epsilon * c * np.log(R_0) / (2 * dy * pml_thickness)
rho_e = (np.arange(1, pml_thickness + 1) - 0.75) / pml_thickness
rho_m = (np.arange(1, pml_thickness + 1) - 0.25) / pml_thickness
for i in range(pml_thickness):
    sigma_pey_yp[:, i] = sigma_max * rho_e[i] ** pml_order
    sigma_pmy_yp[:, i] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order

# Coefficients updating Hx
Chxh_yp = (2 * mu - dt * sigma_pmy_yp) / (2 * mu + dt * sigma_pmy_yp)
Chxez_yp = -(2 * dt / dy) / (2 * mu + dt * sigma_pmy_yp)

# Coefficients updating Ezx
Cezxe_yp = 1
Cezxhy_yp = dt / (dx * epsilon)

# Coefficients updating Ezy
Cezye_yp = (2 * epsilon - dt * sigma_pey_yp) / (2 * epsilon + dt * sigma_pey_yp)
Cezyhx_yp = -(2 * dt / dy) / (2 * epsilon + dt * sigma_pey_yp)

# Create "figures" directory if it doesn't exist
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)


def plot_solution(cmap="viridis", output_path="figures/plot.png"):
    fig, axs = plt.subplots(1, 3)
    datasets = [Ez, Hx, Hy]

    amplitude = 1

    matricies = []
    for ax, data in zip(axs.flat, datasets):
        matricies.append(ax.imshow(data, vmin=-amplitude, vmax=amplitude))

    fig.colorbar(matricies[0], ax=axs, orientation='horizontal', fraction=.1)
    plt.savefig(output_path, dpi=300)
    plt.close()


timestep = 0
while t < t_final:

    # update_Hx()
    Hx[:, pjs:pje-1] = Chxh[:, pjs:pje-1] * Hx[:, pjs:pje-1] + \
        Chxez[:, pjs:pje-1] * (Ez[:, pjs+1:pje] - Ez[:, pjs:pje-1])
    # update_Hy()
    Hy[pis:pie-1, :] = Chyh[pis:pie-1, :] * Hy[pis:pie-1, :] + \
        Chyez[pis:pie-1, :] * (Ez[pis+1:pie, :] - Ez[pis:pie-1, :])
    # update Hx and Hy PML layers
    # for xn
    Hy[0:pis-1, 1:ny] = Chyh_xn * Hy[0:pis-1, 1:ny] + Chyez_xn * \
        (Ez[1:pis, 1:ny] - Ez[0:pis-1, 1:ny])
    # for xp
    Hy[pie:nx, 1:ny] = Chyh_xp * Hy[pie:nx, 1:ny] + Chyez_xp * \
        (Ez[pie+1:nxp1, 1:ny] - Ez[pie:nx, 1:ny])
    # for yn
    Hx[1:nx, 0:pjs-1] = Chxh_yn * Hx[1:nx, 0:pjs-1] + Chxez_yn * \
        (Ez[1:nx, 1:pjs] - Ez[1:nx, 0:pjs-1])
    # for yp
    Hx[1:nx, pje:ny] = Chxh_yp * Hx[1:nx, pje:ny] + Chxez_yp * \
        (Ez[1:nx, pje+1:nyp1] - Ez[1:nx, pje:ny])

    # update_Ez()
    Ez[pis:pie-1, pjs:pje-1] = (
        Ceze[pis:pie-1, pjs:pje-1] * Ez[pis:pie-1, pjs:pje-1] +
        Cezhy[pis:pie-1, pjs:pje-1] *
        (Hy[pis:pie-1, pjs:pje-1] - Hy[pis-1:pie-2, pjs:pje-1]) +
        Cezhx[pis:pie-1, pjs:pje-1] *
        (Hx[pis:pie-1, pjs:pje-1] - Hx[pis:pie-1, pjs-1:pje-2])
    )

    # update_impressed_J()
    Cezj = -(2*dt) / (2 * eps_rz[ps1, ps1] * epsilon + dt * sigma_ez[ps1, ps1])
    Ez[ps1, ps1] = Ez[ps1, ps1] + Cezj * np.sin(2 * np.pi * frequency * t)
    #update_Ez_pml()
    # For xn PML region
    Ezx_xn = Cezxe_xn * Ezx_xn + Cezxhy_xn * (Hy[1:pis, 1:ny] - Hy[0:pis-1, 1:ny])
    Ezy_xn = Cezye_xn * Ezy_xn + Cezyhx_xn * (Hx[1:pis, pjs+1:pje-1] - Hx[1:pis, pjs:pje-2])

    # For xp PML region
    Ezx_xp = Cezxe_xp * Ezx_xp + Cezxhy_xp * (Hy[pie:nx, 1:ny] - Hy[pie-1:nx-1, 1:ny])
    Ezy_xp = Cezye_xp * Ezy_xp + Cezyhx_xp * (Hx[pie:nx, pjs+1:pje-1] - Hx[pie:nx, pjs:pje-2])

    # For yn PML region
    Ezx_yn = Cezxe_yn * Ezx_yn + Cezxhy_yn * (Hy[pis+1:pie-1, 1:pis] - Hy[pis:pie-2, 1:pjs])
    Ezy_yn = Cezye_yn * Ezy_yn + Cezyhx_yn * (Hx[1:nx, 1:pjs] - Hx[1:nx, 0:pjs-1])

    # For yp PML region
    Ezx_yp = Cezxe_yp * Ezx_yp + Cezxhy_yp * (Hy[pis+1:pie-1, pje:ny] - Hy[pis:pie-2, pje:ny])
    Ezy_yp = Cezye_yp * Ezy_yp + Cezyhx_yp * (Hx[1:nx, pje:ny] - Hx[1:nx, pje-1:ny-1])

    # Update the Ez field at the corresponding regions
    Ez[1:pis, 1:pjs] = Ezx_xn[:, 1:pjs-1] + Ezy_yn[1:pis-1, :]
    Ez[1:pis, pje:ny] = Ezx_xn[:, pje-1:nym1] + Ezy_yp[1:pis-1, :]
    Ez[pie:nx, pje:ny] = Ezx_xp[:, pje-1:nym1] + Ezy_yp[pie-1:nxm1, :]
    Ez[pie:nx, 1:pjs] = Ezx_xp[:, 1:pjs-1] + Ezy_yn[pie-1:nxm1, :]
    Ez[pis+1:pie-1, 1:pjs] = Ezx_yn + Ezy_yn[pis:pie-2, :]
    Ez[pis+1:pie-1, pje:ny] = Ezx_yp + Ezy_yp[pis:pie-2, :]
    Ez[1:pis, pjs+1:pje-1] = Ezx_xn[:, pjs:pje-2] + Ezy_xn
    Ez[pie:nx, pjs+1:pje-1] = Ezx_xp[:, pjs:pje-2] + Ezy_xp
    
    # plot solution
    if timestep % 10 == 0:
        output_path = f"{output_dir}/timestep_{timestep:04d}.png"
        plot_solution(output_path=output_path)

    # increment timestep
    timestep += 1
