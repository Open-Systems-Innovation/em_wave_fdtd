"""
2D FDTD electromagnetic wave simulation
Simulates TEz field components with
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

# interior grid parameters (not PML)
domain_nx = 200                     # number of points in x direction
domain_ny = 200                     # number of points in the y direction
size_x = 0.5                 # total domin size in x direction [m]
size_y = 0.5                 # total domin size in x direction [m]
dx = size_x / domain_nx             # grid spacing in x direction [m]
dy = size_y / domain_ny             # grid spacing in y direction [m]

# PML boundary parameters
pml_thickness = 5
pml_reflection_coefficient = 1e-8
pml_order = 2

# overall grid parameters (interior + pml)
nx = domain_nx + pml_thickness
ny = domain_ny + pml_thickness

# time stepping parameters
dt = 0.8 * 1/(c*np.sqrt(1/(dx**2) + 1/(dy**2)))   # based on CFL
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

# set up field grids
Ex_old = np.zeros((nx, ny))         # electric field
Ex_new = np.zeros((nx, ny))         # electric field
Ey_old = np.zeros((nx, ny))         # electric field
Ey_new = np.zeros((nx, ny))         # electric field
Hz_old = np.zeros((nx, ny))         # magnetic field
Hz_new = np.zeros((nx, ny))         # magnetic field
J_ix_old = np.zeros((nx, ny))         # current density
J_ix_new = np.zeros((nx, ny))         # current density
J_iy_old = np.zeros((nx, ny))         # current density
J_iy_new = np.zeros((nx, ny))         # current density
J_iz_old = np.zeros((nx, ny))         # current density
J_iz_new = np.zeros((nx, ny))         # current density
M_ix = np.zeros((nx, ny))         # magnetic current density
M_iy = np.zeros((nx, ny))         # magnetic current density
M_iz = np.zeros((nx, ny))         # magnetic current density
epsilon_x = np.ones((nx, ny))*epsilon  # permitivity [F/m]
epsilon_y = np.ones((nx, ny))*epsilon  # permitivity [F/m]
mu_z = np.ones((nx, ny))*mu      # permeability [H/m]
sigma_ex = np.zeros((nx, ny))   # electric conductivity
sigma_ey = np.zeros((nx, ny))   # electric conductivity
sigma_mx = np.zeros((nx, ny))   # magnetic conductivity
sigma_my = np.zeros((nx, ny))   # magnetic conductivity
sigma_mz = np.zeros((nx, ny))   # magnetic conductivity

# [TODO] Set up initial conditions

# Coefficient update helper functions
C_exe = (2 * epsilon_x - dt * sigma_ex) / (2 * epsilon_x + dt * sigma_ex)
C_exhz = (2 * dt) / (2 * epsilon_x + dt * sigma_ex)*dy
C_exj = -(2 * dt) / (2 * epsilon_x + dt * sigma_ex)
C_eye = (2 * epsilon_y - dt * sigma_ey) / (2 * epsilon_y + dt * sigma_ey)
C_eyhz = -(2 * dt) / (2 * epsilon_y + dt * sigma_ey)*dx
C_eyj = -(2 * dt) / (2 * epsilon_y + dt * sigma_ey)
C_hzh = (2 * mu_z - dt * sigma_mz) / (2 * mu_z + dt * sigma_mz)
C_hzex = (2 * dt) / (2 * mu_z + dt * sigma_mz)*dy
C_hzey = -(2 * dt) / (2 * mu_z + dt * sigma_mz)*dx
C_hzm = -(2 * dt) / (2 * mu_z + dt * sigma_mz)


# Update functions
def update_Ex(i, j):
    Ex_new[i, j] = C_exe[i, j] * Ex_old[i, j] + C_exhz[i, j] * \
        (Hz_new[i, j] - Hz_new[i, j - 1]) + \
        C_exj[i, j] * J_ix_new[i, j]
    return


def update_Ey(i, j):
    Ey_new[i, j] = C_eye[i, j] * Ey_old[i, j] + C_eyhz[i, j] * \
        (Hz_new[i, j] - Hz_new[i - 1, j]) + \
        C_eyj[i, j] * J_iy_new[i, j]
    return


def update_Hz(i, j):
    Hz_new[i, j] = C_hzh[i, j] * Hz_old[i, j] + C_hzex[i, j] * \
        (Ex_old[i, j + 1] - Ex_old[i, j]) + C_hzey[i, j] * \
        (Ey_old[i + 1, j] - Ey_old[i, j]) + C_hzm[i, j] * M_iz[i, j]


# Create "figures" directory if it doesn't exist
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)


def plot_solution(array, cmap="viridis", output_path="figures/plot.png"):
    fig, axs = plt.subplots(1, 3)
    datasets = [Ex_new, Ey_new, Hz_new]

    amplitude = 1
    
    matricies = []
    for ax, data in zip(axs.flat, datasets):
        matricies.append(ax.imshow(data, vmin=-amplitude, vmax=amplitude))

    fig.colorbar(matricies[0], ax=axs, orientation='horizontal', fraction=.1)
    plt.savefig(output_path, dpi=300)
    plt.close()


timestep = 0
while t < t_final:
    # make simple source term
    #Ex_old[50, 50] = amplitude*np.sin(frequency*(timestep * dt))
    #Ex_old[51, 50] = -amplitude*np.sin(frequency*(timestep * dt))
    Ex_old[50, 50] += amplitude * np.sin(2 * np.pi * frequency * timestep * dt)

    # update magnetic field at half timestep
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            update_Hz(i, j)

    # update electric field at full timestep
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            update_Ex(i, j)
            update_Ey(i, j)

    # apply boundary conditions
    # Set all boundaries of Ex to 0
    Ex_new[0, :] = 1       # Top boundary
    Ex_new[-1, :] = 1      # Bottom boundary
    Ex_new[:, 0] = 1       # Left boundary
    Ex_new[:, -1] = 1      # Right boundary

    # Set all boundaries of Ey to 0
    Ey_new[0, :] = 1       # Top boundary
    Ey_new[-1, :] = 1      # Bottom boundary
    Ey_new[:, 0] = 1       # Left boundary
    Ey_new[:, -1] = 1      # Right boundary

    # Set all boundaries of Hz to 0
    Hz_new[0, :] = 1       # Top boundary
    Hz_new[-1, :] = 1      # Bottom boundary
    Hz_new[:, 0] = 1       # Left boundary
    Hz_new[:, -1] = 1      # Right boundary

    # plot solution
    if timestep % 10 == 0:
        output_path = f"{output_dir}/timestep_{timestep:04d}.png"
        plot_solution(Ex_new, output_path=output_path)

    # update old arrays
    Ex_old = Ex_new
    Ey_old = Ey_new
    Hz_old = Hz_new

    # increment timestep
    timestep += 1
