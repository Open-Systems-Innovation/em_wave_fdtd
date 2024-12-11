"""
2D FDTD electromagnetic wave simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

# problem parameters
nx = 100                     # number of points in x direction
ny = 100                     # number of points in the y direction
c = 3e8                      # speed of light
size_x = 0.5               # total domin size in x direction [m]
size_y = 0.5               # total domin size in x direction [m]
dx = size_x / nx             # grid spacing in x direction [m]
dy = size_y / ny             # grid spacing in y direction [m]
# time stepping parameters
dt = 0.9 * 1/(c*np.sqrt(1/(dx**2) + 1/(dy**2)))   # time step based on CFL condtion
t_final = 1                  # final time
t = 0                        # starting time 

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
epsilon_x = np.ones((nx, ny))*8.854e-12  # permitivity [F/m]
epsilon_y = np.ones((nx, ny))*8.854e-12  # permitivity [F/m]
mu_z = np.ones((nx, ny))*np.pi*4e-7      # permeability [H/m]
sigma_ex = np.zeros((nx, ny))   # electric conductivity
sigma_ey = np.zeros((nx, ny))   # electric conductivity
sigma_mx = np.zeros((nx, ny))   # magnetic conductivity
sigma_my = np.zeros((nx, ny))   # magnetic conductivity
sigma_mz = np.zeros((nx, ny))   # magnetic conductivity

# [TODO] Set up initial conditions
#C_exe = (2 * epsilon_x - dt * sigma_ex) / (2 * epsilon_x + dt * sigma_ex)

# Coefficient update helper functions
def C_exe(i,j):
    numerator = (2 * epsilon_x[i,j] - dt * sigma_ex[i,j])
    denominator = (2 * epsilon_x[i,j] + dt * sigma_ex[i,j])
    return numerator / denominator

def C_exhz(i,j):
    numerator = (2 * dt)
    denominator = (2 * epsilon_x[i,j] + dt * sigma_ex[i,j])*dy
    return numerator / denominator                                      

def C_exj(i,j):
    numerator = (2 * dt)
    denominator = (2 * epsilon_x[i,j] + dt * sigma_ex[i,j])
    return numerator / denominator                                      

def C_eye(i,j):
    numerator = (2 * epsilon_y[i,j] - dt * sigma_ey[i,j])
    denominator = (2 * epsilon_y[i,j] + dt * sigma_ey[i,j])
    return numerator / denominator

def C_eyhz(i,j):
    numerator = -(2 * dt)
    denominator = (2 * epsilon_y[i,j] + dt * sigma_ey[i,j])*dx
    return numerator / denominator                                      

def C_eyj(i,j):
    numerator = -(2 * dt)
    denominator = (2 * epsilon_y[i,j] + dt * sigma_ey[i,j])
    return numerator / denominator                                      

def C_hzh(i,j):
    numerator = (2 * mu_z[i,j] - dt * sigma_mz[i,j])
    denominator = (2 * mu_z[i,j] + dt * sigma_mz[i,j])
    return numerator / denominator

def C_hzex(i,j):
    numerator = (2 * dt)
    denominator = (2 * mu_z[i,j] + dt * sigma_mz[i,j])*dy
    return numerator / denominator                                      

def C_hzey(i,j):
    numerator = -(2 * dt)
    denominator = (2 * mu_z[i,j] + dt * sigma_mz[i,j])*dx
    return numerator / denominator                                      

def C_hzm(i,j):
    numerator = -(2 * dt)
    denominator = (2 * mu_z[i,j] + dt * sigma_mz[i,j])
    return numerator / denominator                                      

# Update functions
def update_Ex(i, j):
    Ex_new[i,j] = C_exe(i,j) * Ex_old[i,j] + C_exhz(i,j) * \
        (Hz_new[i,j] - Hz_new[i, j - 1]) + \
        C_exj(i,j) * J_ix_new[i,j]
    return

def update_Ey(i, j):
    Ey_new[i,j] = C_eye(i,j) * Ey_old[i,j] + C_eyhz(i,j) * \
        (Hz_new[i,j] - Hz_new[i - 1, j]) + \
        C_eyj(i,j) * J_iy_new[i,j]
    return

def update_Hz(i,j):
    Hz_new[i,j] = C_hzh(i,j) * Hz_old[i,j] + C_hzex(i,j) * \
        (Ex_old[i, j + 1] - Ex_old[i, j]) + C_hzey(i,j) * \
        (Ey_old[i + 1, j] - Ey_old[i,j]) + C_hzm(i,j) * M_iz[i,j]


# TM (transverse electric) case

# Create "figures" directory if it doesn't exist
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
# Create a figure and subplots for plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].set_title("Ex")
axes[1].set_title("Ey")
axes[2].set_title("Hz")

# Initialize plots with empty data
im_Ex = axes[0].imshow(Ex_old, cmap="RdBu", origin="lower", extent=[0, size_x, 0, size_y], vmin=-1, vmax=1)
im_Ey = axes[1].imshow(Ey_old, cmap="RdBu", origin="lower", extent=[0, size_x, 0, size_y], vmin=-1, vmax=1)
im_Hz = axes[2].imshow(Hz_old, cmap="RdBu", origin="lower", extent=[0, size_x, 0, size_y], vmin=-1, vmax=1)

plt.tight_layout()
timestep = 0
while t < t_final:
    # make simple source term
    Ex_old[50,50] = np.sin(5e9*(timestep * dt))
    print(Ex_old[5,5])
    # update magnetic field at half timestep
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            update_Hz(i,j)
            
    # update electric field at full timestep
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            update_Ex(i,j)
            update_Ey(i,j)
            
    # apply boundary conditions
    # Set all boundaries of Ex to 0
    Ex_new[0, :] = 0       # Top boundary
    Ex_new[-1, :] = 0      # Bottom boundary
    Ex_new[:, 0] = 0       # Left boundary
    Ex_new[:, -1] = 0      # Right boundary

    # Set all boundaries of Ey to 0
    Ey_new[0, :] = 0       # Top boundary
    Ey_new[-1, :] = 0      # Bottom boundary
    Ey_new[:, 0] = 0       # Left boundary
    Ey_new[:, -1] = 0      # Right boundary

    # Set all boundaries of Hz to 0
    Hz_new[0, :] = 0       # Top boundary
    Hz_new[-1, :] = 0      # Bottom boundary
    Hz_new[:, 0] = 0       # Left boundary
    Hz_new[:, -1] = 0      # Right boundary

    # Update the images with new data
    im_Ex.set_data(Ex_old)
    im_Ey.set_data(Ey_old)
    im_Hz.set_data(Hz_old)

    # Swap Ex, Ey arrays for the next timestep
    Ex_old, Ex_new = Ex_new, Ex_old
    Ey_old, Ey_new = Ey_new, Ey_old

    # plot solution
    # Update plot titles with the current time
    axes[0].set_title(f"Ex at t = {t:.2e} s")
    axes[1].set_title(f"Ey at t = {t:.2e} s")
    axes[2].set_title(f"Hz at t = {t:.2e} s")

    # Save the current figure to the "figures" directory
    plt.savefig(f"{output_dir}/timestep_{timestep:04d}.png")
    # increment timestep
    timestep += 1
