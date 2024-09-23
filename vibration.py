import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def calculate_response_cds(K, M, zeta, x0, v0, tf, dt, force_type, force_param, omega_force=0.0):
    """
    Calculate the dynamic response of a damped harmonic oscillator using the Central Difference Scheme.

    Parameters:
    - K: Stiffness (N/m)
    - M: Mass (kg)
    - zeta: Damping ratio
    - x0: Initial displacement (m)
    - v0: Initial velocity (m/s)
    - tf: Final time (s)
    - dt: Time step (s)
    - force_type: "Harmonic" or "Linear"
    - force_param: Amplitude for Harmonic, Slope for Linear
    - omega_force: Frequency for Harmonic force (rad/s)

    Returns:
    - t: Time array
    - x: Displacement array
    - v: Velocity array
    - kinetic: Kinetic energy array
    - potential: Potential energy array
    - total_energy: Total energy array
    """
    omega_n = np.sqrt(K / M)  # Natural frequency (rad/s)
    C = 2 * zeta * np.sqrt(K * M)  # Damping coefficient (N·s/m)

    t = np.arange(0, tf + dt, dt)
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)
    kinetic = np.zeros(n)
    potential = np.zeros(n)
    total_energy = np.zeros(n)

    # Initialize displacement and velocity
    x[0] = x0
    v[0] = v0

    # Calculate initial acceleration
    if force_type == "Harmonic":
        F0 = force_param * np.sin(omega_force * t[0])
    else:  # Linear
        F0 = force_param * t[0]
    a0 = (F0 - C * v0 - K * x0) / M

    # Use Taylor series to estimate x[1]
    x[1] = x0 + dt * v0 + 0.5 * dt**2 * a0

    # Stability Check
    critical_dt = 2 / omega_n
    if dt >= critical_dt:
        st.warning(f"The chosen time step dt = {dt} s may be too large for stability of the Central Difference Scheme. Consider dt < {critical_dt:.4f} s.")

    # Compute response using Central Difference Scheme
    for i in range(1, n-1):
        if force_type == "Harmonic":
            F = force_param * np.sin(omega_force * t[i])
        else:  # Linear
            F = force_param * t[i]

        # Central Difference Formula
        x_next = (2 * x[i] - x[i-1] + dt**2 * (F - C * (x[i] - x[i-1]) / dt - K * x[i]) / M) / (1 + (C * dt) / (2 * M))
        x[i+1] = x_next

    # Calculate velocity using central difference approximation
    v = np.zeros(n)
    v[0] = v0
    for i in range(1, n-1):
        v[i] = (x[i+1] - x[i-1]) / (2 * dt)
    v[-1] = (x[-1] - x[-2]) / dt  # Forward difference for the last point

    # Calculate energies
    kinetic = 0.5 * M * v**2
    potential = 0.5 * K * x**2
    total_energy = kinetic + potential

    return t, x, v, kinetic, potential, total_energy

# Streamlit User Interface
st.set_page_config(page_title="Damped Harmonic Oscillator Simulation", layout="wide")
st.title("Damped Harmonic Oscillator Simulation (Central Difference Scheme)")

# Sidebar for Inputs
st.sidebar.header("System Parameters")
K = st.sidebar.number_input("Stiffness (N/m)", value=1000.0, min_value=0.1, step=100.0)
M = st.sidebar.number_input("Mass (kg)", value=1.0, min_value=0.1, step=0.1)
zeta = st.sidebar.slider("Damping Ratio (ζ)", 0.0, 2.0, 0.05, step=0.01)
x0 = st.sidebar.number_input("Initial Displacement (m)", value=1.0, step=0.1)
v0 = st.sidebar.number_input("Initial Velocity (m/s)", value=0.0, step=0.1)
tf = st.sidebar.number_input("Simulation Time (s)", value=10.0, min_value=0.1, step=0.1)
dt = st.sidebar.number_input("Time Step (s)", value=0.01, min_value=0.001, step=0.001)

st.sidebar.header("External Force")
force_type = st.sidebar.selectbox("Force Type", ["Harmonic", "Linear"])
if force_type == "Harmonic":
    force_amplitude = st.sidebar.number_input("Force Amplitude (N)", value=0.0, step=10.0)
    omega_force = st.sidebar.number_input("Force Frequency (rad/s)", value=1.0, min_value=0.0, step=0.1)
    force_param = force_amplitude
else:
    force_slope = st.sidebar.number_input("Force Slope (N/s)", value=0.0, step=10.0)
    force_param = force_slope
