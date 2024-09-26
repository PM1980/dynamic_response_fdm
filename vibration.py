import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def calculate_response(M, K, zeta, load_type, load_params, tf, dt):
    # Calculate derived parameters
    C = 2 * zeta * np.sqrt(K * M)
    omega = np.sqrt(K/M)
    
    # Time vector
    t = np.arange(0, tf + dt, dt)
    n = len(t)
    
    # Initialize arrays
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    
    # Load vector
    if load_type == "Harmonic":
        Po, w = load_params
        P = Po * np.sin(w * t)
    elif load_type == "Linear":
        P0, P1 = load_params
        P = np.linspace(P0, P1, n)
    elif load_type == "Pulse":
        Po, duration = load_params
        P = np.where(t <= duration, Po, 0)
    
    # Initial conditions
    x[0] = 0
    v[0] = 0
    a[0] = (P[0] - C*v[0] - K*x[0]) / M
    
    # First step
    x[1] = 0.5 * (dt**2 * a[0] + 2*v[0]*dt)
    
    # Time-stepping loop
    for i in range(2, n):
        x[i] = (2*(dt**2)*(P[i-1]-K*x[i-1]) + C*dt*x[i-2] + 4*M*x[i-1] - 2*M*x[i-2]) / (C*dt + 2*M)
    
    # Calculate velocity and acceleration
    v[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    a[1:-1] = (x[2:] - 2*x[1:-1] + x[:-2]) / (dt**2)
    
    return t, x, v, a, P

st.title("SDOF Dynamic Response Simulator")

# Sidebar for inputs
st.sidebar.header("System Parameters")
M = st.sidebar.number_input("Mass (kg)", value=275e3, format="%e")
K = st.sidebar.number_input("Stiffness (N/m)", value=2.287e5, format="%e")
zeta = st.sidebar.number_input("Damping Ratio", value=0.05, format="%f")

st.sidebar.header("Simulation Parameters")
tf = st.sidebar.number_input("Simulation Time (s)", value=200)
dt = st.sidebar.number_input("Time Step (s)", value=0.1)

st.sidebar.header("Load Parameters")
load_type = st.sidebar.selectbox("Load Type", ["Harmonic", "Linear", "Pulse"])

if load_type == "Harmonic":
    Po = st.sidebar.number_input("Load Amplitude (N)", value=1e5)
    w = st.sidebar.number_input("Load Frequency (rad/s)", value=0.1)
    load_params = (Po, w)
elif load_type == "Linear":
    P0 = st.sidebar.number_input("Initial Load (N)", value=0)
    P1 = st.sidebar.number_input("Final Load (N)", value=1e5)
    load_params = (P0, P1)
elif load_type == "Pulse":
    Po = st.sidebar.number_input("Pulse Amplitude (N)", value=1e5)
    duration = st.sidebar.number_input("Pulse Duration (s)", value=10.0)
    load_params = (Po, duration)

# Calculate response
t, x, v, a, P = calculate_response(M, K, zeta, load_type, load_params, tf, dt)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 20))
axs[0].plot(t, P)
axs[0].set_ylabel("Load (N)")
axs[0].set_title("Applied Load")
axs[0].grid(True)

axs[1].plot(t, x)
axs[1].set_ylabel("Displacement (m)")
axs[1].set_title("Displacement Response")
axs[1].grid(True)

axs[2].plot(t, v)
axs[2].set_ylabel("Velocity (m/s)")
axs[2].set_title("Velocity Response")
axs[2].grid(True)

axs[3].plot(t, a)
axs[3].set_ylabel("Acceleration (m/s²)")
axs[3].set_title("Acceleration Response")
axs[3].grid(True)

axs[3].set_xlabel("Time (s)")

plt.tight_layout()
st.pyplot(fig)
