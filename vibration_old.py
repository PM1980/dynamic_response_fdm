import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def calculate_response(K, M, zeta, x0, v0, tf, dt, force_type, force_param):
    omega = np.sqrt(K / M)
    C = 2 * zeta * np.sqrt(K * M)
    
    t = np.arange(0, tf + dt, dt)
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)
    
    x[0], v[0] = x0, v0
    
    for i in range(1, n):
        if force_type == "Harmonic":
            F = force_param * np.sin(omega * t[i])
        else:  # Linear
            F = force_param * t[i]
        
        a = (F - C * v[i-1] - K * x[i-1]) / M
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt
    
    return t, x

st.title("Damped Harmonic Oscillator Simulation")

st.sidebar.header("Parameters")
K = st.sidebar.number_input("Stiffness (N/m)", value=1000.0, min_value=0.1)
M = st.sidebar.number_input("Mass (kg)", value=1.0, min_value=0.1)
zeta = st.sidebar.slider("Damping Ratio", 0.0, 2.0, 0.1)
x0 = st.sidebar.number_input("Initial Displacement (m)", value=1.0)
v0 = st.sidebar.number_input("Initial Velocity (m/s)", value=0.0)
tf = st.sidebar.number_input("Simulation Time (s)", value=10.0, min_value=0.1)
dt = st.sidebar.number_input("Time Step (s)", value=0.1, min_value=0.01)

force_type = st.sidebar.selectbox("Force Type", ["Harmonic", "Linear"])
if force_type == "Harmonic":
    force_param = st.sidebar.number_input("Force Amplitude (N)", value=0.0)
else:
    force_param = st.sidebar.number_input("Force Slope (N/s)", value=0.0)

t, x = calculate_response(K, M, zeta, x0, v0, tf, dt, force_type, force_param)

fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_title("Oscillator Response")
st.pyplot(fig)
