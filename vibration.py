import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("Dynamic Response of a Damped Harmonic Oscillator")
    st.write("""
    This application simulates the dynamic response of a damped harmonic oscillator under external forces.
    Define the system parameters and select the type of external force to visualize the displacement over time.
    """)

    st.sidebar.header("System Parameters")

    # User Inputs
    K = st.sidebar.number_input("Stiffness (K) [N/m]", min_value=1e-3, value=2.287e5, format="%.3e")
    M = st.sidebar.number_input("Mass (M) [kg]", min_value=1e-3, value=275e3, format="%.3e")
    zeta = st.sidebar.slider("Damping Ratio (ζ)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    x0 = st.sidebar.number_input("Initial Displacement (x₀) [m]", value=1.0)
    vo = st.sidebar.number_input("Initial Velocity (v₀) [m/s]", value=0.0)

    tf = st.sidebar.number_input("Final Time (tf) [s]", min_value=0.1, value=50.0, step=0.1)
    dt = st.sidebar.number_input("Time Step (dt) [s]", min_value=1e-4, value=0.1, step=0.01)

    st.sidebar.header("External Force")

    force_type = st.sidebar.selectbox("Select Force Type", options=["Harmonic", "Linear"])

    if force_type == "Harmonic":
        Po = st.sidebar.number_input("Force Amplitude (P₀) [N]", value=0.0)
        w = st.sidebar.number_input("Force Frequency (ω) [rad/s]", value=0.4)
    elif force_type == "Linear":
        slope = st.sidebar.number_input("Force Slope [N/s]", value=0.0)

    # Calculate Derived Parameters
    C = 2 * zeta * np.sqrt(K * M)
    omega = np.sqrt(K / M)

    # Time Vector
    n = int(tf / dt)
    tempo = np.linspace(0, tf, n + 1)

    # External Force Vector
    if force_type == "Harmonic":
        P = Po * np.sin(w * tempo)
    elif force_type == "Linear":
        P = slope * tempo

    # Displacement Vector Initialization
    x = np.zeros(n + 1)
    x[0] = x0

    # Stability Check for Central Difference Method
    critical_dt = 2 / omega
    if dt >= critical_dt:
        st.warning(f"The chosen time step dt = {dt} s may be too large for stability. Consider dt < {critical_dt:.4f} s.")

    # Dynamic Response Calculation with Damping
    # Initial Acceleration
    a0 = (P[0] - C * vo - K * x[0]) / M

    # Calculate x[1] using Taylor Series Expansion
    x[1] = x[0] + dt * vo + (dt**2 / 2) * a0

    # Precompute denominator for efficiency
    denominator = 1 + (C * dt) / (2 * M)
    omega_sq_dt2_over2 = (omega**2 * dt**2) / 2

    for i in range(2, n + 1):
        P_prev = P[i - 1]
        numerator = (2 * x[i - 1] * (1 - omega_sq_dt2_over2)
                     - x[i - 2]
                     - (C * dt / M) * (x[i - 1] - x[i - 2])
                     + (P_prev * dt**2) / M)
        x[i] = numerator / denominator

    # Plotting the Dynamic Response
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tempo, x, 'b-', linewidth=2, label='Displacement x(t)')
    ax.set_title('Dynamic Response of Damped Harmonic Oscillator', fontsize=16)
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Displacement [m]', fontsize=14)
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

if __name__ == "__main__":
    main()
