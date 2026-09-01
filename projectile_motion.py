# =============================================================================
# PROJECTILE MOTION SIMULATOR — Streamlit App
# -----------------------------------------------------------------------------
# A teaching tool for 12th-standard Physics students showing how Python can
# visualize motion under gravity (projectile motion) interactively.
#
# HOW TO RUN LOCALLY:
#   1. pip install -r requirements.txt
#   2. streamlit run projectile_motion.py
#
# HOW TO DEPLOY FOR FREE (Streamlit Community Cloud):
#   1. Push this file to the SAME GitHub repo as your other Streamlit app(s)
#      (requirements.txt can be shared/reused — no changes needed).
#   2. Go to share.streamlit.io -> "New app" -> pick this repo/branch and
#      set "Main file path" to projectile_motion.py.
#   3. Deploy. You'll get its own separate public URL, even though it lives
#      in the same repo as your Boussinesq app.
#
#   TIP: If you'd rather have ONE app with a menu instead of two separate
#   links, rename this file to pages/1_Projectile_Motion.py and your other
#   app to pages/2_Boussinesq_Stress.py inside the same repo (with a small
#   Home.py as the entry point). Streamlit auto-builds a sidebar menu from
#   the "pages/" folder. Happy to set that up if you want it.
#
# THEORY:
#   For a projectile launched with initial speed u at angle theta (from the
#   horizontal), ignoring air resistance:
#
#       x(t) = u * cos(theta) * t
#       y(t) = u * sin(theta) * t  -  0.5 * g * t^2
#
#       Time of flight   T = 2 * u * sin(theta) / g
#       Max height       H = (u * sin(theta))^2 / (2 * g)
#       Range            R = u^2 * sin(2*theta) / g
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Page setup --------------------------------------------------------------
st.set_page_config(page_title="Projectile Motion Simulator", layout="wide")


# --- Core physics functions ---------------------------------------------------
def projectile_kinematics(u, theta_deg, g):
    """
    u          : initial speed (m/s)
    theta_deg  : launch angle from horizontal (degrees)
    g          : acceleration due to gravity (m/s^2)

    Returns time-of-flight T, max height H, range R, and (t, x, y) arrays
    for plotting the trajectory.
    """
    theta = np.radians(theta_deg)
    T = 2 * u * np.sin(theta) / g
    H = (u * np.sin(theta)) ** 2 / (2 * g)
    R = (u ** 2) * np.sin(2 * theta) / g

    t = np.linspace(0, T, 300)
    x = u * np.cos(theta) * t
    y = u * np.sin(theta) * t - 0.5 * g * t ** 2
    y = np.clip(y, 0, None)  # guard against tiny negative values from rounding

    return T, H, R, t, x, y


# --- Header / theory -----------------------------------------------------------
st.title("Projectile Motion Simulator")
st.latex(r"x(t) = u\cos\theta \cdot t \qquad\qquad y(t) = u\sin\theta \cdot t - \tfrac{1}{2}g t^2")
st.markdown(
    "Launch a projectile with initial speed **u** at angle **θ** from the "
    "horizontal, ignoring air resistance. Adjust the controls in the sidebar "
    "and watch the trajectory, range, and max height update."
)

# --- Sidebar controls ----------------------------------------------------------
st.sidebar.header("Inputs")
u = st.sidebar.slider("Initial speed, u (m/s)", min_value=1.0, max_value=50.0,
                       value=20.0, step=0.5)
theta_deg = st.sidebar.slider("Launch angle, θ (degrees)", min_value=1, max_value=89,
                               value=45, step=1)
g = st.sidebar.slider("Acceleration due to gravity, g (m/s²)", min_value=1.0,
                       max_value=20.0, value=9.8, step=0.1)

# --- Compute -------------------------------------------------------------------
T, H, R, t, x, y = projectile_kinematics(u, theta_deg, g)

# --- Headline results ------------------------------------------------------
col_a, col_b, col_c = st.columns(3)
col_a.metric("Time of Flight, T", f"{T:.2f} s")
col_b.metric("Max Height, H", f"{H:.2f} m")
col_c.metric("Range, R", f"{R:.2f} m")

st.divider()

# --- Trajectory plot ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, color='royalblue', linewidth=2, label='Trajectory')
ax.fill_between(x, y, color='royalblue', alpha=0.08)

# Mark the peak (max height) with a star
t_peak = T / 2
x_peak = u * np.cos(np.radians(theta_deg)) * t_peak
ax.plot(x_peak, H, marker='*', markersize=22, color='gold',
        markeredgecolor='black', markeredgewidth=1.2, linestyle='None',
        zorder=5, label='Max height point')

# Mark launch and landing points
ax.plot(0, 0, marker='o', markersize=10, color='seagreen', label='Launch point')
ax.plot(R, 0, marker='o', markersize=10, color='firebrick', label='Landing point')

ax.axhline(0, color='saddlebrown', linewidth=3)
ax.set_xlabel('Horizontal distance, x (m)')
ax.set_ylabel('Height, y (m)')
ax.set_title(f'Trajectory for u = {u:.1f} m/s, θ = {theta_deg}°, g = {g:.1f} m/s²')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(left=-0.05 * max(R, 1))
ax.set_ylim(bottom=-0.05 * max(H, 1))
ax.legend(loc='upper right', fontsize=9)

st.pyplot(fig)

st.caption(
    "This model ignores air resistance. Try θ = 45° to see the angle that "
    "gives the maximum range for a fixed speed — a classic result worth "
    "verifying yourself from the range formula."
)
