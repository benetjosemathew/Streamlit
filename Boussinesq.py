# =============================================================================
# STRESS IN SOIL DUE TO A POINT LOAD (Boussinesq's Theory) — Streamlit App
# -----------------------------------------------------------------------------
# A teaching tool for First-Year Civil Engineering students showing how
# Python can be used to explore geotechnical concepts, deployed as a
# proper shareable web app (no Colab / notebook needed).
#
# HOW TO RUN LOCALLY:
#   1. pip install streamlit numpy matplotlib
#   2. streamlit run streamlit_app.py
#
# HOW TO DEPLOY FOR FREE (Streamlit Community Cloud):
#   1. Push this file + requirements.txt to a public GitHub repo.
#   2. Go to share.streamlit.io, sign in with GitHub.
#   3. Click "New app", pick the repo/branch/file (streamlit_app.py).
#   4. Deploy. You get a public URL students can open directly.
#
# THEORY (Boussinesq, 1885):
#   For a point load P applied on the surface of a semi-infinite, homogeneous,
#   isotropic, elastic soil mass, the vertical stress increase at a point
#   located at depth z and radial distance r from the load line is:
#
#       sigma_z = (3P / (2*pi*z^2)) * [1 / (1 + (r/z)^2)]^(5/2)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Page setup --------------------------------------------------------------
st.set_page_config(page_title="Boussinesq Point Load Stress", layout="wide")


# --- Core engineering function -----------------------------------------------
def boussinesq_sigma_z(P, r, z):
    """
    Vertical stress increase (sigma_z) at radial distance r and depth z
    below a point load P applied on the soil surface (Boussinesq solution).

    P : point load (kN)
    r : radial (horizontal) distance from the load axis (m)
    z : depth below the surface (m)

    Returns sigma_z in kPa (kN/m^2)
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    z_safe = np.where(z <= 0, 1e-6, z)
    influence_factor = 1.0 / (1.0 + (r / z_safe) ** 2) ** 2.5
    sigma_z = (3 * P / (2 * np.pi * z_safe ** 2)) * influence_factor
    return sigma_z


# --- Header / theory -----------------------------------------------------------
st.title("Stress in Soil due to a Point Load — Boussinesq's Equation")
st.latex(r"\sigma_z = \dfrac{3P}{2\pi z^2}\left[\dfrac{1}{1+(r/z)^2}\right]^{5/2}")
st.markdown(
    "For a point load **P** applied on the surface of a semi-infinite, "
    "homogeneous, isotropic, elastic soil mass, this gives the increase in "
    "vertical stress at depth **z** and radial distance **r** from the load axis. "
    "Adjust the controls in the sidebar and watch the plots update."
)

# --- Sidebar controls ----------------------------------------------------------
st.sidebar.header("Inputs")
P = st.sidebar.slider("Point Load, P (kN)", min_value=10, max_value=1000,
                       value=100, step=10)
r = st.sidebar.slider("Radial distance, r (m)", min_value=0.0, max_value=10.0,
                       value=1.0, step=0.1)
z_max = st.sidebar.slider("Max depth to plot (m)", min_value=2.0, max_value=30.0,
                           value=10.0, step=1.0)
z_example = st.sidebar.slider("Example depth for printed result (m)",
                               min_value=0.5, max_value=z_max, value=min(2.0, z_max),
                               step=0.5)

# --- Headline result -----------------------------------------------------------
sigma_example = boussinesq_sigma_z(P, r, z_example)
st.metric(label=f"σz at r = {r:.2f} m, z = {z_example:.2f} m",
          value=f"{sigma_example:.3f} kPa")

st.divider()

# --- Plots -----------------------------------------------------------------
col1, col2 = st.columns(2)

z_vals = np.linspace(0.2, z_max, 400)
sigma_vals = boussinesq_sigma_z(P, r, z_vals)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5.5))
    ax1.plot(sigma_vals, z_vals, color='firebrick', linewidth=2)
    ax1.invert_yaxis()
    ax1.set_xlabel(r'Vertical stress, $\sigma_z$ (kPa)')
    ax1.set_ylabel('Depth, z (m)')
    ax1.set_title(f'Stress vs Depth  (r = {r:.2f} m fixed)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(0, color='saddlebrown', linewidth=3)
    ax1.text(ax1.get_xlim()[1] * 0.4, -0.3, 'Ground surface', color='saddlebrown')
    st.pyplot(fig1)

with col2:
    r_grid = np.linspace(-z_max, z_max, 200)
    z_grid = np.linspace(0.2, z_max, 200)
    R, Z = np.meshgrid(r_grid, z_grid)
    Sigma = boussinesq_sigma_z(P, np.abs(R), Z)

    fig2, ax2 = plt.subplots(figsize=(6, 5.5))
    contour = ax2.contourf(R, Z, Sigma, levels=25, cmap='YlOrRd')
    cbar = fig2.colorbar(contour, ax=ax2)
    cbar.set_label(r'$\sigma_z$ (kPa)')
    ax2.invert_yaxis()
    ax2.axvline(r, color='blue', linestyle='--', linewidth=1.5,
                label=f'r = {r:.2f} m (selected)')
    ax2.plot(0, 0, marker='v', color='black', markersize=12, label='Point load P')
    ax2.set_xlabel('Radial distance, r (m)  [+ and - sides shown]')
    ax2.set_ylabel('Depth, z (m)')
    ax2.set_title('Stress Bulb Beneath a Point Load')
    ax2.legend(loc='lower right', fontsize=8)
    st.pyplot(fig2)

st.caption(
    "Boussinesq's theory assumes a homogeneous, isotropic, semi-infinite elastic "
    "soil medium and a point load applied at the surface. Widely used for "
    "estimating stress distribution and settlement beneath foundations."
)
