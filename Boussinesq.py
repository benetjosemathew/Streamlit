# =============================================================================
# STRESS IN SOIL DUE TO A POINT LOAD (Boussinesq's Theory)
# -----------------------------------------------------------------------------
# A teaching tool for First-Year Civil Engineering students showing how
# Python + interactive widgets can be used to explore geotechnical concepts.
#
# HOW TO USE IN GOOGLE COLAB:
#   1. Open a new Colab notebook (colab.research.google.com).
#   2. Paste this ENTIRE file into a single code cell.
#   3. Run the cell (Shift+Enter). Sliders + plots will appear below it.
#
# THEORY (Boussinesq, 1885):
#   For a point load P applied on the surface of a semi-infinite, homogeneous,
#   isotropic, elastic soil mass, the vertical stress increase at a point
#   located at depth z and radial distance r from the load line is:
#
#       sigma_z = (3P / (2*pi*z^2)) * [1 / (1 + (r/z)^2)]^(5/2)
#
#   This is one of the most widely used results in geotechnical engineering
#   (footing design, stress bulbs, settlement estimation, etc.)
# =============================================================================

# --- 1. Install / import dependencies (Colab already has these, this is safe) ---
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, Layout
from IPython.display import display, Markdown

# --- 2. Core engineering function -------------------------------------------
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
    # Avoid division by zero right at the surface (z=0)
    z_safe = np.where(z <= 0, 1e-6, z)
    influence_factor = 1.0 / (1.0 + (r / z_safe) ** 2) ** 2.5
    sigma_z = (3 * P / (2 * np.pi * z_safe ** 2)) * influence_factor
    return sigma_z


# --- 3. Interactive UI -------------------------------------------------------
style = {'description_width': '160px'}
slider_layout = Layout(width='500px')

P_slider = FloatSlider(value=100, min=10, max=1000, step=10,
                        description='Point Load, P (kN)',
                        style=style, layout=slider_layout)

r_slider = FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1,
                        description='Radial dist., r (m)',
                        style=style, layout=slider_layout)

zmax_slider = FloatSlider(value=10.0, min=2.0, max=30.0, step=1.0,
                           description='Max depth to plot (m)',
                           style=style, layout=slider_layout)


def update(P, r, z_max):
    # ---- 3a. Numeric answer at a chosen r, scanning depth ----
    z_vals = np.linspace(0.2, z_max, 400)
    sigma_vals = boussinesq_sigma_z(P, r, z_vals)

    # Point of interest at a representative depth (z = 2 m) for a printed example
    z_example = min(2.0, z_max)
    sigma_example = boussinesq_sigma_z(P, r, z_example)

    display(Markdown(
        f"### Result\n"
        f"At **r = {r:.2f} m**, **z = {z_example:.2f} m**, "
        f"**sigma_z = {sigma_example:.3f} kPa** for P = {P:.0f} kN\n\n"
        f"---"
    ))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Plot 1: sigma_z vs depth along the chosen r ----
    ax1 = axes[0]
    ax1.plot(sigma_vals, z_vals, color='firebrick', linewidth=2)
    ax1.invert_yaxis()  # depth increases downward
    ax1.set_xlabel(r'Vertical stress, $\sigma_z$ (kPa)')
    ax1.set_ylabel('Depth, z (m)')
    ax1.set_title(f'Stress vs Depth  (r = {r:.2f} m fixed)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(0, color='saddlebrown', linewidth=3)  # ground surface
    ax1.text(ax1.get_xlim()[1]*0.5, -0.3, 'Ground surface', color='saddlebrown')

    # ---- Plot 2: 2D stress bulb (contours) in the r-z plane ----
    ax2 = axes[1]
    r_grid = np.linspace(-z_max, z_max, 200)
    z_grid = np.linspace(0.2, z_max, 200)
    R, Z = np.meshgrid(r_grid, z_grid)
    Sigma = boussinesq_sigma_z(P, np.abs(R), Z)

    contour = ax2.contourf(R, Z, Sigma, levels=25, cmap='YlOrRd')
    cbar = fig.colorbar(contour, ax=ax2)
    cbar.set_label(r'$\sigma_z$ (kPa)')
    ax2.invert_yaxis()
    ax2.axvline(r, color='blue', linestyle='--', linewidth=1.5,
                label=f'r = {r:.2f} m (slider)')
    ax2.plot(0, 0, marker='v', color='black', markersize=12, label='Point load P')
    ax2.set_xlabel('Radial distance, r (m)  [+ and - sides shown]')
    ax2.set_ylabel('Depth, z (m)')
    ax2.set_title('Stress Bulb Beneath a Point Load')
    ax2.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.show()


display(Markdown(
    "## Stress in Soil due to a Point Load — Boussinesq's Equation\n"
    r"$$\sigma_z = \dfrac{3P}{2\pi z^2}\left[\dfrac{1}{1+(r/z)^2}\right]^{5/2}$$"
    "\n\nUse the sliders below to change the load and geometry, and watch "
    "the stress distribution respond in real time."
))

interact(update, P=P_slider, r=r_slider, z_max=zmax_slider)