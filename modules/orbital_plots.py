from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import frame_transformation
import utils

# --- TLE Loading for Period Calculation ---
try:
    with open("TLE_GONETSM24.txt", "r") as f:
        TLE_data = f.read().splitlines()
    sat = utils.create_satrec(TLE_data)
    mean_motion = sat.no_kozai / 60  # [rad/s]
    period = 2 * np.pi / mean_motion  # Orbital period
except Exception as e:
    period = None
# -------------------------------------------------------------------

# Constants from plots.py
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 'large'  # Matches plots.py size


def _compute_kep(y_cart: np.ndarray):
    a, ecc, i_rad, i_deg, Om_rad, Om_deg, om_rad, om_deg, th_rad, th_deg, u_deg = (
        frame_transformation.cartesian_to_keplerian(y_cart)
    )
    return a, ecc, i_deg, Om_deg, u_deg


def _plot_kep_residual_single(
        time,
        elem_sgp4,
        elem_int,
        elem_opt,
        ylabel_main,
        ylabel_diff,
        title,
        filename,
):
    """
    Plots 3 curves + 2 residuals using plots.py style (Blue/Orange/Green).
    Handles overlaps using linestyles.
    """
    time = np.asarray(time, dtype=float)
    elem_sgp4 = np.asarray(elem_sgp4, dtype=float)
    elem_int = np.asarray(elem_int, dtype=float)
    elem_opt = np.asarray(elem_opt, dtype=float)

    # Residuals
    res_int = elem_int - elem_sgp4
    res_opt = elem_opt - elem_sgp4

    # Setup Figure
    max_time = np.nanmax(time) if len(time) > 0 else 0
    orbital_lines = []
    if period is not None and max_time > period:
        orbital_lines = np.arange(period, max_time + 1, period, dtype=float)
        # Wider figure if vertical lines present
        fig, ax1 = plt.subplots(figsize=(10 + len(orbital_lines) * 0.5, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))

    fig.subplots_adjust(top=0.85)

    # --- MAIN CURVES (Left Axis) ---
    l1, = ax1.plot(time, elem_sgp4, label="SGP4",
                   color="tab:blue", linestyle="-", linewidth=9, alpha=0.9)

    l2, = ax1.plot(time, elem_int, label="Integrated",
                   color="tab:orange", linestyle="-", linewidth=5, alpha=0.9)

    l3, = ax1.plot(time, elem_opt, label="Optimised",
                   color="tab:cyan", linestyle="--", linewidth=3, alpha=0.8)

    ax1.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax1.set_ylabel(ylabel_main, fontsize=LABEL_SIZE)
    ax1.tick_params(axis="both", labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Minor ticks
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())


    # --- RESIDUALS (Right Axis) ---
    ax2 = ax1.twinx()

    l4, = ax2.plot(time, res_int, label="Res: Int - SGP4",
                   color="tab:orange", linestyle=":", linewidth=1.5, alpha=1)

    l5, = ax2.plot(time, res_opt, label="Res: Opt - SGP4",
                   color="tab:green", linestyle=":", linewidth=1.5, alpha=1)

    ax2.set_ylabel(ylabel_diff, fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # --- Vertical Lines (Orbital Period) ---
    if period is not None and len(orbital_lines) > 0:
        for lx in orbital_lines:
            ax1.axvline(lx, color='black', linestyle='-.', alpha=0.3, zorder=0)
            ax1.text(lx, 1.05, f"{int(lx / period)} T ≈ {int(lx)} s",
                     transform=ax1.get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=12, color='gray')

    # --- Legend ---
    lines = [l1, l2, l3, l4, l5]
    labels = [l.get_label() for l in lines]

    if period is not None and len(orbital_lines) > 0:
        lines.append(Line2D([0], [0], color='black', linestyle='-.', alpha=0.3, label='1T = orbital period'))
        labels.append('1T = orbital period')

    # Legend placement matching plots.py
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.08),
               ncol=3, fontsize=LEGEND_SIZE, frameon=True)

    ax1.set_title(title, y=1.22, fontsize=TITLE_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(filename)


def plot_kep_elements_and_residuals_three(
        time: np.ndarray,
        y_sgp4: np.ndarray,
        y_int: np.ndarray,
        y_opt: np.ndarray,
        out_dir: Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_sgp4, e_sgp4, i_sgp4, Om_sgp4, u_sgp4 = _compute_kep(y_sgp4)
    a_int, e_int, i_int, Om_int, u_int = _compute_kep(y_int)
    a_opt, e_opt, i_opt, Om_opt, u_opt = _compute_kep(y_opt)

    files = []

    files.append(_plot_kep_residual_single(
        time, a_sgp4, a_int, a_opt,
        ylabel_main="a [km]", ylabel_diff=r"$\Delta a$ [km]",
        title="Semi-major axis ($a$)",
        filename=out_dir / "kep_res_a.png"
    ))

    files.append(_plot_kep_residual_single(
        time, e_sgp4, e_int, e_opt,
        ylabel_main="e [-]", ylabel_diff=r"$\Delta e$ [-]",
        title="Eccentricity ($e$)",
        filename=out_dir / "kep_res_e.png"
    ))

    files.append(_plot_kep_residual_single(
        time, i_sgp4, i_int, i_opt,
        ylabel_main="i [deg]", ylabel_diff=r"$\Delta i$ [deg]",
        title="Inclination ($i$)",
        filename=out_dir / "kep_res_i.png"
    ))

    files.append(_plot_kep_residual_single(
        time, Om_sgp4, Om_int, Om_opt,
        ylabel_main=r"$\Omega$ [deg]", ylabel_diff=r"$\Delta \Omega$ [deg]",
        title=r"RAAN ($\Omega$)",
        filename=out_dir / "kep_res_RAAN.png"
    ))

    files.append(_plot_kep_residual_single(
        time, u_sgp4, u_int, u_opt,
        ylabel_main="u [deg]", ylabel_diff=r"$\Delta u$ [deg]",
        title="Argument of Latitude ($u$)",
        filename=out_dir / "kep_res_u.png"
    ))

    return files


def _mag(v):
    return np.linalg.norm(v, axis=1)


def _plot_mag_single(
        time,
        mag_sgp4,
        mag_int,
        mag_opt,
        ylabel_main,
        ylabel_diff,
        title,
        filename,
):
    """
    Plots magnitudes with same style as Keplerian elements.
    """
    time = np.asarray(time, dtype=float)

    diff_int = mag_int - mag_sgp4
    diff_opt = mag_opt - mag_sgp4

    max_time = np.nanmax(time) if len(time) > 0 else 0
    orbital_lines = []
    if period is not None and max_time > period:
        orbital_lines = np.arange(period, max_time + 1, period, dtype=float)
        fig, ax1 = plt.subplots(figsize=(10 + len(orbital_lines) * 0.5, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))

    fig.subplots_adjust(top=0.85)

    # Main Lines: Blue, Orange, Green (Dashed for Opt to see overlap)
    l1, = ax1.plot(time, mag_sgp4, label="SGP4",
                   color="tab:blue", linestyle="-", linewidth=9, alpha=0.9)
    l2, = ax1.plot(time, mag_int, label="Integrated",
                   color="tab:orange", linestyle="-", linewidth=5, alpha=0.9)
    l3, = ax1.plot(time, mag_opt, label="Optimised",
                   color="tab:cyan", linestyle="--", linewidth=3, alpha=0.8)

    ax1.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax1.set_ylabel(ylabel_main, fontsize=LABEL_SIZE)
    ax1.tick_params(axis="both", labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Residuals: Orange Dotted, Green Dotted
    ax2 = ax1.twinx()
    l4, = ax2.plot(time, diff_int, label="Res: Int - SGP4",
                   color="tab:orange", linestyle=":", linewidth=1.5, alpha=0.9)
    l5, = ax2.plot(time, diff_opt, label="Res: Opt - SGP4",
                   color="tab:green", linestyle=":", linewidth=1.5, alpha=0.9)

    ax2.set_ylabel(ylabel_diff, fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if period is not None and len(orbital_lines) > 0:
        for lx in orbital_lines:
            ax1.axvline(lx, color='black', linestyle='-.', alpha=0.3, zorder=0)
            ax1.text(lx, 1.05, f"{int(lx / period)} T ≈ {int(lx)} s",
                     transform=ax1.get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=12, color='gray')

    lines = [l1, l2, l3, l4, l5]
    labels = [l.get_label() for l in lines]
    if period is not None and len(orbital_lines) > 0:
        lines.append(Line2D([0], [0], color='black', linestyle='-.', alpha=0.3, label='1T = orbital period'))
        labels.append('1T = orbital period')

    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.08),
               ncol=3, fontsize=LEGEND_SIZE, frameon=True)

    ax1.set_title(title, y=1.22, fontsize=TITLE_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(filename)


def plot_mag_and_diff_three(
        time: np.ndarray,
        r_sgp4: np.ndarray,
        r_int: np.ndarray,
        r_opt: np.ndarray,
        v_sgp4: np.ndarray,
        v_int: np.ndarray,
        v_opt: np.ndarray,
        out_dir: Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    files.append(_plot_mag_single(
        time,
        _mag(r_sgp4), _mag(r_int), _mag(r_opt),
        ylabel_main="||r|| [km]", ylabel_diff=r"$\Delta ||r||$ [km]",
        title="Position Magnitude",
        filename=out_dir / "mag_pos_diff.png"
    ))

    files.append(_plot_mag_single(
        time,
        _mag(v_sgp4), _mag(v_int), _mag(v_opt),
        ylabel_main="||v|| [km/s]", ylabel_diff=r"$\Delta ||v||$ [km/s]",
        title="Velocity Magnitude",
        filename=out_dir / "mag_vel_diff.png"
    ))

    return files


def plot_ae_opt_metrics_vs_dt(dt_vals, total_times, n_calls, avg_times, out_dir, filename):
    out_dir = Path(out_dir)
    filename = out_dir / filename

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    l1, = ax1.plot(dt_vals, total_times, 'o-', color=color1, label='Total CPU Time [s]', linewidth=2)
    ax1.set_xlabel('Integration step dt [s]', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Total Time [s]', color=color1, fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    l2, = ax2.plot(dt_vals, n_calls, 's--', color=color2, label='# Func Calls', linewidth=2)
    ax2.set_ylabel('# Calls', color=color2, fontsize=LABEL_SIZE)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    color3 = 'tab:green'
    l3, = ax3.plot(dt_vals, avg_times, '^:', color=color3, label='Avg Time/Call [s]', linewidth=2)
    ax3.set_ylabel('Avg Time [s]', color=color3, fontsize=LABEL_SIZE)
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=TICK_SIZE)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=LEGEND_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return str(filename)


def plot_ae_cgpd_vs_dt(dt_vals, cgpd_vals, out_dir, filename):
    out_dir = Path(out_dir)
    filename = out_dir / filename

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dt_vals, cgpd_vals, 'o-', color='tab:blue', linewidth=2)

    ax.set_title("CGPD of optimised orbit vs dt", fontsize=TITLE_SIZE)
    ax.set_xlabel("Integration time step dt [s]", fontsize=LABEL_SIZE)
    ax.set_ylabel("CGPD of optimised orbit [km]", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return str(filename)


def plot_ae_total_vs_cgpd(total_times, cgpd_vals, out_dir, filename):
    out_dir = Path(out_dir)
    filename = out_dir / filename

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cgpd_vals, total_times, 'o-', color='tab:purple', linewidth=2)

    ax.set_title("Total CPU Time vs CGPD", fontsize=TITLE_SIZE)
    ax.set_xlabel("CGPD [km]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Total CPU Time [s]", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return str(filename)


def plot_ae_total_vs_cgpd_additional(total_times, cgpd_vals, dt_vals, out_dir, filename):
    """
    Plots Total CPU Time (Left Y) and Time Step dt (Right Y) as a function of CGPD (X).
    Visualizes the cost of accuracy.
    """
    out_dir = Path(out_dir)
    filename = out_dir / filename

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Asse Y1: Total CPU Time (Viola)
    color1 = 'tab:purple'
    l1, = ax1.plot(cgpd_vals, total_times, 'o-', color=color1, label='Total CPU Time', linewidth=2)
    ax1.set_xlabel("CGPD [km]", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Total CPU Time [s]", color=color1, fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=':', alpha=0.6)



    ax2 = ax1.twinx()
    color2 = 'tab:brown'
    l2, = ax2.plot(cgpd_vals, dt_vals, 's--', color=color2, label='Time Step (dt)', linewidth=2, alpha=0.6)
    ax2.set_ylabel("Time Step dt [s]", color=color2, fontsize=LABEL_SIZE)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)

    ax1.set_title("Pareto Front: Computational Cost & Step Size vs Accuracy", fontsize=TITLE_SIZE)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=LEGEND_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return str(filename)


def plot_convergence(history, out_dir, filename):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / filename

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, 'o-', color='tab:blue', linewidth=2)

    ax.set_title("Optimisation Convergence", fontsize=TITLE_SIZE)
    ax.set_xlabel("Iteration", fontsize=LABEL_SIZE)
    ax.set_ylabel("Cost Function (CGPD) [km]", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return str(filename)


def plot_energy_three(
        time: np.ndarray,
        r_sgp4: np.ndarray, v_sgp4: np.ndarray,
        r_int: np.ndarray, v_int: np.ndarray,
        r_opt: np.ndarray, v_opt: np.ndarray,
        out_dir: Path,
        mu: float = 398600.4418
):
    """
    Plots the Specific Mechanical Energy (v^2/2 - mu/r) and residuals.
    Useful to verify conservation of energy (Integrator) vs Decay (SGP4).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / "energy_diff.png"

    time = np.asarray(time, dtype=float)

    def calc_E(r_vec, v_vec):
        r_norm = np.linalg.norm(r_vec, axis=1)
        v_norm = np.linalg.norm(v_vec, axis=1)
        return 0.5 * v_norm ** 2 - mu / r_norm

    E_sgp4 = calc_E(r_sgp4, v_sgp4)
    E_int = calc_E(r_int, v_int)
    E_opt = calc_E(r_opt, v_opt)

    res_int = E_int - E_sgp4
    res_opt = E_opt - E_sgp4

    max_time = np.nanmax(time) if len(time) > 0 else 0
    orbital_lines = []
    if period is not None and max_time > period:
        orbital_lines = np.arange(period, max_time + 1, period, dtype=float)
        fig, ax1 = plt.subplots(figsize=(10 + len(orbital_lines) * 0.5, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 8))

    fig.subplots_adjust(top=0.85)

    # Main Curves (Left Axis) - Energy
    l1, = ax1.plot(time, E_sgp4, label="SGP4",
                   color="tab:blue", linestyle="-", linewidth=2.5, alpha=0.6)
    l2, = ax1.plot(time, E_int, label="Integrated",
                   color="tab:orange", linestyle="-", linewidth=2, alpha=0.8)
    l3, = ax1.plot(time, E_opt, label="Optimised",
                   color="tab:green", linestyle="--", linewidth=2.2, alpha=1.0)

    ax1.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax1.set_ylabel(r"Specific Energy [km$^2$/s$^2$]", fontsize=LABEL_SIZE)
    ax1.tick_params(axis="both", labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Residuals (Right Axis) - Delta Energy
    ax2 = ax1.twinx()
    l4, = ax2.plot(time, res_int, label="Res: Int - SGP4",
                   color="tab:orange", linestyle=":", linewidth=1.5, alpha=0.9)
    l5, = ax2.plot(time, res_opt, label="Res: Opt - SGP4",
                   color="tab:green", linestyle=":", linewidth=1.5, alpha=0.9)

    ax2.set_ylabel(r"$\Delta$ Energy [km$^2$/s$^2$]", fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Vertical Lines (Orbital Period)
    if period is not None and len(orbital_lines) > 0:
        for lx in orbital_lines:
            ax1.axvline(lx, color='black', linestyle='-.', alpha=0.3, zorder=0)
            ax1.text(lx, 1.05, f"{int(lx / period)} T ≈ {int(lx)} s",
                     transform=ax1.get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=12, color='gray')

    # Legend
    lines = [l1, l2, l3, l4, l5]
    labels = [l.get_label() for l in lines]
    if period is not None and len(orbital_lines) > 0:
        lines.append(Line2D([0], [0], color='black', linestyle='-.', alpha=0.3, label='1T = orbital period'))
        labels.append('1T = orbital period')

    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=LEGEND_SIZE, frameon=True)

    ax1.set_title("Orbital Specific Energy & Residuals", fontsize=TITLE_SIZE, y=1.15)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(filename)


def plot_convergence_dual(history_cgpd, history_grad, out_dir, filename):
    out_dir = Path(out_dir)
    filename = out_dir / filename

    iters = np.arange(1, len(history_cgpd) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    l1, = ax1.plot(iters, history_cgpd, 'o-', color=color1, label='CGPD [km]', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Cost Function (CGPD) [km]', color=color1, fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    l2, = ax2.plot(iters, history_grad, 's--', color=color2, label='||Gradient||', linewidth=2)
    ax2.set_ylabel('Gradient Norm (Scaled)', color=color2, fontsize=LABEL_SIZE)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
    ax2.set_yscale('log')

    ax1.set_title("Optimization Convergence: CGPD & Gradient", fontsize=TITLE_SIZE)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=LEGEND_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return str(filename)


def plot_ae_opt_metrics_with_theory(dt_vals, total_times, n_calls, avg_times, out_dir, filename):
    out_dir = Path(out_dir)
    filename = out_dir / filename

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Total Time (Left Axis)
    color1 = 'tab:blue'
    l1, = ax1.plot(dt_vals, total_times, 'o-', color=color1, label='Total CPU Time [s]', linewidth=2)
    ax1.set_xlabel('Integration step dt [s]', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Total Time [s]', color=color1, fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. N Calls (Right Axis 1)
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    l2, = ax2.plot(dt_vals, n_calls, 's--', color=color2, label='# Func Calls', linewidth=2)
    ax2.set_ylabel('# Calls', color=color2, fontsize=LABEL_SIZE)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)

    # 3. Avg Time + Theory (Right Axis 2 - Offset)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    color3 = 'tab:green'

    l3, = ax3.plot(dt_vals, avg_times, '^:', color=color3, label='Avg Time/Call [s]', linewidth=2)


    theory_shape = dt_vals ** (-0.75)
    # scale_factor = np.mean(avg_times) / np.mean(theory_shape)
    # theory_vals = scale_factor * theory_shape
    theory_vals = theory_shape
    l4, = ax3.plot(dt_vals, theory_vals, color='black', linestyle='--', alpha=0.6,
                   linewidth=2, label=r' $dt^{-0.75} \ [s^{-\frac{3}{4}}]$')

    ax3.set_ylabel('Avg Time [s]', color=color3, fontsize=LABEL_SIZE)
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=TICK_SIZE)

    # Legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=LEGEND_SIZE)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return str(filename)
