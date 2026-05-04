import time as pytime
import numpy as np
from typing import Callable

from modules import integrator



def cgpd_rms(r_ref: np.ndarray, r_test: np.ndarray) -> float:
    """
    Compute CGPD = RMS over time of the magnitude of the position difference.

    Parameters
    ----------
    r_ref : (N, 3) array
        Reference positions (SGP4) in km.
    r_test : (N, 3) array
        Test positions (integrated / optimised) in km.

    Returns
    -------
    float
        CGPD in meters.
    """
    r_ref = np.asarray(r_ref, dtype=float)
    r_test = np.asarray(r_test, dtype=float)

    if r_ref.shape != r_test.shape:
        raise ValueError(f"Shape mismatch in CGPD: ref {r_ref.shape}, test {r_test.shape}")

    diff = r_test - r_ref
    gpd = np.linalg.norm(diff, axis=1)  # km
    # cgpd_km = np.sqrt(np.mean(gpd**2))
    n = gpd.size
    cgpd_km = np.linalg.norm(gpd) / np.sqrt(n)
    return float(cgpd_km)  # km


def eval_objective_and_orbit(
    delta_y0: np.ndarray,
    y0_nominal: np.ndarray,
    dt: float,
    tf: float,
    rhs: Callable,
    r_sgp4: np.ndarray,
):
    """
    Evaluate objective f(x) = CGPD between SGP4 orbit and integrated orbit
    with perturbed initial state.

    Parameters
    ----------
    delta_y0 : (6,) array
        Initial state perturbation [dx, dy, dz, dvx, dvy, dvz] with
        positions in meters, velocities in m/s.
    y0_nominal : (6,) array
        Nominal initial state in km and km/s.
    dt : float
        Integration time step [s].
    tf : float
        Final time [s].
    rhs : callable
        RHS function for the ODE: rhs(t, y) -> dy/dt in km/s and km/s^2.
    r_sgp4 : (N+1, 3) array
        SGP4 positions in km, on the same time grid as the integrator.

    Returns
    -------
    cgpd : float
        CGPD in meters.
    y_iter : (N+1, 6) array
        Integrated state (iterated orbit) in km and km/s.
    t_step : float
        CPU time for this integration [s].
    """
    delta_y0 = np.asarray(delta_y0, dtype=float)
    if delta_y0.shape != (6,):
        raise ValueError(f"delta_y0 must be shape (6,), got {delta_y0.shape}")

    y0_nominal = np.asarray(y0_nominal, dtype=float)
    r_sgp4 = np.asarray(r_sgp4, dtype=float)

    # Convert perturbations to km and km/s
    delta_pos_km = delta_y0[:3]
    delta_vel_kms = delta_y0[3:]

    y0_pert = y0_nominal.copy()
    y0_pert[:3] += delta_pos_km
    y0_pert[3:] += delta_vel_kms

    # Integrate orbit with perturbed initial state
    t_start = pytime.perf_counter()
    y_iter = integrator.RK4(y0_pert, dt, tf, rhs)
    t_step = pytime.perf_counter() - t_start

    # Positions for CGPD
    r_iter = y_iter[:, :3]
    cgpd = cgpd_rms(r_sgp4, r_iter)

    return cgpd, y_iter, t_step


def make_objective(
    y0_nominal: np.ndarray,
    dt: float,
    tf: float,
    rhs: Callable,
    r_sgp4: np.ndarray,
):
    """
    Returns an objective function f(x) -> CGPD (meters)

    The returned function takes delta_y0 in [m, m/s] and returns
    CGPD in meters.
    """
    y0_nominal = np.asarray(y0_nominal, dtype=float)
    r_sgp4 = np.asarray(r_sgp4, dtype=float)

    def f(delta_y0: np.ndarray) -> float:
        cgpd, _, _ = eval_objective_and_orbit(delta_y0, y0_nominal, dt, tf, rhs, r_sgp4)
        return cgpd

    return f
