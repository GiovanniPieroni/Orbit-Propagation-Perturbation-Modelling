import numpy as np
from config import DEFAULTS


def cartesian_to_keplerian(state):
    mu = DEFAULTS["mu_E"]
    r = state[:, 0:3]   # (N,3)
    v = state[:, 3:6]   # (N,3)

    # Magnitudes
    r_mag = np.linalg.norm(r, axis=1)       # (N,)
    v_mag = np.linalg.norm(v, axis=1)       # (N,)

    # Semi-major axis
    a = 1 / (2 / r_mag - v_mag**2 / mu)     # (N,)

    # Angular momentum
    h = np.cross(r, v)                      # (N,3)
    h_mag = np.linalg.norm(h, axis=1)       # (N,)

    # Eccentricity vector
    e_vec = np.cross(v, h) / mu - r / r_mag[:, None]  # (N,3)
    e_mag = np.linalg.norm(e_vec, axis=1)             # (N,)

    # Inclination
    i = np.arccos(h[:, 2] / h_mag)          # (N,)

    # Node vector
    e_z = np.array([0.0, 0.0, 1.0])
    n = np.cross(e_z, h)                    # (N,3)
    n_mag = np.linalg.norm(n, axis=1)       # (N,)

    # RAAN
    Omega = np.arctan2(n[:, 1], n[:, 0])    # (N,)

    # Argument of pericentre
    e_unit = e_vec / e_mag[:, None]         # (N,3)
    n_unit = n / n_mag[:, None]            # (N,3)

    e_dot_n = np.sum(e_unit * n_unit, axis=1)
    om = np.arccos(np.clip(e_dot_n, -1.0, 1.0))      # (N,)

    # Quadrant correction using h
    om *= np.sign(np.sum(np.cross(n_unit, e_unit) * h, axis=1))  # (N,)

    # True anomaly
    r_unit = r / r_mag[:, None]
    e_dot_r = np.sum(r_unit * e_unit, axis=1)
    theta = np.arccos(np.clip(e_dot_r, -1.0, 1.0))   # (N,)
    theta *= np.sign(np.sum(np.cross(e_vec, r) * h, axis=1))     # (N,)

    # Degrees
    i_deg = np.degrees(i)
    Om_deg = wrap_to_360(np.degrees(Omega))
    om_deg = wrap_to_180(np.degrees(om))
    th_deg = wrap_to_360(np.degrees(theta))
    u_deg = wrap_to_180(om_deg + th_deg)

    return (a,
            e_mag,
            i, i_deg,
            Omega, Om_deg,
            om, om_deg,
            theta, th_deg,
            u_deg)



def wrap_to_180(alfa):
    return (alfa + 180) % 360 - 180


def wrap_to_360(alfa):
    return alfa % 360


def keplerian_to_cartesian(sat, kep):
    mu = sat.mu

    a = kep.semi_major_axis
    e = kep.eccentricity
    i = kep.inclination_rad
    RAAN = kep.RAAN_rad
    om = kep.arg_of_per_rad
    th = kep.true_anomaly_rad

    #  trig
    ci, si = np.cos(i), np.sin(i)
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    co, so = np.cos(om), np.sin(om)

    #  radius
    r = a * (1 - e ** 2) / (1 + e * np.cos(th))

    #  perifocal vectors
    x_pf = r * np.cos(th)
    y_pf = r * np.sin(th)

    #  rotation matrix coeffs
    l1 = cO * co - sO * so * ci
    l2 = -cO * so - sO * co * ci
    m1 = sO * co + cO * so * ci
    m2 = -sO * so + cO * co * ci
    n1 = so * si
    n2 = co * si

    #  final position vectors
    x = l1 * x_pf + l2 * y_pf
    y = m1 * x_pf + m2 * y_pf
    z = n1 * x_pf + n2 * y_pf

    pos = np.column_stack((x, y, z))

    #  velocities
    H = np.sqrt(mu * a * (1 - e ** 2))

    vx_pf = -np.sin(th)
    vy_pf = e + np.cos(th)

    vx = mu / H * (l1 * vx_pf + l2 * vy_pf)
    vy = mu / H * (m1 * vx_pf + m2 * vy_pf)
    vz = mu / H * (n1 * vx_pf + n2 * vy_pf)

    vel = np.column_stack((vx, vy, vz))

    return pos, vel


# ----------------------------------------------------- A4 ----------------------------------------------------------- #



