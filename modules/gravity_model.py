import numpy as np
from debug import pick
import disturbances


def point_mass(r: np.ndarray):
    mu = pick("mu_E")
    r_norm = np.linalg.norm(r)
    r_ddot = - mu * r / r_norm**3

    return r_ddot


def total_acceleration(r, v, *,
                       use_point_mass=True,
                       use_J2=False,
                       use_drag=False,
                       use_SRP=False,
                       use_tbp=False,
                       params=None):
    """
    Returns total acceleration from the selected disturbances.
    r: position [km]
    v: velocity [km/s]

    :type r: np.ndarray
    :type v: np.ndarray
    :type use_point_mass: bool
    :type use_J2: bool
    :type use_drag: bool
    :type use_SRP: bool
    :type use_tbp: bool
    :type params: dict
    """

    a = np.zeros(3)

    if use_point_mass:
        a += point_mass(r)

    if use_J2:
        a += disturbances.j2_acc(r)

    if use_drag:
        a += disturbances.drag_acc(r, v)

    if use_SRP:
        r_e_sun = params["r_e_sun"]
        a += disturbances.srp_acc(r, r_e_sun)

    if use_tbp:
        a += disturbances.tbp_acc(r, params["r_e_sun"], params["r_e_moon"])

    return a
