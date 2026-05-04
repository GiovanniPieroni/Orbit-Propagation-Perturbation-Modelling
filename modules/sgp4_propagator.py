import numpy as np
import math


def initial_position_velocity(satellite, ti):
    """
    Computes initial position and velocity of the satellite using sgp4 model

    Args:
        satellite (satrec): satellite object
        ti (s): initial time in seconds
    Returns:
        ri: array of initial position in TEME, 3x1
        vi: array of initial velocity in TEME, 3x1
    """
    ti_days = ti / 86400
    frac, integer = math.modf(ti_days)
    if ti_days > 1:
        raise RuntimeError("Initial time is not within 24 hours of first epoch!")
    jd_i, fr_i = satellite.jdsatepoch + integer, satellite.jdsatepochF + frac
    e, ri, vi = satellite.sgp4(jd_i, fr_i)
    # make sure everything went fine
    assert (e == 0), "SGP4 failed at initial epoch"  # If e is /= 0 -> raise AssertionError('SGP4 failed')
    ri = np.array(ri)   # [km]
    vi = np.array(vi)   # [km/s]
    return ri, vi


def final_position_velocity(satellite, time):
    jd_i, fr_i = satellite.jdsatepoch, satellite.jdsatepochF
    fr = time.reshape(-1, 1) / 86400
    jd = jd_i + fr * 0
    e_prop, r_prop, v_prop = satellite.sgp4_array(jd, fr + fr_i)
    assert all(e_prop == 0), f"SGP4 failed at {sum(e_prop==0)} epochs"

    rf = np.array(r_prop[-1])
    vf = np.array(v_prop[-1])

    return rf, vf


def propagated_position_velocity(satellite, time):
    jd_i, fr_i = satellite.jdsatepoch, satellite.jdsatepochF
    fr = time.reshape(-1, 1) / 86400

    jd = jd_i + fr * 0
    e_prop, r_prop, v_prop = satellite.sgp4_array(jd, fr + fr_i)
    assert all(e_prop == 0), f"SGP4 failed at {sum(e_prop==0)} epochs"

    r_prop = np.array(r_prop)
    v_prop = np.array(v_prop)

    return r_prop, v_prop

