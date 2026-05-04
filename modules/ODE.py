import gravity_model
import numpy as np


def make_rhs(*,
             include_point_mass=True,
             include_j2=False,
             include_srp=False,
             include_drag=False,
             include_tbp=False,
             params=None):
    """
    Returns f(t, y) to pass to integrator.
    y = [r(0:3); v(3:6)], ECI.
    """
    if params is None:
        params = {}

    def rhs(t, y):
        r = y[:3]
        v = y[3:]

        a = gravity_model.total_acceleration(
            r, v,
            use_point_mass=include_point_mass,
            use_J2=include_j2,
            use_drag=include_drag,
            use_SRP=include_srp,
            use_tbp=include_tbp,
            params=params
        )

        dy = np.zeros_like(y)
        dy[:3] = v
        dy[3:] = a
        return dy

    return rhs
