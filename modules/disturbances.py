import numpy as np
from debug import pick



def j2_acc(r):
    # Parameters:
    mu_earth = pick("mu_E")
    R_earth = pick("R_E")
    J2 = pick("J_2")

    x, y, z = r
    r_norm = np.linalg.norm(r)
    a_x = -3 / 2 * mu_earth * J2 * R_earth ** 2 / r_norm ** 5 * x * (1 - 5 * z ** 2 / r_norm ** 2)
    a_y = -3 / 2 * mu_earth * J2 * R_earth ** 2 / r_norm ** 5 * y * (1 - 5 * z ** 2 / r_norm ** 2)
    a_z = -3 / 2 * mu_earth * J2 * R_earth ** 2 / r_norm ** 5 * z * (3 - 5 * z ** 2 / r_norm ** 2)
    a = np.array([a_x, a_y, a_z])

    return a


def j2_acc_vec(r: np.ndarray):
    R_earth = pick("R_E")
    mu_earth = pick("mu_E")
    J2 = pick("J_2")

    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r_norm = np.linalg.norm(r, axis=1)
    a_x = -3 / 2 * mu_earth * J2 * R_earth ** 2 / r_norm ** 5 * x * (1 - 5 * z ** 2 / r_norm ** 2)
    a_y = -3 / 2 * mu_earth * J2 * R_earth ** 2 / r_norm ** 5 * y * (1 - 5 * z ** 2 / r_norm ** 2)
    a_z = -3 / 2 * mu_earth * J2 * R_earth ** 2 / r_norm ** 5 * z * (3 - 5 * z ** 2 / r_norm ** 2)
    a = np.array([a_x, a_y, a_z])

    return a



def srp_acc(r_e_sat, r_e_sun):
    """
    3x1 SRP acceleration for ODE integration (single state vector).
    Includes correct umbra/penumbra model and correct night-side condition.
    Returns acceleration in km/s^2.
    """

    # Convert to meters
    r_sat = r_e_sat * 1e3
    r_sun = r_e_sun * 1e3

    R_earth = pick("R_E") * 1e3
    R_sun   = pick("R_S") * 1e3

    B_R = pick("rad_press_ball_coeff")
    P   = pick("flux") / pick("light_speed")

    # Vector Sun → satellite
    r_sun_sat = r_sat - r_sun
    norm_sun_sat = np.linalg.norm(r_sun_sat)
    e_sun_sat = r_sun_sat / norm_sun_sat

    # PROJECTION OF r_sat onto Sun→sat direction
    proj_scalar = np.dot(r_sat, e_sun_sat)
    r_ps = proj_scalar * e_sun_sat

    # Distance from shadow axis
    r_e_p = r_sat - r_ps
    h_g = np.linalg.norm(r_e_p) - R_earth

    # Penumbra radius
    R_p = (np.linalg.norm(r_ps) / norm_sun_sat) * R_sun

    eta = h_g / R_p

    # --- CORRECT NIGHT-SIDE CONDITION ---
    sun_dir = r_sun / np.linalg.norm(r_sun)     # Earth → Sun
    proj_day = np.dot(r_sat, sun_dir)
    night = proj_day < 0                        # True if behind Earth

    # Atmospheric attenuation
    f_a = 1.0

    # Default acceleration (no SRP)
    a = np.zeros(3)

    # UMBRA
    if eta < -1 and night:
        a[:] = 0

    # PENUMBRA
    elif -1 <= eta < 1 and night:
        f_g = 1 - (1/np.pi)*np.arccos(eta) + (eta/np.pi)*np.sqrt(1 - eta**2)
        a[:] = -f_g * f_a * B_R * P * e_sun_sat

    # FULL ILLUMINATION (night side but Earth does not block)
    elif eta >= 1 and night:
        a[:] = -B_R * P * e_sun_sat

    # DAY SIDE (always full illumination)
    else:
        a[:] = -B_R * P * e_sun_sat

    return a / 1e3   # convert to km/s^2


def srp_acc_vec(r_e_sat, r_e_sun):
    """
    Vectorized SRP acceleration with umbra/penumbra/full illumination.
    Fixes the missing night-side condition that caused 2 eclipses per orbit.
    Returns acceleration in km/s^2.
    """

    # Convert to meters
    r_sat = r_e_sat * 1e3     # (N,3)
    r_sun = r_e_sun * 1e3     # (N,3)
    R_earth = pick("R_E") * 1e3
    R_sun = pick("R_S") * 1e3

    B_R = pick("rad_press_ball_coeff")
    P   = pick("flux") / pick("light_speed")

    # --- Vector Sun → satellite ---
    r_sun_sat = r_sat - r_sun                     # (N,3)
    norm_sun_sat = np.linalg.norm(r_sun_sat, axis=1)   # (N,)
    e_sun_sat = r_sun_sat / norm_sun_sat[:, None]      # (N,3)

    # --- Projection of satellite position onto Sun→sat direction ---
    proj_scalar = np.sum(r_sat * e_sun_sat, axis=1)[:, None]
    r_ps = proj_scalar * e_sun_sat

    # Distance from axis of the shadow cone
    r_e_p = r_sat - r_ps
    h_g = np.linalg.norm(r_e_p, axis=1) - R_earth      # (N,)

    # Penumbra radius at intersection
    R_p = (np.linalg.norm(r_ps, axis=1) / norm_sun_sat) * R_sun

    # Normalized geometric parameter η
    eta = h_g / R_p



    # Earth - Sun direction
    sun_dir = r_sun / np.linalg.norm(r_sun)

    # Projection of Earth→sat on Earth→Sun
    # proj_day > 0 -> day side, proj_day < 0 -> night side
    proj_day = np.sum(r_sat * sun_dir, axis=1)

    mask_night = proj_day < 0        # (N,) boolean array

    # --- Final output vector ---
    a = np.zeros_like(r_sat)

    # Masks (only applied on NIGHT SIDE!)
    mask_shadow      = (eta < -1)                & mask_night
    mask_penumbra    = (eta >= -1) & (eta < 1)   & mask_night
    mask_full        = (eta >= 1)                & mask_night

    # --- UMBRA ---
    # already zero from initialization

    # --- PENUMBRA ---
    if np.any(mask_penumbra):
        eta_p = eta[mask_penumbra]
        f_g = 1 - (1/np.pi)*np.arccos(eta_p) + (eta_p/np.pi)*np.sqrt(1 - eta_p**2)
        a[mask_penumbra] = -f_g[:, None] * B_R * P * e_sun_sat[mask_penumbra]

    # --- FULL ILLUMINATION (night-side but uncovered, and implicitly day-side) ---
    if np.any(mask_full):
        a[mask_full] = -B_R * P * e_sun_sat[mask_full]

    # Day-side points not in any shadow remain zero in eta-mask logic,
    # but they *must* be fully illuminated:
    mask_day = ~mask_night
    a[mask_day] = -B_R * P * e_sun_sat[mask_day]

    return a / 1e3   # to km/s^2



def drag_acc(r, v):
    # 3x1 acceleration due to drag, including rotating atmosphere

    # v is given in ECI => We need to remove Earth rotation (assuming atmosphere fixed to it)
    w_e = pick("w_E")
    w_e_vec = np.array([0, 0, w_e])
    v_a = v - np.linalg.cross(w_e_vec, r)  # type: ignore
    B = pick("ball_coeff")
    rho_mean = pick("ref_rho")
    v_a_norm = np.linalg.norm(v_a)
    h = np.linalg.norm(r) - pick("R_E")
    rho = np.exp(-h/pick("scale_height")) * rho_mean
    a = - 1 / 2 * rho * B * v_a_norm * v_a

    return a


def drag_acc_vec(r, v):
    # 3x1 acceleration due to drag, including rotating atmosphere

    # v is given in ECI => We need to remove Earth rotation (assuming atmosphere fixed to it)
    w_e = pick("w_E")
    w_e_vec = np.array([0, 0, w_e])

    # Using meters<
    r_m = r * 1e3
    v_m = v * 1e3

    v_a = v_m - np.linalg.cross(w_e_vec, r_m)  # type : ignore

    B = pick("ball_coeff")
    rho = pick("ref_rho")
    v_a_norm = np.linalg.norm(v_a, axis=1)

    a = - 1 / 2 * (rho * B * v_a_norm)[:, None] * v_a

    return a / 1e3  # Converting to km/s


def tbp_acc(r, r_sun, r_moon):
    # 3x1 acceleration due to Sun+Moon

    mu_s = pick("mu_S")
    mu_m = pick("mu_M")

    x, y, z = r
    x_s, y_s, z_s = r_sun
    x_m, y_m, z_m = r_moon

    # i = satellite, k = Earth, m = Moon, s = Sun
    r_is = np.sqrt((x_s - x) ** 2 + (y_s - y) ** 2 + (z_s - z) ** 2)
    r_ks = np.sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2)

    r_im = np.sqrt((x_m - x) ** 2 + (y_m - y) ** 2 + (z_m - z) ** 2)
    r_km = np.sqrt(x_m ** 2 + y_m ** 2 + z_m ** 2)

    # We consider both the perturbation of the Moon and of the Sun
    ax_s = mu_s * ((x_s - x) / r_is ** 3 - x_s / r_ks ** 3)
    ay_s = mu_s * ((y_s - y) / r_is ** 3 - y_s / r_ks ** 3)
    az_s = mu_s * ((z_s - z) / r_is ** 3 - z_s / r_ks ** 3)

    ax_m = mu_m * ((x_m - x) / r_im ** 3 - x_m / r_km ** 3)
    ay_m = mu_m * ((y_m - y) / r_im ** 3 - y_m / r_km ** 3)
    az_m = mu_m * ((z_m - z) / r_im ** 3 - z_m / r_km ** 3)

    a = np.array([ax_s + ax_m, ay_s + ay_m, az_s + az_m])

    return a



def tbp_acc_vec(r, r_sun, r_moon):
    # 3x1 acceleration due to Sun+Moon

    mu_s = pick("mu_S")
    mu_m = pick("mu_M")

    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    a = np.zeros_like(r)
    x_s, y_s, z_s = r_sun
    x_m, y_m, z_m = r_moon

    # i = satellite, k = Earth, m = Moon, s = Sun
    r_is = np.sqrt((x_s - x) ** 2 + (y_s - y) ** 2 + (z_s - z) ** 2)
    # print(r_is.shape[0])
    r_ks = np.sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2)

    r_im = np.sqrt((x_m - x) ** 2 + (y_m - y) ** 2 + (z_m - z) ** 2)
    r_km = np.sqrt(x_m ** 2 + y_m ** 2 + z_m ** 2)

    # We consider both the perturbation of the Moon and of the Sun
    ax_s = mu_s * ((x_s - x) / r_is ** 3 - x_s / r_ks ** 3)
    ay_s = mu_s * ((y_s - y) / r_is ** 3 - y_s / r_ks ** 3)
    az_s = mu_s * ((z_s - z) / r_is ** 3 - z_s / r_ks ** 3)

    ax_m = mu_m * ((x_m - x) / r_im ** 3 - x_m / r_km ** 3)
    ay_m = mu_m * ((y_m - y) / r_im ** 3 - y_m / r_km ** 3)
    az_m = mu_m * ((z_m - z) / r_im ** 3 - z_m / r_km ** 3)
    a[:, 0], a[:, 1], a[:, 2] = ax_s+ax_m, ay_s + ay_m, az_s + az_m

    return a
