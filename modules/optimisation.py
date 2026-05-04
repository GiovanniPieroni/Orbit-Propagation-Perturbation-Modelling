from dataclasses import dataclass
import time as pytime
import numpy as np
from debug import pick

from modules import objective as obj_mod


@dataclass
class OptResults:
    best_delta_y0: np.ndarray  # (6,) in [m, m/s]
    best_value: float  # CGPD [m]
    best_state: np.ndarray  # (N+1, 6) orbit in km, km/s
    history_values: list  # best-so-far CGPD per iteration
    history_grads: list  # Gradient norm per iteration
    history_times: list  # CPU time per objective call [s]
    total_time: float  # total CPU time [s]
    n_calls: int  # number of objective calls


def run_smart_optimization(
        y0_nominal,
        dt,
        tf,
        rhs_func,
        r_sgp4_ref,
        max_iters=5,
        patience=2,
        tolerance=1e-4,
        epsilon_grad=1e-4,  # IMPORTANTE: Uniformare questo valore
        warm_start_delta=None
):
    """
    1. Warm Start (se disponibile) o Nominale (0,0,0)
    2. Physics check (Energy +/-)
    3. Random Cone Search
    """

    range_pos = 1e-1
    range_vel_mag = 5e-4
    range_vel_rnd = 1e-4

    v_ref_unit = y0_nominal[3:6] / np.linalg.norm(y0_nominal[3:6])

    global_best_val = np.inf
    global_best_result = None
    no_improve_count = 0

    for k in range(max_iters):

        if k == 0:
            if warm_start_delta is not None:
                initial_guess = warm_start_delta.copy()
                desc = "Warm Start (Previous Optimal)"
            else:
                initial_guess = np.zeros(6)
                desc = "Nominal SGP4 (Zero Delta)"

        elif k == 1:
            if warm_start_delta is not None:
                initial_guess = np.zeros(6)
                desc = "Nominal SGP4 (Check)"
            else:
                delta_v = v_ref_unit * range_vel_mag
                initial_guess = np.concatenate([np.zeros(3), delta_v])
                desc = "Physics: Vel +Energy"

        elif k == 2:
            delta_v = v_ref_unit * -range_vel_mag
            initial_guess = np.concatenate([np.zeros(3), delta_v])
            desc = "Physics: Vel -Energy"

        else:
            delta_v_rand = random_velocity_cone(
                v_ref=y0_nominal[3:6],
                max_angle_rad=np.deg2rad(15),
                mag_range=(1e-5, range_vel_rnd)
            )
            delta_pos_rand = np.random.normal(0, range_pos, 3)
            base_delta = global_best_result.best_delta_y0 if global_best_result else np.zeros(6)

            initial_guess = base_delta + np.concatenate([delta_pos_rand, delta_v_rand])
            desc = f"Random: Cone Search (Iter {k})"

        print(f"    [Iter {k}] {desc}")

        # --- BFGS ---
        try:
            res = bfgs_search(
                y0_nominal=y0_nominal,
                dt=dt,
                tf=tf,
                rhs=rhs_func,
                r_sgp4=r_sgp4_ref,
                initial_delta=initial_guess,
                epsilon=epsilon_grad
            )

            curr_val = res.best_value

        except Exception as e:
            print(f"      Error: {e}")
            curr_val = np.inf
            res = None

        if res and curr_val < (global_best_val - tolerance):
            print(f"      >>> New Best: {curr_val:.4f} km")
            global_best_val = curr_val
            global_best_result = res
            no_improve_count = 0
        else:
            no_improve_count += 1
            # print(f"      No improv ({curr_val:.4f}). Patience {no_improve_count}/{patience}")

        if no_improve_count > patience:
            break

    return global_best_result


def random_velocity_cone(v_ref, max_angle_rad, mag_range):
    v_hat = v_ref / np.linalg.norm(v_ref)

    if abs(v_hat[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(v_hat, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)

    theta = np.random.uniform(0.0, max_angle_rad)
    phi = np.random.uniform(0.0, 2 * np.pi)

    direction = (
            np.cos(theta) * v_hat
            + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    )

    magnitude = np.random.uniform(mag_range[0], mag_range[1])
    return magnitude * direction


def compute_gradient_forward(delta_scaled, current_val, epsilon, scale, y0_nm, dt, tf, rhs, r_sgp4):
    n = len(delta_scaled)
    grad = np.zeros(n)
    n_calls = 0
    t_spent = 0.0

    for i in range(n):
        d_test = delta_scaled.copy()
        d_test[i] += epsilon

        delta_phys = d_test * scale

        val_test, _, t_s = obj_mod.eval_objective_and_orbit(
            delta_phys, y0_nm, dt, tf, rhs, r_sgp4
        )

        # Forward diff
        grad[i] = (val_test - current_val) / epsilon

        n_calls += 1
        t_spent += t_s

    return grad, t_spent, n_calls


def bfgs_search(
        y0_nominal: np.ndarray,
        dt: float,
        tf: float,
        rhs,
        r_sgp4: np.ndarray,
        initial_delta: np.ndarray,
        epsilon=1e-4,
) -> OptResults:
    # SCALING: 100m pos ~ 0.1 m/s vel
    scale = np.array([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])

    x_k = initial_delta / scale

    n_vars = len(initial_delta)
    I = np.eye(n_vars)
    H_k = np.eye(n_vars)

    history_vals = []
    history_grads = []
    history_times = []
    n_calls = 0
    t_start = pytime.perf_counter()

    # Initial value
    x_phys = x_k * scale
    f_k, best_state, t_s = obj_mod.eval_objective_and_orbit(x_phys, y0_nominal, dt, tf, rhs, r_sgp4)
    n_calls += 1

    # Initial gradient
    g_k, t_g, c_g = compute_gradient_forward(x_k, f_k, epsilon, scale, y0_nominal, dt, tf, rhs, r_sgp4)
    n_calls += c_g

    # Saving initial state
    history_vals.append(f_k)
    history_grads.append(np.linalg.norm(g_k))

    # BFGS LOOP
    max_iter = 50

    for k in range(max_iter):
        if np.linalg.norm(g_k) < 1e-6:
            break

        p_k = -H_k @ g_k

        if np.dot(p_k, g_k) > 0:
            H_k = I
            p_k = -g_k

        # --- LINE SEARCH ---
        alpha = 1.0
        c1 = 1e-4
        rho = 0.2

        accepted = False
        x_next = None
        f_next = None
        state_next = None

        for _ in range(15):
            x_try = x_k + alpha * p_k
            x_try_phys = x_try * scale

            f_try, s_try, t_try = obj_mod.eval_objective_and_orbit(x_try_phys, y0_nominal, dt, tf, rhs, r_sgp4)
            n_calls += 1
            history_times.append(t_try)

            # Armijo Condition (+ tolerance for numerical noise)
            if f_try <= f_k + c1 * alpha * np.dot(g_k, p_k) + 1e-6:
                accepted = True
                x_next = x_try
                f_next = f_try
                state_next = s_try
                break

            alpha *= rho

        if not accepted:
            break

        g_next, t_g, c_g = compute_gradient_forward(x_next, f_next, epsilon, scale, y0_nominal, dt, tf, rhs, r_sgp4)
        n_calls += c_g

        s_k = x_next - x_k
        y_k = g_next - g_k

        sy = np.dot(y_k, s_k)
        if sy > 1e-10:
            rho_k = 1.0 / sy
            V = I - rho_k * np.outer(s_k, y_k)
            H_k = V @ H_k @ V.T + rho_k * np.outer(s_k, s_k)

        # Advance
        x_k = x_next
        f_k = f_next
        g_k = g_next
        best_state = state_next

        history_vals.append(f_k)
        history_grads.append(np.linalg.norm(g_k))

    total_time = pytime.perf_counter() - t_start

    return OptResults(
        best_delta_y0=x_k * scale,
        best_value=float(f_k),
        best_state=best_state,
        history_values=history_vals,
        history_grads=history_grads,
        history_times=history_times,
        total_time=float(total_time),
        n_calls=int(n_calls),
    )
