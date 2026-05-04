################################################## CONFIGURATION #######################################################
# Libraries and files:
import numpy as np
from debug import pick
from yaml import dump
import subprocess
import sys
import time as pytime
from pathlib import Path

# Make sure Python sees the modules package
sys.path.append(str(Path(__file__).parent / "modules"))
from modules import integrator, plots, frame_transformation, getsource,  utils,  sgp4_propagator, \
                    ODE,  objective, optimisation, orbital_plots


# Make sure the output directory exists
out_dir = Path("plots")
out_dir.mkdir(exist_ok=True, parents=True)

# Data storage
data_dir = Path("data")
data_dir.mkdir(exist_ok=True, parents=True)

# Answers dictionary initialization:
Answers = dict()

# Language:
Answers["ss_REG_Language_GEN_0"] = "Python"



kep_three_src, kep_three_code = getsource.getsourcefunc(
    orbital_plots.plot_kep_elements_and_residuals_three
)
kep_single_src, kep_single_code = getsource.getsourcefunc(orbital_plots._plot_kep_residual_single)
Answers["ls_MDT_OrbKepResPlot_SRC_0"] = kep_three_src
Answers["ss_REG_OrbKepResPlot_CODE_1"] = kep_three_code + '\n\n' + kep_single_code


################################################## TASK: Retrieve the TLE data #########################################
NORAD = pick("NORAD")
TLE_data = utils.get_tle_data(NORAD, "TLE_GONETSM24.txt")
Answers["ls_MDT_TLE_GEN_0"] = TLE_data

################################################## TASK: Update your orbit (SGP4) #######################################

sat = utils.create_satrec(TLE_data)

mean_motion = sat.no_kozai / 60.0  # [rad/s]
T = 2 * np.pi / mean_motion        # orbital period [s]

rev = pick("rev")                  # number of propagated orbital periods
ti = pick("ti")
tf = rev * T

N = pick("N")                      # number of steps for base dt
dt = (tf - ti) / N
time_SGP4 = np.linspace(ti, tf, N + 1)

Answers["sf_MDT_OrbTimeIni_NUM_0"] = float(ti)
Answers["sf_MDT_OrbTimeEnd_NUM_0"] = float(tf)
Answers["sf_MDT_OrbTimeStep_NUM_0"] = float(dt)

ri, vi = sgp4_propagator.initial_position_velocity(sat, ti)
rf, vf = sgp4_propagator.final_position_velocity(sat, time_SGP4)

Answers["lf_REG_OrbPosIni_NUM_0"] = (ri * 1e3).tolist()
Answers["lf_REG_OrbVelIni_NUM_0"] = (vi * 1e3).tolist()
Answers["lf_REG_OrbPosEnd_NUM_1"] = (rf * 1e3).tolist()
Answers["lf_REG_OrbVelEnd_NUM_1"] = (vf * 1e3).tolist()

# Full SGP4 orbit
r_SGP4, v_SGP4 = sgp4_propagator.propagated_position_velocity(sat, time_SGP4)
y_SGP4 = np.concatenate((r_SGP4, v_SGP4), axis=1)  # (N+1, 6), km and km/s

np.savetxt(data_dir / "SGP4_state.txt", y_SGP4, fmt="%.10e")
np.savetxt(data_dir / "SGP4_time.txt", time_SGP4, fmt="%.10e")

################################################## TASK: Perform the orbit integration #################################

# Integrator name:
Answers["ss_MDT_Integrator_GEN_0"] = "RK4"

# Force model:
force_components = ["J2"]
Answers["ls_MDT_ForceModelComponents_GEN_0"] = force_components



# Build RHS
moon_pos = np.array([3.417822610584837e5,  1.215989230467656e5,  7.652650559494867e4])
sun_pos = np.array([1.831719180077896e7, -1.339293092861967e8, -5.805577839671485e7])
# Answers['lf_REG_SunPos_GEN_0'] = (sun_pos*1e3).tolist()
# Answers['lf_REG_MoonPos_GEN_0'] = (moon_pos*1e3).tolist()

params = {
    "r_e_sun": sun_pos,
    "r_e_moon": moon_pos,
}

RHS_int = ODE.make_rhs(
    include_point_mass=True,
    include_j2=True,
    include_srp=False,
    include_drag=False,
    include_tbp=pick("include_TBP"),
    params=params,
)

y0_int = y_SGP4[0, :]




t_cpu_start = pytime.perf_counter()
t_wall_start = pytime.time()

y_int = integrator.RK4(y0_int, dt, tf, RHS_int)

t_cpu_total = pytime.perf_counter() - t_cpu_start
t_wall_total = pytime.time() - t_wall_start

t_cpu_eff = t_cpu_total / (y_int.shape[0] - 1)
t_wall_eff = t_wall_total / (y_int.shape[0] - 1)

print(f"Wall-clock time for one integration: {t_wall_total:.6f} s")

Answers["sf_REG_EulerTimeTotal_NUM_0.5"] = float(t_wall_total)
Answers["sf_REG_EulerTimeEff_NUM_0.5"] = float(t_wall_eff)

Answers["lf_REG_IntPosIni_NUM_0"] = (y_int[0, 0:3] * 1e3).tolist()
Answers["lf_REG_IntVelIni_NUM_0"] = (y_int[0, 3:6] * 1e3).tolist()
Answers["lf_REG_IntPosEnd_NUM_1"] = (y_int[-1, 0:3] * 1e3).tolist()
Answers["lf_REG_IntVelEnd_NUM_1"] = (y_int[-1, 3:6] * 1e3).tolist()

np.savetxt(data_dir / "INT_state.txt", y_int, fmt="%.10e")
np.savetxt(data_dir / "INT_time.txt", time_SGP4, fmt="%.10e")

################################################## TASK: Plot the Keplerian elements ###################################

# SGP4 orbit
(a_SGP4, e_SGP4, i_rad_SGP4, i_deg_SGP4,
 Om_rad_SGP4, Om_deg_SGP4, om_rad_SGP4, om_deg_SGP4,
 th_rad_SGP4, th_deg_SGP4, u_deg_SGP4) = frame_transformation.cartesian_to_keplerian(y_SGP4)

plot_SGP4_files = []
f_aei_SGP4 = out_dir / "a_e_i_SGP4.png"
f_uOm_SGP4 = out_dir / "u_RAAN_SGP4.png"
plot_SGP4_files.extend([str(f_aei_SGP4), str(f_uOm_SGP4)])

if not pick("skip_plots"):
    plots.plot_three_scales(
        time=time_SGP4.tolist(),
        data1=a_SGP4, label1="Semi-major Axis ($a$)", ylabel1="a [km]",
        data2=e_SGP4, label2="Eccentricity ($e$)", ylabel2="e [-]",
        data3=i_deg_SGP4, label3="Inclination ($i$)", ylabel3="i [deg]",
        title="SGP4 orbit: a, e, i",
        filename=str(f_aei_SGP4),
    )
    plots.plot_two_scales(
        time=time_SGP4.tolist(),
        data1=u_deg_SGP4, label1=r"Argument of latitude ($u$)", ylabel1="u [deg]",
        data2=Om_deg_SGP4, label2=r"RAAN ($\Omega$)", ylabel2=r"$\Omega$ [deg]",
        title="SGP4 orbit: u and RAAN",
        filename=str(f_uOm_SGP4),
    )
Answers["ls_REG_OrbKep_PLOT_1"] = plot_SGP4_files

# Integrated orbit
(a_int_k, e_int_k, i_rad_int_k, i_deg_int_k,
 Om_rad_int_k, Om_deg_int_k, om_rad_int_k, om_deg_int_k,
 th_rad_int_k, th_deg_int_k, u_deg_int_k) = frame_transformation.cartesian_to_keplerian(y_int)

plot_INT_files = []
f_aei_INT = out_dir / "a_e_i_INT.png"
f_uOm_INT = out_dir / "u_RAAN_INT.png"
plot_INT_files.extend([str(f_aei_INT), str(f_uOm_INT)])

if not pick("skip_plots"):
    plots.plot_three_scales(
        time=time_SGP4.tolist(),
        data1=a_int_k, label1="Semi-major Axis ($a$)", ylabel1="a [km]",
        data2=e_int_k, label2="Eccentricity ($e$)", ylabel2="e [-]",
        data3=i_deg_int_k, label3="Inclination ($i$)", ylabel3="i [deg]",
        title="Integrated orbit: a, e, i",
        filename=str(f_aei_INT),
    )
    plots.plot_two_scales(
        time=time_SGP4.tolist(),
        data1=u_deg_int_k, label1=r"Argument of latitude ($u$)", ylabel1="u [deg]",
        data2=Om_deg_int_k, label2=r"RAAN ($\Omega$)", ylabel2=r"$\Omega$ [deg]",
        title="Integrated orbit: u and RAAN",
        filename=str(f_uOm_INT),
    )
Answers["ls_REG_EulerKep_PLOT_1"] = plot_INT_files

################################################## TASK: Objective function ###########################################

obj_src1, obj_code1 = getsource.getsourcefunc(objective.make_objective)
obj_src2, obj_code2 = getsource.getsourcefunc(objective.cgpd_rms)
obj_src3, obj_code3 = getsource.getsourcefunc(objective.eval_objective_and_orbit)

Answers["ls_MDT_ObjFun_SRC_0"] = obj_src1
Answers["ss_REG_ObjFun_CODE_1"] = obj_code1 + "\n\n" + obj_code3 + "\n\n" + obj_code2

f_obj = objective.make_objective(
    y0_nominal=y0_int,
    dt=dt,
    tf=tf,
    rhs=RHS_int,
    r_sgp4=y_SGP4[:, :3],
)

delta_y0_zero = np.zeros(6)
cgpd_int = f_obj(delta_y0_zero)
Answers["sf_REG_CGPDInt_NUM_1"] = float(cgpd_int*1e3)

################################################## TASK: Implement an optimisation algorithm ###########################
Answers["ss_MDT_OptName_GEN_0"] = "bfgs"

opt_src, opt_code = getsource.getsourcefunc(optimisation.bfgs_search)
Answers["ls_MDT_Opt_SRC_0"] = opt_src
Answers["ss_REG_Opt_CODE_6"] = opt_code

# ################################################## TASK: Optimise the initial state vector #############################

init_positions = []   # (N, 3)
final_cgpd = []       # (N,)

# Parameters for global search
max_global_iters = 10  # Total attempts
patience = 5  # Unsuccessful Iterations before stopping (CGPD_i+1 > CGPD_i)
tolerance = 1e-3  # Minimum difference in CGPD for declaring a better one

# Initial delta range
range_pos = 1e-1  # 0.1 km
range_vel = 1e-5  # 1e-5 km/s


print("\n\n========================================================")
print(f"=== MAIN OPTIMIZATION (dt={dt:.4f}s) ===")
print(    "========================================================")

OPT_PARAMS = {
    'max_iters': 6,
    'patience': 3,
    'tolerance': 1e-4,
    'epsilon_grad': 1e-4  # Epsilon per il gradiente
}


opt_results = optimisation.run_smart_optimization(
    y0_nominal=y0_int,
    dt=dt,
    tf=tf,
    rhs_func=RHS_int,
    r_sgp4_ref=y_SGP4[:, :3],
    warm_start_delta=None,
    **OPT_PARAMS
)

if opt_results is None:
    raise RuntimeError("Optimization failed to find a valid result.")

print(f"\n=== Optimization completed ===")
print(f"Best CGPD Final: {opt_results.best_value:.9f} km")


Answers["sf_REG_OptTotal_NUM_0.5"] = float(opt_results.total_time)
Answers["sf_REG_OptEff_NUM_0.5"] = float(opt_results.total_time / max(opt_results.n_calls, 1))

y_opt = opt_results.best_state

Answers["lf_REG_OptPosIni_NUM_0"] = (y_opt[0, 0:3] * 1e3).tolist()
Answers["lf_REG_OptVelIni_NUM_0"] = (y_opt[0, 3:6] * 1e3).tolist()
Answers["lf_REG_OptPosEnd_NUM_0"] = (y_opt[-1, 0:3] * 1e3).tolist()
Answers["lf_REG_OptVelEnd_NUM_0"] = (y_opt[-1, 3:6] * 1e3).tolist()
Answers["sf_REG_CGPDOpt_NUM_1"] = float(opt_results.best_value*1e3)

np.savetxt(data_dir / "OPT_state.txt", y_opt, fmt="%.10e")

# Verification
CGPD_ver = objective.cgpd_rms(y_SGP4[:, :3], opt_results.best_state[:, :3])
if abs(CGPD_ver - opt_results.best_value) > 1e-6:
    print(f"Warning: Re-calculated CGPD ({CGPD_ver}) differs from optimiser output.")
else:
    print("CGPD verification passed.")
################################################## TASK: Convergence plot ##############################################
conv_file = orbital_plots.plot_convergence(
    opt_results.history_values, out_dir=out_dir, filename="conv_cgpd.png"
)
Answers["ls_REG_Conv_PLOT_2"] = [conv_file]

## Additional plot
# Source code filename: ls_MDT_AltPlot_SRC_0
# ss_REG_AltPlot_CODE_0
conv_dual_file = orbital_plots.plot_convergence_dual(
    opt_results.history_values,
    opt_results.history_grads,
    out_dir=out_dir,
    filename="conv_cgpd_grad.png"
)
Answers["ls_REG_Conv_PLOT_2"].append(str(conv_dual_file))

additional_plot_src, additional_plot_code = getsource.getsourcefunc(orbital_plots.plot_convergence_dual)



################################################## TASK: Keplerian elements and their residuals ########################
kep_files = orbital_plots.plot_kep_elements_and_residuals_three(
    time=time_SGP4,
    y_sgp4=y_SGP4,
    y_int=y_int,
    y_opt=y_opt,
    out_dir=out_dir,
)
Answers["ls_REG_OrbKepRes_PLOT_7"] = kep_files



################################################## TASK: Position and velocity magnitude plots #########################
# alt_src, alt_code = getsource.getsourcefunc(orbital_plots.plot_mag_and_diff_three)
# Answers["ls_MDT_AltPlot_SRC_0"] = alt_src
# Answers["ss_REG_AltPlot_CODE_0"] = alt_code

pos_file, vel_file = orbital_plots.plot_mag_and_diff_three(
    time=time_SGP4,
    r_sgp4=y_SGP4[:, :3],
    r_int=y_int[:, :3],
    r_opt=y_opt[:, :3],
    v_sgp4=y_SGP4[:, 3:6],
    v_int=y_int[:, 3:6],
    v_opt=y_opt[:, 3:6],
    out_dir=out_dir,
)
Answers["ls_REG_MagDiff_PLOT_5"] = [pos_file, vel_file]



## Energy
mu_val = pick("mu_E")

energy_file = orbital_plots.plot_energy_three(
    time=time_SGP4,
    r_sgp4=y_SGP4[:, :3], v_sgp4=y_SGP4[:, 3:6],
    r_int=y_int[:, :3],   v_int=y_int[:, 3:6],
    r_opt=y_opt[:, :3],   v_opt=y_opt[:, 3:6],
    out_dir=out_dir,
    mu=mu_val
)

################################################## TASK: Analyse optimisation results  #################################
# Regarding the plot(s) in the tasks above (Plot the convergence of the optimisation and Plot the SGP4,
# integrated and optimised orbits):

Answers["ss_REG_OrbObs_OIC_10"] = """
- O1: The plot conv_cgpd_grad.png shows a rapid initial descent of the objective function, dropping from > 2 km 
      to ≈ 500 m within the first three iterations. The algorithm reaches the final plateau of ≈ 300 m more slowly, 
      and CGPD values are essentially constant after the 20th iteration, where the gradient norm remains well below 1e-2
      and decreases to values lower than 1e-6.

- O2: In kep_res_u.png, the difference in the argument of latitude (Δu) for the Integrated orbit shows a 
      monotonic increase from 0 to ≈ 0.035 deg. In contrast, the Optimized orbit residual remains bounded 
      within ± 0.005 deg and crosses zero twice, reaching its minimum magnitude approximately at the midpoint 
      of the simulation time.

- O3: The plot mag_pos_diff.png shows that the Integrated position residual starts at 0 km and diverges 
      monotonically to about − 0.4 km. The Optimized residual, instead, starts with a non-zero offset of 
      approximately + 0.2 km and decreases to ≈ − 0.2 km, crossing zero again approximately at the
      midpoint of the simulation time.

- O4: The plots of the residuals of the orbital elements (a, e, i, RAAN) show that the residuals between 
      opt/SGP4 and int/SGP4 generally follow similar trends but are vertically offset relative to each other. 
      None of the optimized element residuals are ~zero, tend to zero or oscillate around zero.
"""

Answers["ss_REG_OrbInt_OIC_15"] = """
- I1: Based on O1, the convergence behavior reflects the geometry of the cost function. The sharp initial drop
      indicates the algorithm starting on a steep slope where large gradient magnitudes drive rapid improvements. The 
      final plateau corresponds to the optimizer navigating the flatter region near the local minimum, where the 
      gradient magnitude decreases, requiring fine adjustments to satisfy the tolerance.

- I2: Observations O2 and O3 indicate that the optimizer minimizes the RMS by converting a unilateral divergence 
      (seen in the integrated orbit) into a bounded, symmetric error distribution. Since the numerical model lacks 
      perturbation terms present in SGP4 (e.g., drag), the optimizer compensates for the dynamic mismatch by 
      selecting an initial state with a bias. This forces the satellite to start with a positive error, drift 
      through zero, and end with a negative error, mathematically yielding a lower RMS than a function diverging from zero.
"""

Answers["ss_REG_OrbCon_OIC_23"] = """
Conclusion: When minimizing the RMS cumulative global position difference between two dynamical models with different force formulations, 
            the optimal solution is a trajectory that centers the error distribution around zero (symmetric deviation) 
            rather than matching the initial conditions. Consequently, achieving a minimum RMS residual implies 
            accepting a systematic bias in the orbital elements to compensate for the physical model discrepancies 
            over the integration interval.
"""

################################################## ASSIGNMENT EXCELLENCE ###############################################
print("\n\n========================================================")
print("=== ASSIGNMENT EXCELLENCE: DT SENSITIVITY ===")
print("========================================================")

N_AE = np.geomspace(100, 1000, 10, dtype=int)

AE_dt_vals = []
AE_total_times = []
AE_n_calls = []
AE_avg_call_times = []
AE_cgpd_vals = []

previous_best_delta = None

for idx_N, N_j in enumerate(N_AE):

    # 1.
    dt_j = (tf - ti) / N_j
    time_j = np.linspace(ti, tf, N_j + 1)

    print(f"\n>>> AE Run {idx_N + 1}/{len(N_AE)} | Steps: {N_j} | dt: {dt_j:.4f} s")

    # 2.
    r_S_j, v_S_j = sgp4_propagator.propagated_position_velocity(sat, time_j)

    # 3.
    res_j = optimisation.run_smart_optimization(
        y0_nominal=y0_int,
        dt=dt_j,
        tf=tf,
        rhs_func=RHS_int,
        r_sgp4_ref=r_S_j,
        warm_start_delta=previous_best_delta,  # Passiamo il risultato precedente!
        **OPT_PARAMS
    )

    # 4.
    if res_j is not None:

        previous_best_delta = res_j.best_delta_y0.copy()

        AE_dt_vals.append(dt_j)
        AE_total_times.append(res_j.total_time)
        AE_n_calls.append(res_j.n_calls)
        AE_avg_call_times.append(res_j.total_time / max(res_j.n_calls, 1))
        AE_cgpd_vals.append(res_j.best_value)

        print(f"    -> Result: CGPD = {res_j.best_value:.4f} km")
    else:
        print(f"    -> Optimization failed for this step.")
        AE_dt_vals.append(dt_j)
        AE_total_times.append(np.nan)
        AE_n_calls.append(np.nan)
        AE_avg_call_times.append(np.nan)
        AE_cgpd_vals.append(np.nan)


# Convert lists to arrays for plotting
AE_dt_vals = np.array(AE_dt_vals)
AE_total_times = np.array(AE_total_times)
AE_n_calls = np.array(AE_n_calls)
AE_avg_call_times = np.array(AE_avg_call_times)
AE_cgpd_vals = np.array(AE_cgpd_vals)


# --- Generate Plots ---

# 1. Metrics vs dt (CPU time, etc.)
ae_metrics_file = orbital_plots.plot_ae_opt_metrics_vs_dt(
    dt_vals=AE_dt_vals,
    total_times=AE_total_times,
    n_calls=AE_n_calls,
    avg_times=AE_avg_call_times,
    out_dir=out_dir,
    filename="ae_opt_metrics_vs_dt.png",
)
Answers["ls_AEX_OptTotalStep_PLOT_2"] = [ae_metrics_file]


ae_metrics_file_additional = orbital_plots.plot_ae_opt_metrics_with_theory(
    dt_vals=AE_dt_vals,
    total_times=AE_total_times,
    n_calls=AE_n_calls,
    avg_times=AE_avg_call_times,
    out_dir=out_dir,
    filename="ae_opt_metrics_vs_dt_additional.png",
)

Answers["ls_AEX_OptTotalStep_PLOT_2"].append(ae_metrics_file_additional)
ae_metrics_additional_src, ae_metrics_additional_code = getsource.getsourcefunc(orbital_plots.plot_ae_opt_metrics_with_theory)
additional_plot_code += '\n\n' + ae_metrics_additional_code


# 2. CGPD vs dt
ae_cgpd_dt_file = orbital_plots.plot_ae_cgpd_vs_dt(
    dt_vals=AE_dt_vals,
    cgpd_vals=AE_cgpd_vals,
    out_dir=out_dir,
    filename="ae_cgpd_vs_dt.png",
)
Answers["ls_AEX_CGPDStep_PLOT_2"] = [ae_cgpd_dt_file]

# 3. Total Time vs CGPD
ae_total_cgpd_file = orbital_plots.plot_ae_total_vs_cgpd(
    total_times=AE_total_times,
    cgpd_vals=AE_cgpd_vals,
    out_dir=out_dir,
    filename="ae_total_vs_cgpd.png",
)
Answers["ls_AEX_OptTotalCGPD_PLOT_2"] = [ae_total_cgpd_file]


ae_total_cgpd_additional_file = orbital_plots.plot_ae_total_vs_cgpd_additional(
    total_times=AE_total_times,
    cgpd_vals=AE_cgpd_vals,
    dt_vals=AE_dt_vals,
    out_dir=out_dir,
    filename="ae_totalCPU_and_dt_vs_cgpd.png",
)

Answers["ls_AEX_OptTotalCGPD_PLOT_2"].append(ae_total_cgpd_additional_file)
ae_total_additional_src, ae_total_additional_code = getsource.getsourcefunc(orbital_plots.plot_ae_total_vs_cgpd_additional)
additional_plot_code += '\n\n' + ae_total_additional_code

# --- Analysis Text ---

Answers["ss_AEX_OptStep_OIC_3"] = """
- O1: The plot ae_cgpd_vs_dt.png shows that CGPD decreases rapidly as the time step is reduced from ~170 s to ~100 s 
      (dropping from >300 m to ~40 m). However, for dt < 80 s, the curve flattens into a plateau, maintaining a 
      nearly constant value between 20 m and 30 m regardless of further time step reductions.

- O2: The plot ae_totalCPU_and_cgpd_vs_dt.png (Pareto Front) exhibits an asymptotic behavior: as the CGPD approaches the 
      floor, the Total CPU Time increases non-linearly. Specifically, reducing the time step below 80 s 
      increases the computational cost from ~5 s to >25 s, without yielding visible reductions in CGPD.

- O3: In ae_opt_metrics_vs_dt_additional.png, the average time per call to the optimization function deviates from a strictly
      linear inverse relationship (proportional to 1/dt) and visually evolves as the plotted theoretical reference 
      curve dt^(-0.75).
"""

Answers["ss_AEX_OptStep_OIC_5"] = """
- I1: Based on O1 and O2, the plateau in the CGPD and the non-linear increase in total CPU time indicate that reducing
      the integration step below a certain threshold yields diminishing returns. This behavior occurs because the 
      numerical truncation error (which depends on dt) becomes negligible compared to the physical model mismatch. 
      Consequently, further reducing dt increases the computational cost (as N increases) without improving the 
      solution.

- I2: Based on O3, the trend of Avg Time/Call is not linearly proportional to 1/dt (or N) due to the fixed 
      computational overhead of the Python interpreter and memory allocation. This constant overhead becomes 
      significant relative to the integration time, masking the theoretical O(N) complexity of the optimization algorithm, 
      and resulting in the observed sub-linear scaling (~ N^0.75).
"""

Answers["ss_AEX_OptStep_OIC_6"] = """
Conclusion: The total error is bounded by the fidelity of the dynamical model (physical accuracy floor), which cannot 
            be overcome by numerical refinement alone. Therefore, decreasing the time step (dt) is effective only 
            until the numerical truncation error drops below the physical model mismatch error. The optimal strategy 
            is not to minimize dt indefinitely, but to select the specific step size where numerical precision matches 
            the physical model's accuracy limit, achieving Pareto efficiency between computational cost and trajectory validity.
"""

################################################## CODE EXCELLENCE / AI USE / FEEDBACK ################################

Answers["ss_CEX_Explain_GEN_10"] = """
- Developed a TLE caching and retrieval system with fallback logic, local storage, NORAD/name matching, 
malformed-response handling, and incremental caching.
- The code is structured in files with functions for specific or general tasks.
- Vectorised the CartoKep and KeptoCar transformation, eliminating loops.
- Implemented a layered configuration system (DEFAULTS -> CONFIG.yaml -> CLI arguments) with the `pick()` 
selector.
"""
# - Implemented a robust "Smart Multi-Start" optimization strategy that prioritizes physical intuition (Energy/Period correction) before resorting to random stochastic search.
# - Used `np.geomspace` for the Time Step sensitivity analysis to efficiently cover multiple orders of magnitude of integration precision.
# - Modularized the plotting logic into a separate `orbital_plots` module to keep the main script clean.
# - Implemented a generic objective function capable of handling variable time grids dynamically.

Answers["ss_MDT_AI_GEN_0"] = """|
I used AI to get help with the syntax, the libraries and the yaml file generation. Also, 
plotting functions and secondary functions have been refined by AI for better clarity.
"""

Answers['sf_REG_WORKLOAD_GEN_0'] = 27
Answers['ss_REG_FEEDBACK_GEN_0'] = """|
Interesting assignment. 
"""
################################################## WRITE ANSWER SHEET & SANITY CHECK ###################################
# Source code filename: ls_MDT_AltPlot_SRC_0
# ss_REG_AltPlot_CODE_0
Answers['ls_MDT_AltPlot_SRC_0'] = additional_plot_src
Answers['ss_REG_AltPlot_CODE_0'] = additional_plot_code



with open("answer-sheet.yaml", "w") as f:
    dump(Answers, f, sort_keys=False, default_flow_style=False)

print("\n--- Running Sanity Check ---")
sanity_script = Path(__file__).parent / "answer-sheet-sanity.py"
yaml_file = Path(__file__).parent / "answer-sheet.yaml"

if sanity_script.exists():
    subprocess.run([sys.executable, str(sanity_script), str(yaml_file)])
else:
    print("Sanity check script not found, skipping.")
