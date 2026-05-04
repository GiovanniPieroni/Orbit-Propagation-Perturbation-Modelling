################################################## CONFIGURATION #######################################################
# Libraries and files:
import sys
import numpy as np
import subprocess
import time as pytime
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "modules"))
from modules.utils import ragged_to_matrix
from modules import integrator, plots, errors, gravity_model, frame_transformation, getsource, utils, sgp4_propagator
from debug import pick
from yaml import dump
from dataclasses import dataclass

# Make sure the output directory exists
out_dir = Path("plots")
out_dir.mkdir(exist_ok=True, parents=True)

# Answers dictionary initialization:
Answers = dict()

# Data storage
data_dir = Path("data")
data_dir.mkdir(exist_ok=True, parents=True)

# Language:
Answers['ss_REG_Language_GEN_0'] = "Python"

# Satellite
NORAD = pick("NORAD")
TLE_data = utils.get_tle_data(NORAD, "TLE_GONETSM24.txt")
Answers['ls_MDT_TLE_GEN_0'] = TLE_data
# GONETS-M 24
# 1 54151U 22139B   25329.49630750  .00000033  00000+0  16772-3 0  9999
# 2 54151  82.4951 114.0793 0015785 316.8305  43.1537 12.42859288140348
############################################## Task 1: Update the orbit ###############################################
sat = utils.create_satrec(TLE_data)

mean_motion = sat.no_kozai / 60  # [rad/s]
T = 2 * np.pi / mean_motion  # Orbital period
rev = pick("rev")  # Number of propagated orbital periods
ti = pick("ti")  # Arbitrary
tf = rev * T  # Final time
N = pick("N")  # Number of time steps
# time = np.linspace(ti, tf, N
dt = (tf - ti) / N
time_SGP4 = np.arange(ti, tf + dt, dt, dtype=float)

Answers['sf_MDT_OrbTimeIni_NUM_0'] = ti  # Initial time in seconds
Answers['sf_MDT_OrbTimeEnd_NUM_0'] = tf  # Final time in seconds
Answers['sf_MDT_OrbTimeStep_NUM_0'] = dt  # The time step in seconds

ri, vi = sgp4_propagator.initial_position_velocity(sat, ti)
rf, vf = sgp4_propagator.final_position_velocity(sat, time_SGP4)

Answers['lf_REG_OrbPosIni_NUM_0'] = (ri * 1e3).tolist()  # Initial 3D position
Answers['lf_REG_OrbVelIni_NUM_0'] = (vi * 1e3).tolist()  # Initial 3D velocity
Answers['lf_REG_OrbPosEnd_NUM_1'] = (rf * 1e3).tolist()  # Final 3D position
Answers['lf_REG_OrbVelEnd_NUM_1'] = (vf * 1e3).tolist()  # Final 3D velocity

if pick("debug"):
    print(f"Time interval length: {time_SGP4.size}")
    print(f"ti = {ti}")
    print(f"tf = {tf}")
    print(f"ri = {Answers['lf_REG_OrbPosIni_NUM_0']}")
    print(f"rf = {Answers['lf_REG_OrbPosEnd_NUM_1']}")
    print(f"vi = {Answers['lf_REG_OrbVelIni_NUM_0']}")
    print(f"vf = {Answers['lf_REG_OrbVelEnd_NUM_1']}")

######################################## Task 2: Point mass gravitational model ########################################
Answers['ls_MDT_Grav_SRC_0'], Answers['ss_REG_Grav_CODE_1'] = getsource.getsourcefunc(gravity_model.point_mass)

######################################### Task 3: Euler integrator #####################################################
Answers['ls_MDT_Euler_SRC_0'], Answers['ss_REG_Euler_CODE_1'] = getsource.getsourcefunc(integrator.Euler)

########################################## Task 4: Euler orbit integration #############################################
y_i = np.concatenate((ri, vi))
N_euler = pick("N_euler")
rev_euler = pick("rev_euler")
tf_euler = rev_euler * T
dt_euler = (tf_euler - ti) / N_euler

# time_euler = np.arange(ti, tf_euler +dt_euler, dt_euler, dtype=float)
time_euler = np.linspace(ti, tf_euler, N_euler + 1)

Answers['sf_MDT_EulerTimeEnd_NUM_0'] = tf_euler
Answers['sf_MDT_EulerTimeStep_NUM_0'] = dt_euler

y_euler, elapsed_euler = integrator.Euler(y_i, dt_euler, tf_euler, gravity_model.point_mass, time=True)
elapsed_average_euler = elapsed_euler / N
Answers['sf_REG_EulerTimeTotal_NUM_0.5'] = elapsed_euler  # CPU time
Answers['sf_REG_EulerTimeEff_NUM_0.5'] = elapsed_average_euler  # average computational time per integration step

if elapsed_euler >= pick("max_int_time"):
    print(f"Computation time: {elapsed_euler:.3f} s\n")
    raise RuntimeError(f"Integration with Euler takes longer than {pick('max_int_time')} seconds; increase dt.\n")

Answers['lf_REG_EulerPosIni_NUM_0'] = (y_euler[0, 0:3] * 1e3).tolist()  # Initial 3D position
Answers['lf_REG_EulerVelIni_NUM_0'] = (y_euler[0, 3:6] * 1e3).tolist()  # Initial 3D velocity
Answers['lf_REG_EulerPosEnd_NUM_2'] = (y_euler[-1, 0:3] * 1e3).tolist()  # Final 3D position
Answers['lf_REG_EulerVelEnd_NUM_2'] = (y_euler[-1, 3:6] * 1e3).tolist()  # Final 3D velocity

######################################### Task 5: Plot keplerian elements ##############################################
# SGP4 propagation and transformation to keplerian elements
r_SGP4, v_SGP4 = sgp4_propagator.propagated_position_velocity(sat, time_euler)
y_SGP4 = np.concatenate((r_SGP4, v_SGP4), axis=1)

(a_SGP4, e_SGP4, i_rad_SGP4, i_deg_SGP4,
 Om_rad_SGP4, Om_deg_SGP4, om_rad_SGP4, om_deg_SGP4,
 th_rad_SGP4, th_deg_SGP4, u_deg_SGP4) = frame_transformation.cartesian_to_keplerian(y_SGP4)

# Define the path and name for the file
plot_SGP4_filenames = []
plot_aei_SGP4_filename = out_dir / "a_e_i_SGP4.png"
plot_SGP4_filenames.append(str(plot_aei_SGP4_filename))
plot_uth_SGP4_filename = out_dir / "u_RAAN_SGP4.png"
plot_SGP4_filenames.append(str(plot_uth_SGP4_filename))

if not pick("skip_plots"):
    plots.plot_three_scales(
        time=time_euler.tolist(),
        # Data 1: Semi-major Axis
        data1=a_SGP4,
        label1="Semi-major Axis ($a$)",
        ylabel1="a [km]",
        # Data 2: Eccentricity
        data2=e_SGP4,
        label2="Eccentricity ($e$)",
        ylabel2="e [-]",
        # Data 3: Inclination
        data3=i_deg_SGP4,
        label3="Inclination ($i$)",
        ylabel3="i [deg]",
        # General Settings
        title="Semi-major axis, eccentricity and inclination evolution of SGP4 propagated orbit",
        filename=str(plot_aei_SGP4_filename)
    )

    plots.plot_two_scales(
        time=time_euler.tolist(),
        data1=Om_deg_SGP4,
        label1=r"RAAN ($\Omega$)",
        ylabel1=r"$\Omega$ [deg]",
        data2=u_deg_SGP4,
        label2=r"Argument of latitude ($u$ = $\omega$ + $\theta$)",
        ylabel2=r"u [deg]",

        title="Argument of latitude and RAAN of SGP4 propagated orbit",
        filename=str(plot_uth_SGP4_filename)

    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

Answers['ls_REG_OrbKepPlotFile_PLOT_1'] = plot_SGP4_filenames

(a_euler, e_euler, i_rad_euler, i_deg_euler,
 Om_rad_euler, Om_deg_euler, om_rad_euler, om_deg_euler,
 th_rad_euler, th_deg_euler, u_deg_euler) = frame_transformation.cartesian_to_keplerian(y_euler)

# Define the path and name for the file
plot_euler_filenames = []
plot_aei_euler_filename = out_dir / "a_e_i_Euler.png"
plot_euler_filenames.append(str(plot_aei_euler_filename))
plot_uth_euler_filename = out_dir / "u_RAAN_Euler.png"
plot_euler_filenames.append(str(plot_uth_euler_filename))

# Call the function
if not pick("skip_plots"):
    plots.plot_three_scales(
        time=time_euler.tolist(),
        # Data 1: Semi-major Axis
        data1=a_euler,
        label1="Semi-major Axis ($a$)",
        ylabel1="a [km]",
        # Data 2: Eccentricity
        data2=e_euler,
        label2="Eccentricity ($e$)",
        ylabel2="e [-]",
        # Data 3: Inclination
        data3=i_deg_euler,
        label3="Inclination ($i$)",
        ylabel3="i [deg]",
        # General Settings
        title=f"Semi-major axis, eccentricity and inclination evolution of Euler integrated orbit, dt = {dt_euler:.3f} "
              f"s",
        filename=str(plot_aei_euler_filename)
    )

    plots.plot_two_scales(
        time=time_euler.tolist(),
        data1=Om_deg_euler,
        label1=r"RAAN ($\Omega$)",
        ylabel1=r"$\Omega$ [deg]",
        data2=u_deg_euler,
        label2=r"Argument of latitude ($u$ = $\omega$ + $\theta$)",
        ylabel2=r"u [deg]",

        title="Argument of latitude and RAAN of Euler integrated orbit",
        filename=str(plot_uth_euler_filename)

    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

Answers['ls_REG_EulerKepPlotFile_PLOT_1'] = plot_euler_filenames

# Data save:
np.savetxt("data/SGP4_state.txt", y_SGP4, fmt="%.10e")
np.savetxt("data/SGP4_time.txt", time_euler, fmt="%.10e")

np.savetxt("data/Euler_state.txt", y_euler, fmt="%.10e")
np.savetxt("data/Euler_time.txt", time_euler, fmt="%.10e")


################################### Task 6: Difference between Euler and SGP4 orbits ###################################
@dataclass
class KeplerianIterationResults:
    a_iter_euler: np.ndarray
    e_iter_euler: np.ndarray
    i_iter_euler: np.ndarray
    Om_iter_euler: np.ndarray
    om_iter_euler: np.ndarray
    th_iter_euler: np.ndarray
    u_iter_euler: np.ndarray
    a_iter_SGP4: np.ndarray
    e_iter_SGP4: np.ndarray
    i_iter_SGP4: np.ndarray
    Om_iter_SGP4: np.ndarray
    om_iter_SGP4: np.ndarray
    th_iter_SGP4: np.ndarray
    u_iter_SGP4: np.ndarray


@dataclass
class GPDIterationResults:
    time_iter: np.ndarray
    dt_iter: float
    y_iter: np.ndarray
    y_SGP4_iter: np.ndarray
    gpd: np.ndarray
    y_pos_mag_iter: np.ndarray
    label_dt: str
    keplerian_results: KeplerianIterationResults


@dataclass
class GPDAllIterationsResults:
    dt_GPD: list
    time_GPD: np.ndarray
    GPD_SGP4_Euler: np.ndarray
    y_euler_iter: np.ndarray
    y_pos_mag: np.ndarray
    GPD_label_list: list
    a_euler_list: list
    e_euler_list: list
    i_euler_list: list
    Om_euler_list: list
    om_euler_list: list
    th_euler_list: list
    u_euler_list: list
    a_SGP4_list: list
    e_SGP4_list: list
    i_SGP4_list: list
    Om_SGP4_list: list
    om_SGP4_list: list
    th_SGP4_list: list
    u_SGP4_list: list


def reference_orbit_SGP4(time_grid, dt, tf):
    """Return SGP4 state on the supplied time grid."""
    r_ref, v_ref = sgp4_propagator.propagated_position_velocity(sat, time_grid)
    return np.concatenate((r_ref, v_ref), axis=1)


def reference_orbit_Euler(time_grid, dt, tf):
    """Return Euler-integrated state on the supplied dt and tf."""
    return integrator.Euler(y_i, dt, tf, gravity_model.point_mass)


# To compare them, they must have the same steps, so SGP4 must be re-calculated:
N_GPD = np.round(np.logspace(pick("N_GPD_exp_min"), pick("N_GPD_exp_max"), num=pick("N_GPD_length"))).astype(int) - 1

GPD_sc, GPD_code = getsource.getsourcefunc(errors.compute_GPD)
Answers['ls_MDT_GlobErr_SRC_0'] = GPD_sc
Answers['ss_REG_GlobErr_CODE_2'] = GPD_code

tf_GPD = pick("rev_GPD") * T  # Final time in GPD iteration, always the same

# Call the function to calculate the GPD for every dt considered
# Call the function to calculate the GPD for every dt considered (Euler vs SGP4)
all_results = errors.run_all_GPD_iterations(
    N_GPD=N_GPD,
    ti=ti,
    tf_GPD=tf_GPD,
    y_i=y_i,
    gravity_model=gravity_model,
    integrator_func=integrator.Euler,  # numerical orbit
    reference_orbit_func=reference_orbit_SGP4,  # SGP4 reference
    KeplerianIterationResults=KeplerianIterationResults,
    GPDIterationResults=GPDIterationResults,
    GPDAllIterationsResults=GPDAllIterationsResults,
)

# unpack
dt_GPD = all_results.dt_GPD
time_GPD = all_results.time_GPD
np.savetxt("data/time_GPD.txt", time_GPD, fmt="%.10e")
GPD_SGP4_Euler = all_results.GPD_SGP4_Euler
np.savetxt("data/GPD_SGP4_Euler.txt", GPD_SGP4_Euler, fmt="%.10e")
y_euler_iter = all_results.y_euler_iter
y_pos_mag = all_results.y_pos_mag
GPD_label_list = all_results.GPD_label_list

a_euler_list = all_results.a_euler_list
e_euler_list = all_results.e_euler_list
i_euler_list = all_results.i_euler_list
Om_euler_list = all_results.Om_euler_list
om_euler_list = all_results.om_euler_list
th_euler_list = all_results.th_euler_list
u_euler_list = all_results.u_euler_list

a_SGP4_list = all_results.a_SGP4_list
e_SGP4_list = all_results.e_SGP4_list
i_SGP4_list = all_results.i_SGP4_list
Om_SGP4_list = all_results.Om_SGP4_list
om_SGP4_list = all_results.om_SGP4_list
th_SGP4_list = all_results.th_SGP4_list
u_SGP4_list = all_results.u_SGP4_list

Answers['lf_MDT_GlobErrSteps_NUM_0.5'] = dt_GPD

# Define the path and name for the file
plot_errors_filenames = []

plot_GPD_SGP4_Euler_filename = out_dir / "GPD_SGP4_Euler.png"
plot_errors_filenames.append(str(plot_GPD_SGP4_Euler_filename))
plot_pos_mag_Euler_filename = out_dir / "position_mag_Euler.png"
plot_errors_filenames.append(str(plot_pos_mag_Euler_filename))

if not pick("skip_plots"):
    plots.plot_log(
        time_GPD,
        GPD_SGP4_Euler,
        xlabel="Time [s]",
        ylabel="GPD [km]",
        label_list=GPD_label_list,
        title="GPD between Euler integrated and SGP4 orbit",
        filename=str(plot_GPD_SGP4_Euler_filename),
    )
    plots.plot_log(
        time_GPD,
        y_pos_mag,
        xlabel="Time [s]",
        ylabel="Position magnitude [km]",
        label_list=GPD_label_list,
        title="Magnitude of position at each time",
        filename=str(plot_pos_mag_Euler_filename),
    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

# Euler matrices
a_euler_mat = ragged_to_matrix(a_euler_list)
e_euler_mat = ragged_to_matrix(e_euler_list)
i_euler_mat = ragged_to_matrix([i for i in i_euler_list])
Om_euler_mat = ragged_to_matrix([Om for Om in Om_euler_list])
om_euler_mat = ragged_to_matrix([om for om in om_euler_list])
th_euler_mat = ragged_to_matrix([th for th in th_euler_list])
u_euler_mat = ragged_to_matrix([u for u in u_euler_list])

# SGP4 matrices
a_SGP4_mat = ragged_to_matrix(a_SGP4_list)
e_SGP4_mat = ragged_to_matrix(e_SGP4_list)
i_SGP4_mat = ragged_to_matrix([i for i in i_SGP4_list])
Om_SGP4_mat = ragged_to_matrix([Om for Om in Om_SGP4_list])
om_SGP4_mat = ragged_to_matrix([om for om in om_SGP4_list])
th_SGP4_mat = ragged_to_matrix([th for th in th_SGP4_list])
u_SGP4_mat = ragged_to_matrix([u for u in u_SGP4_list])

da_mat = -a_SGP4_mat + a_euler_mat
de_mat = -e_SGP4_mat + e_euler_mat
di_mat = -i_SGP4_mat + i_euler_mat
dOm_mat = -Om_SGP4_mat + Om_euler_mat
dom_mat = - om_SGP4_mat + om_euler_mat
dth_mat = -th_SGP4_mat + th_euler_mat
du_mat = frame_transformation.wrap_to_180(-u_SGP4_mat + u_euler_mat)

# Define the path and name for the file
plot_a_SGP4_Euler_filename = out_dir / "a_SGP4_Euler.png"
plot_errors_filenames.append(str(plot_a_SGP4_Euler_filename))
plot_e_SGP4_Euler_filename = out_dir / "e_SGP4_Euler.png"
plot_errors_filenames.append(str(plot_e_SGP4_Euler_filename))
plot_i_SGP4_Euler_filename = out_dir / "i_SGP4_Euler.png"
plot_errors_filenames.append(str(plot_i_SGP4_Euler_filename))
plot_Om_SGP4_Euler_filename = out_dir / "Om_SGP4_Euler.png"
plot_errors_filenames.append(str(plot_Om_SGP4_Euler_filename))
plot_u_SGP4_Euler_filename = out_dir / "u_SGP4_Euler.png"
plot_errors_filenames.append(str(plot_u_SGP4_Euler_filename))





if not pick("skip_plots"):
    plots.plot_three_scales_mat(
        time=time_GPD[[2, -1]],
        # Data 1:
        data1=a_euler_mat[[2, -1]],
        label1="a from Euler",
        ylabel1=r"$a_E$ [km]",
        # Data 2:
        data2=a_SGP4_mat[[2, -1]],
        label2="a from SGP4",
        ylabel2=r"$a_{SGP4}$ [km]",
        # Data 3:
        data3=da_mat[[2, -1]],
        label3=r"$\Delta$ a between Euler and SGP4",
        ylabel3=r"$\Delta$ a [km]",
        # General Settings
        title="Semi-major axis",
        filename=str(plot_a_SGP4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD[[2, -1]],
        # Data 1:
        data1=e_euler_mat[[2, -1]],
        label1="e from Euler",
        ylabel1=r"$e_E$ [-]",
        # Data 2:
        data2=e_SGP4_mat[[2, -1]],
        label2="e from SGP4",
        ylabel2=r"$e_{SGP4}$ [-]",
        # Data 3:
        data3=de_mat[[2, -1]],
        label3=r"$\Delta$ e between Euler and SGP4",
        ylabel3=r"$\Delta$ e [-]",
        # General Settings
        title="Eccentricity",
        filename=str(plot_e_SGP4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD[[2, -1]],
        # Data 1:
        data1=i_euler_mat[[2, -1]],
        label1="i from Euler",
        ylabel1=r"$i_E$ [deg]",
        # Data 2:
        data2=i_SGP4_mat[[2, -1]],
        label2="i from SGP4",
        ylabel2=r"$i_{SGP4}$ [deg]",
        # Data 3:
        data3=di_mat[[2, -1]],
        label3=r"$\Delta$ i between Euler and SGP4",
        ylabel3=r"$\Delta$ i [deg]",
        # General Settings
        title="Inclination",
        filename=str(plot_i_SGP4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD[[2, -1]],
        # Data 1:
        data1=Om_euler_mat[[2, -1]],
        label1=r"$\Omega$ from Euler",
        ylabel1=r"$\Omega_E$ [deg]",
        # Data 2:
        data2=Om_SGP4_mat[[2, -1]],
        label2=r"$\Omega$ from SGP4",
        ylabel2=r"$\Omega_{SGP4}$ [deg]",
        # Data 3:
        data3=dOm_mat[[2, -1]],
        label3=r"$\Delta$ $\Omega$ between Euler and SGP4",
        ylabel3=r"$\Delta$ $\Omega$ [deg]",
        # General Settings
        title="RAAN",
        filename=str(plot_Om_SGP4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD[[2, -1]],
        # Data 1: u euler
        data1=u_euler_mat[[2, -1]],
        label1=r"$u$ from Euler",
        ylabel1=r"$u_E$ [deg]",
        # Data 2: u sgp4
        data2=u_SGP4_mat[[2, -1]],
        label2=r"$u$ from SGP4",
        ylabel2=r"$u_{SGP4}$ [deg]",
        # Data 3: delta u
        data3=du_mat[[2, -1]],
        label3=r"$\Delta$ $u$ between Euler and SGP4",
        ylabel3=r"$\Delta$ $u$ [deg]",
        # General Settings
        title="Argument of latitude",
        filename=str(plot_u_SGP4_Euler_filename)
    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

# Report code and filenames of plots
plot_GPD_source, plot_GPD_code = getsource.getsourcefunc(plots.plot_log)
plot_diff_source, plot_diff_code = getsource.getsourcefunc(plots.plot_three_scales_mat)
gpd_diff_plots_source = list(set(plot_GPD_source + plot_diff_source))
gpd_diff_plot_code = plot_GPD_code + "\n\n" + plot_diff_code

Answers['ls_MDT_GlobErrPlot_SRC_0'] = gpd_diff_plots_source
Answers['ss_REG_GlobErrPlot_CODE_2'] = gpd_diff_plot_code


# CGPD calculation:
@dataclass
class CGPDIterationResults:
    time_iter: np.ndarray
    dt_iter: float
    y_iter: np.ndarray
    y_SGP4_iter: np.ndarray
    cgpd: np.ndarray
    y_pos_mag_iter: np.ndarray
    label_dt: str


@dataclass
class CGPDAllIterationsResults:
    dt_CGPD: list
    time_CGPD: np.ndarray
    CGPD_SGP4_Euler: np.ndarray
    y_euler_iter: np.ndarray
    y_pos_mag: np.ndarray
    CGPD_label_list: list


N_CGPD = np.round(np.logspace(3, pick("N_CGPD_exp_max"), num=pick("N_CGPD_length"))).astype(int) - 1
# tf_GPD = 2 * T  # Final time in GPD iteration, always the same
# N_CGPD = N_GPD
tf_CGPD = tf_GPD

# Call the function to calculate the CGPD for every dt considered
all_results_CGPD = errors.run_all_CGPD_iterations(
    N_CGPD=N_CGPD,
    ti=ti,
    tf_GPD=tf_GPD,
    y_i=y_i,
    gravity_model=gravity_model,
    integrator_func=integrator.Euler,
    reference_orbit_func=reference_orbit_SGP4,
    CGPDIterationResults=CGPDIterationResults,
    CGPDAllIterationsResults=CGPDAllIterationsResults,
)

# unpack
dt_CGPD = all_results_CGPD.dt_CGPD
time_CGPD = all_results_CGPD.time_CGPD
np.savetxt("data/time_CGPD.txt", time_CGPD, fmt="%.10e")

CGPD_SGP4_Euler = all_results_CGPD.CGPD_SGP4_Euler
np.savetxt("data/CGPD_SGP4_Euler.txt", CGPD_SGP4_Euler, fmt="%.10e")
y_euler_iter_CGPD = all_results_CGPD.y_euler_iter
y_pos_mag_CGPD = all_results_CGPD.y_pos_mag
CGPD_label_list = all_results_CGPD.CGPD_label_list

Answers['lf_MDT_CumGlobErrSteps_NUM_0.5'] = dt_CGPD  # Step sizes considered for CGPD calculation

# Report code and filenames of plots
plot_CGPD_filenames = []
plot_CGPD_SGP4_Euler_filename = out_dir / "CGPD_SGP4_Euler.png"
plot_CGPD_filenames.append(str(plot_CGPD_SGP4_Euler_filename))

dt_CGPD = np.array(dt_CGPD)
idx_dt_CGPD = np.argsort(dt_CGPD)[::-1]

dt_CGPD_sorted = dt_CGPD[idx_dt_CGPD]  # dt from maximum to minimum

if not pick("skip_plots"):
    plots.plot_log(
        dt_CGPD_sorted,
        CGPD_SGP4_Euler[idx_dt_CGPD],
        xlabel="dt [s]",
        ylabel="CGPD [km]",
        label_list=["CGPD for different dt"],
        title="CGPD between Euler integrated and SGP4 orbit",
        filename=str(plot_CGPD_SGP4_Euler_filename),
        scatter=True,
        x_inverted=True
    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

# Reporting sourcecode, code, and filenames
plot_CGPD_source, plot_CGPD_code = getsource.getsourcefunc(plots.plot_log)
CGPD_source, CGPD_code = getsource.getsourcefunc(errors.compute_CGPD)
CGPD_computation_and_plot_source = list(set(plot_CGPD_source + CGPD_source))
CGPD_computation_and_plot_code = plot_CGPD_code + "\n\n" + CGPD_code
Answers['ls_MDT_CumGlobErrPlot_SRC_0'] = CGPD_computation_and_plot_source
Answers['ss_REG_CumGlobErrPlot_CODE_2'] = CGPD_computation_and_plot_code

Answers['ls_REG_CumGlobErr_PLOT_7'] = plot_CGPD_filenames


## Energy, angular momentum, and position plot
r_euler_iter = y_euler_iter[:, 24:27]
v_euler_iter = y_euler_iter[:, 27:30]

E = (np.linalg.norm(v_euler_iter, axis=1))**2/2 - pick("mu")/np.linalg.norm(r_euler_iter, axis=1)
h = np.linalg.cross(r_euler_iter, v_euler_iter)
h_mag = np.linalg.norm(h, axis=1)

plot_E_h_pos = out_dir / "E_h_pos.png"
plot_errors_filenames.append(str(plot_E_h_pos))

# plot_E_h_Euler.png
if not pick("skip_plots"):
    plots.plot_three_scales(
        time=time_GPD[4, :],
        # Data 1:
        data1=E,
        label1="Energy",
        ylabel1=r"$E$ [$km^2/s^2$] ",
        # Data 2:
        data2=h_mag,
        label2="Angular momentum",
        ylabel2=r"$h$ [$km^2/s$]",
        # Data 3:
        data3=y_pos_mag[4, :],
        label3="Position magnitude",
        ylabel3="Position magnitude [km]",
        # General Settings
        title=f"Energy, angular momentum, and position magnitude. dt = {dt_GPD[-1]} s",
        filename=str(plot_E_h_pos),

        ax1_big=True
    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

Answers['ls_REG_GlobErr_PLOT_7'] = plot_errors_filenames






############################### Task 7: Analyse difference between Euler and SGP4 orbits ###############################
Answers['ss_REG_OrbObs_OIC_10'] = """|
- O1: Plot "a_e_i_Euler.png" shows that the semi-major axis linearly increases of approximately 500 km, the inclination 
remains constant, and the eccentricity exhibits near-periodic oscillations.
- O2: Plot "Om_SGP4_Euler.png" shows that the RAAN from the Euler integrator is constant and the same for every dt used,
whereas SGP4 orbit's RAAN shows both a periodic behaviour and a negative drift.
- O3: From the plot "GPD_SGP4_Euler.png", the GPD increases with time and, on the log–log axes, each curve is close to a
straight line, with smaller GPD values for smaller time steps. 
- O4: The plot "E_h_pos.png" shows that both specific energy and specific angular momentum drift linearly during the 
three orbital periods represented: the specific energy becomes less negative (its magnitude decreases) and the specific
angular momentum increases.
"""

# - O1: Plot "a_e_i_Euler.png" shows semi-major axis, eccentricity and inclination of the orbit integrated with the Euler
# method. The semi-major axis linearly increases of approximately 500 km, the inclination remains constant, and the
# eccentricity exhibits near-periodic oscillations. From the plot "i_SGP4_Euler.png" we observe that the inclination of
# the SGP4 orbit shows a near-sinusoidal pattern.
#
# - O2: Plot "Om_SGP4_Euler.png" shows that the RAAN from the Euler integrator is constant and the same for every dt used,
# whereas SGP4 orbit's RAAN shows both a periodic behaviour and a negative drift.
# .
#
# - O3: From the plot "GPD_SGP4_Euler.png", the GPD increases with time and, on the log–log axes, each curve is close to a
# straight line, with smaller GPD values for smaller time steps. Plot "CGPD_SGP4_Euler.png" shows that the Cumulative
# Global Position Difference at the final time between the Euler orbit and the SGP4 orbit decreases almost linearly with
# decreasing time steps on the log–log plot: with the largest time step it is of the order of 10^4 km, whereas with the
# smallest time step it is of the order of a few 10^2 km.

Answers['ss_REG_OrbInt_OIC_15'] = """ |
- I1: The drift of the semi-major axis observed in O1 is caused by the Euler integrator violating the conservation of orbital
energy, as seen in O4. Since E = -mu / (2a), the semi-major axis must increase as the specific mechanical energy 
becomes less negative. This energy drift arises because the Euler integrator introduces local truncation errors that 
accumulate over time, so neither energy nor angular momentum remain constant, although they should be conserved in the two-body
(point-mass) problem. 

- I2: From O3, the nearly straight line of CGPD versus time step on the log–log plot indicates a power-law dependence
of the form CGPD directly proportional to dt, consistent with the first-order global truncation error of the Euler 
method. Reducing the time step therefore decreases the cumulative position error approximately proportionally to dt. 
"""
# Moreover, as observed in O1 and O2, inclination and RAAN remain constant over the integration period, because they
# are not coupled to the orbital energy and because the Euler integration includes only the acceleration due to a point-mass,
# not taking into account any perturbation (of any kind).

Answers['ss_REG_OrbCon_OIC_23'] = """|
- Integrating with a first-order method and using a point-mass gravitational model yields unreliable results, with 
respect to those obtained with SGP4 propagation, for two main reasons: truncation error and model discrepancy. The
truncation error can be mitigated, but the linear reduction of CGPD with smaller time steps implies that an extremely small
time step (near machine precision) is required. On the other hand, the constant RAAN and inclination observed in the 
Euler integration prove that the point-mass model is fundamentally insufficient to reproduce the SGP4 dynamics, 
regardless of the integration precision.
"""

############################### Task AE 1: Implement an alternative integration algorithm ##############################
Answers['ss_AEX_AltName_GEN_0.1'] = "RK4"
Answers['ls_AEX_Alt_SRC_0.1'], Answers['ss_AEX_Alt_CODE_2'] = getsource.getsourcefunc(integrator.RK4)

##################################### Task AE 2: Perform the orbit integration #########################################
y_i_alt = np.concatenate((ri, vi))
N_RK4 = pick("N_RK4")
rev_RK4 = pick("rev_RK4")
tf_RK4 = rev_RK4 * T
dt_RK4 = (tf_RK4 - ti) / N_RK4
tf_RK4 = tf_RK4
# time_RK4 = np.arange(ti, tf_RK4 +dt_RK4, dt_RK4, dtype=float)
time_RK4 = np.linspace(ti, tf_RK4, N_RK4 + 1)

Answers['sf_AEX_AltTimeEnd_NUM_0.1'] = tf_RK4
Answers['sf_AEX_AltTimeStep_NUM_0.1'] = dt_RK4

start = pytime.time()
y_RK4 = integrator.RK4(y_i, dt_RK4, tf_RK4, gravity_model.point_mass)
elapsed = pytime.time() - start
elapsed_avrg = elapsed / N
Answers['sf_AEX_AtlTimeTotal_NUM_0.5'] = elapsed  # CPU time
Answers['sf_AEX_AltTimeEff_NUM_0.5'] = elapsed_avrg  # average computational time per integration step

if elapsed >= pick("max_int_time"):
    print(f"Computation time: {elapsed:.3f} s\n")
    raise RuntimeError(f"Integration with RK4 takes longer than {pick("max_int_time")} seconds; increase dt.\n")

Answers['lf_AEX_AltPosIni_NUM_0'] = (y_RK4[0, 0:3] * 1e3).tolist()  # Initial 3D position
Answers['lf_AEX_AltVelIni_NUM_0'] = (y_RK4[0, 3:6] * 1e3).tolist()  # Initial 3D velocity
Answers['lf_AEX_AltPosEnd_NUM_2'] = (y_RK4[-1, 0:3] * 1e3).tolist()  # Final 3D position
Answers['lf_AEX_AltVelEnd_NUM_2'] = (y_RK4[-1, 3:6] * 1e3).tolist()  # Final 3D velocity


######################################## Task AE 3: Assess the differences #############################################
@dataclass
class KeplerianIterationResults_alt:
    a_iter_euler: np.ndarray
    e_iter_euler: np.ndarray
    i_iter_euler: np.ndarray
    Om_iter_euler: np.ndarray
    om_iter_euler: np.ndarray
    th_iter_euler: np.ndarray
    u_iter_euler: np.ndarray
    a_iter_SGP4: np.ndarray
    e_iter_SGP4: np.ndarray
    i_iter_SGP4: np.ndarray
    Om_iter_SGP4: np.ndarray
    om_iter_SGP4: np.ndarray
    th_iter_SGP4: np.ndarray
    u_iter_SGP4: np.ndarray


@dataclass
class GPDIterationResults_alt:
    time_iter: np.ndarray
    dt_iter: float
    y_iter: np.ndarray
    y_SGP4_iter: np.ndarray  # reference orbit
    gpd: np.ndarray
    y_pos_mag_iter: np.ndarray
    label_dt: str
    keplerian_results: KeplerianIterationResults_alt


@dataclass
class GPDAllIterationsResults_alt:
    dt_GPD: list
    time_GPD: np.ndarray
    GPD_SGP4_Euler: np.ndarray  # GPD between alt integrator and reference
    y_euler_iter: np.ndarray  # states of alt integrator
    y_pos_mag: np.ndarray
    GPD_label_list: list
    a_euler_list: list
    e_euler_list: list
    i_euler_list: list
    Om_euler_list: list
    om_euler_list: list
    th_euler_list: list
    u_euler_list: list
    a_SGP4_list: list
    e_SGP4_list: list
    i_SGP4_list: list
    Om_SGP4_list: list
    om_SGP4_list: list
    th_SGP4_list: list
    u_SGP4_list: list


# Call the function to calculate the GPD for every dt considered
alt_integrator = integrator.RK4


N_GPD_alt = (np.round(
    np.logspace(
        pick("N_GPD_exp_min_alt"),
        pick("N_GPD_exp_max_alt"),
        num=pick("N_GPD_length_alt"),
    )
).astype(int) - 1)

tf_GPD_alt = pick("rev_GPD_alt") * T

# RK4 vs Euler
all_results_alt_Euler = errors.run_all_GPD_iterations(
    N_GPD=N_GPD_alt,
    ti=ti,
    tf_GPD=tf_GPD_alt,
    y_i=y_i_alt,
    gravity_model=gravity_model,
    integrator_func=alt_integrator,  # RK4 numerical orbit
    reference_orbit_func=reference_orbit_Euler,  # Euler reference
    KeplerianIterationResults=KeplerianIterationResults_alt,
    GPDIterationResults=GPDIterationResults_alt,
    GPDAllIterationsResults=GPDAllIterationsResults_alt,
)
# RK4 vs SGP4
all_results_alt_SGP4 = errors.run_all_GPD_iterations(
    N_GPD=N_GPD_alt,
    ti=ti,
    tf_GPD=tf_GPD_alt,
    y_i=y_i_alt,
    gravity_model=gravity_model,
    integrator_func=alt_integrator,             # RK4 numerical orbit
    reference_orbit_func=reference_orbit_SGP4,  # SGP4 reference
    KeplerianIterationResults=KeplerianIterationResults_alt,
    GPDIterationResults=GPDIterationResults_alt,
    GPDAllIterationsResults=GPDAllIterationsResults_alt,
)

# unpack
dt_GPD_alt_Euler = all_results_alt_Euler.dt_GPD
time_GPD_alt_Euler = all_results_alt_Euler.time_GPD
GPD_alt_ref_Euler = all_results_alt_Euler.GPD_SGP4_Euler
np.savetxt("data/GPD_alt_vs_ref_Euler.txt", GPD_alt_ref_Euler, fmt="%.10e")
y_alt_iter_Euler = all_results_alt_Euler.y_euler_iter
y_pos_mag_alt_Euler = all_results_alt_Euler.y_pos_mag
GPD_label_list_alt_Euler = all_results_alt_Euler.GPD_label_list

a_alt_list_Euler = all_results_alt_Euler.a_euler_list
e_alt_list_Euler = all_results_alt_Euler.e_euler_list
i_alt_list_Euler = all_results_alt_Euler.i_euler_list
Om_alt_list_Euler = all_results_alt_Euler.Om_euler_list
om_alt_list_Euler = all_results_alt_Euler.om_euler_list
th_alt_list_Euler = all_results_alt_Euler.th_euler_list
u_alt_list_Euler = all_results_alt_Euler.u_euler_list

a_ref_list_Euler = all_results_alt_Euler.a_SGP4_list
e_ref_list_Euler = all_results_alt_Euler.e_SGP4_list
i_ref_list_Euler = all_results_alt_Euler.i_SGP4_list
Om_ref_list_Euler = all_results_alt_Euler.Om_SGP4_list
om_ref_list_Euler = all_results_alt_Euler.om_SGP4_list
th_ref_list_Euler = all_results_alt_Euler.th_SGP4_list
u_ref_list_Euler = all_results_alt_Euler.u_SGP4_list



# unpack
dt_GPD_alt_SGP4 = all_results_alt_SGP4.dt_GPD
time_GPD_alt_SGP4 = all_results_alt_SGP4.time_GPD
GPD_alt_ref_SGP4 = all_results_alt_SGP4.GPD_SGP4_Euler
np.savetxt("data/GPD_alt_vs_ref_SGP4.txt", GPD_alt_ref_SGP4, fmt="%.10e")
y_alt_iter_SGP4 = all_results_alt_SGP4.y_euler_iter
y_pos_mag_alt_SGP4 = all_results_alt_SGP4.y_pos_mag
GPD_label_list_alt_SGP4 = all_results_alt_SGP4.GPD_label_list

a_alt_list_SGP4 = all_results_alt_SGP4.a_euler_list
e_alt_list_SGP4 = all_results_alt_SGP4.e_euler_list
i_alt_list_SGP4 = all_results_alt_SGP4.i_euler_list
Om_alt_list_SGP4 = all_results_alt_SGP4.Om_euler_list
om_alt_list_SGP4 = all_results_alt_SGP4.om_euler_list
th_alt_list_SGP4 = all_results_alt_SGP4.th_euler_list
u_alt_list_SGP4 = all_results_alt_SGP4.u_euler_list

a_ref_list_SGP4 = all_results_alt_SGP4.a_SGP4_list
e_ref_list_SGP4 = all_results_alt_SGP4.e_SGP4_list
i_ref_list_SGP4 = all_results_alt_SGP4.i_SGP4_list
Om_ref_list_SGP4 = all_results_alt_SGP4.Om_SGP4_list
om_ref_list_SGP4 = all_results_alt_SGP4.om_SGP4_list
th_ref_list_SGP4 = all_results_alt_SGP4.th_SGP4_list
u_ref_list_SGP4 = all_results_alt_SGP4.u_SGP4_list



# Alternative integrator (RK4)
a_alt_mat = ragged_to_matrix(a_alt_list_Euler)
e_alt_mat = ragged_to_matrix(e_alt_list_Euler)
i_alt_mat = ragged_to_matrix([i for i in i_alt_list_Euler])
Om_alt_mat = ragged_to_matrix([Om for Om in Om_alt_list_Euler])
om_alt_mat = ragged_to_matrix([om for om in om_alt_list_Euler])
th_alt_mat = ragged_to_matrix([th for th in th_alt_list_Euler])
u_alt_mat = ragged_to_matrix([u for u in u_alt_list_Euler])

# Euler matrices (reference in RK4 vs Euler)
a_euler_mat_alt = ragged_to_matrix(a_ref_list_Euler)
e_euler_mat_alt = ragged_to_matrix(e_ref_list_Euler)
i_euler_mat_alt = ragged_to_matrix([i for i in i_ref_list_Euler])
Om_euler_mat_alt = ragged_to_matrix([Om for Om in Om_ref_list_Euler])
om_euler_mat_alt = ragged_to_matrix([om for om in om_ref_list_Euler])
th_euler_mat_alt = ragged_to_matrix([th for th in th_ref_list_Euler])
u_euler_mat_alt = ragged_to_matrix([u for u in u_ref_list_Euler])

# SGP4 matrices (reference in RK4 vs SGP4)
a_SGP4_mat_alt = ragged_to_matrix(a_ref_list_SGP4)
e_SGP4_mat_alt = ragged_to_matrix(e_ref_list_SGP4)
i_SGP4_mat_alt = ragged_to_matrix([i for i in i_ref_list_SGP4])
Om_SGP4_mat_alt = ragged_to_matrix([Om for Om in Om_ref_list_SGP4])
om_SGP4_mat_alt = ragged_to_matrix([om for om in om_ref_list_SGP4])
th_SGP4_mat_alt = ragged_to_matrix([th for th in th_ref_list_SGP4])
u_SGP4_mat_alt = ragged_to_matrix([u for u in u_ref_list_SGP4])


da_RK4_Euler = a_alt_mat - a_euler_mat_alt
de_RK4_Euler = e_alt_mat - e_euler_mat_alt
di_RK4_Euler = i_alt_mat - i_euler_mat_alt
dOm_RK4_Euler = Om_alt_mat - Om_euler_mat_alt
dom_RK4_Euler = om_alt_mat - om_euler_mat_alt
dth_RK4_Euler = th_alt_mat - th_euler_mat_alt
du_RK4_Euler = frame_transformation.wrap_to_180(u_alt_mat - u_euler_mat_alt)

da_RK4_SGP4 = a_alt_mat - a_SGP4_mat_alt
de_RK4_SGP4 = e_alt_mat - e_SGP4_mat_alt
di_RK4_SGP4 = i_alt_mat - i_SGP4_mat_alt
dOm_RK4_SGP4 = Om_alt_mat - Om_SGP4_mat_alt
dom_RK4_SGP4 = om_alt_mat - om_SGP4_mat_alt
dth_RK4_SGP4 = th_alt_mat - th_SGP4_mat_alt
du_RK4_SGP4 = frame_transformation.wrap_to_180(u_alt_mat - u_SGP4_mat_alt)


# Define the path and name for the file
plot_errors_filenames_alt = []

plot_GPD_alt_ref_Euler_filename = out_dir / "GPD_RK4_Euler.png"
plot_errors_filenames_alt.append(str(plot_GPD_alt_ref_Euler_filename))
plot_GPD_alt_ref_SGP4_filename = out_dir / "GPD_RK4_SGP4.png"
plot_errors_filenames_alt.append(str(plot_GPD_alt_ref_SGP4_filename))


plot_a_RK4_Euler_filename = out_dir / "a_RK4_Euler.png"
plot_errors_filenames_alt.append(str(plot_a_RK4_Euler_filename))
plot_e_RK4_Euler_filename = out_dir / "e_RK4_Euler.png"
plot_errors_filenames_alt.append(str(plot_e_RK4_Euler_filename))
plot_i_RK4_Euler_filename = out_dir / "i_RK4_Euler.png"
plot_errors_filenames_alt.append(str(plot_i_RK4_Euler_filename))
plot_u_RK4_Euler_filename = out_dir / "u_RK4_Euler.png"
plot_errors_filenames_alt.append(str(plot_u_RK4_Euler_filename))
plot_Om_RK4_Euler_filename = out_dir / "Om_RK4_Euler.png"
plot_errors_filenames_alt.append(str(plot_Om_RK4_Euler_filename))

if not pick("skip_plots"):
    plots.plot_log(
        time_GPD_alt_Euler,
        GPD_alt_ref_Euler,
        xlabel="Time [s]",
        ylabel="GPD [km]",
        label_list=GPD_label_list_alt_Euler,
        title="GPD between alternative integrated orbit and Euler orbit",
        filename=str(plot_GPD_alt_ref_Euler_filename),
    )
    plots.plot_log(
        time_GPD_alt_SGP4,
        GPD_alt_ref_SGP4,
        xlabel="Time [s]",
        ylabel="GPD [km]",
        label_list=GPD_label_list_alt_SGP4,
        title="GPD between alternative integrated orbit and SGP4 orbit",
        filename=str(plot_GPD_alt_ref_SGP4_filename),
    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

if not pick("skip_plots"):
    plots.plot_three_scales_mat(
        time=time_GPD_alt_Euler,
        # Data 1:
        data1=a_alt_mat,
        label1="a from RK4",
        ylabel1=r"$a_{RK4}$ [km]",
        # Data 2:
        data2=a_euler_mat_alt,
        label2="a from Euler",
        ylabel2=r"$a_{E}$ [km]",
        # Data 3:
        data3=da_RK4_Euler,
        label3=r"$\Delta$ a between RK4 and Euler",
        ylabel3=r"$\Delta$ a [km]",
        # General Settings
        title="Semi-major axis - RK4 and Euler orbits",
        filename=str(plot_a_RK4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD_alt_Euler,
        # Data 1:
        data1=e_alt_mat,
        label1="e from RK4",
        ylabel1=r"$e_{RK4}$ [-]",
        # Data 2:
        data2=e_euler_mat_alt,
        label2="e from Euler",
        ylabel2=r"$e_{E}$ [-]",
        # Data 3:
        data3=de_RK4_Euler,
        label3=r"$\Delta$ e between RK4 and Euler",
        ylabel3=r"$\Delta$ e [-]",
        # General Settings
        title="Eccentricity - RK4 and Euler orbits",
        filename=str(plot_e_RK4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD_alt_Euler,
        # Data 1:
        data1=i_alt_mat,
        label1="i from RK4",
        ylabel1=r"$i_{RK4}$ [deg]",
        # Data 2:
        data2=i_euler_mat_alt,
        label2="i from Euler",
        ylabel2=r"$i_{E}$ [deg]",
        # Data 3:
        data3=di_RK4_Euler,
        label3=r"$\Delta$ i between RK4 and Euler",
        ylabel3=r"$\Delta$ i [deg]",
        # General Settings
        title="Inclination - RK4 and Euler orbits",
        filename=str(plot_i_RK4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD_alt_Euler,
        # Data 1:
        data1=u_alt_mat,
        label1="u from RK4",
        ylabel1=r"$u_{RK4}$ [deg]",
        # Data 2:
        data2=u_euler_mat_alt,
        label2="u from Euler",
        ylabel2=r"$u_{E}$ [deg]",
        # Data 3:
        data3=du_RK4_Euler,
        label3=r"$\Delta$ u between RK4 and Euler",
        ylabel3=r"$\Delta$ u [deg]",
        # General Settings
        title="Argument of Latitude - RK4 and Euler orbits",
        filename=str(plot_u_RK4_Euler_filename)
    )
    plots.plot_three_scales_mat(
        time=time_GPD_alt_Euler,
        # Data 1:
        data1=Om_alt_mat,
        label1=r"$\Omega$ from RK4",
        ylabel1=r"$\Omega_{RK4}$ [deg]",
        # Data 2:
        data2=Om_euler_mat_alt,
        label2=r"$\Omega$ from Euler",
        ylabel2=r"$\Omega_{E}$ [deg]",
        # Data 3:
        data3=dOm_RK4_Euler,
        label3=r"$\Delta$ $\Omega$ between RK4 and Euler",
        ylabel3=r"$\Delta$ $\Omega$ [deg]",
        # General Settings
        title="RAAN - RK4 and Euler orbits",
        filename=str(plot_Om_RK4_Euler_filename)
    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")


# Reports data
Answers['lf_AEX_AltGlobErrSteps_NUM_0.1'] = dt_GPD_alt_Euler

Answers['ls_AEX_AltGlobErr_PLOT_2.2'] = plot_errors_filenames_alt




# CGPD calculation:
@dataclass
class CGPDIterationResults:
    time_iter: np.ndarray
    dt_iter: float
    y_iter: np.ndarray
    y_SGP4_iter: np.ndarray
    cgpd: np.ndarray
    y_pos_mag_iter: np.ndarray
    label_dt: str


@dataclass
class CGPDAllIterationsResults:
    dt_CGPD: list
    time_CGPD: np.ndarray
    CGPD_SGP4_Euler: np.ndarray
    y_euler_iter: np.ndarray
    y_pos_mag: np.ndarray
    CGPD_label_list: list


N_CGPD_alt = np.round(np.logspace(pick("N_CGPD_exp_min_alt"), pick("N_CGPD_exp_max_alt"), num=pick("N_CGPD_length_alt"))).astype(int) - 1
tf_CGPD_alt = pick("rev_GPD_alt") * T  # Final time in GPD iteration, always the same

# Call the function to calculate the CGPD for every dt considered
all_results_CGPD_alt_SGP4 = errors.run_all_CGPD_iterations(
    N_CGPD=N_CGPD_alt,
    ti=ti,
    tf_GPD=tf_CGPD_alt,
    y_i=y_i,
    gravity_model=gravity_model,
    integrator_func=integrator.RK4,
    reference_orbit_func=reference_orbit_SGP4,
    CGPDIterationResults=CGPDIterationResults,
    CGPDAllIterationsResults=CGPDAllIterationsResults,
)
all_results_CGPD_alt_Euler = errors.run_all_CGPD_iterations(
    N_CGPD=N_CGPD_alt,
    ti=ti,
    tf_GPD=tf_CGPD_alt,
    y_i=y_i,
    gravity_model=gravity_model,
    integrator_func=integrator.Euler,
    reference_orbit_func=reference_orbit_SGP4,
    CGPDIterationResults=CGPDIterationResults,
    CGPDAllIterationsResults=CGPDAllIterationsResults,
)

# unpack
dt_CGPD_alt_SGP4 = all_results_CGPD_alt_SGP4.dt_CGPD
time_CGPD_alt_SGP4 = all_results_CGPD_alt_SGP4.time_CGPD
CGPD_RK4_SGP4 = all_results_CGPD_alt_SGP4.CGPD_SGP4_Euler
np.savetxt("data/CGPD_RK4_SGP4.txt", CGPD_RK4_SGP4, fmt="%.10e")
y_SGP4_iter_CGPD = all_results_CGPD_alt_SGP4.y_euler_iter
CGPD_label_list_SGP4 = all_results_CGPD_alt_SGP4.CGPD_label_list

# unpack
dt_CGPD_alt_Euler = all_results_CGPD_alt_Euler.dt_CGPD
time_CGPD_alt_Euler= all_results_CGPD_alt_Euler.time_CGPD
CGPD_RK4_Euler = all_results_CGPD_alt_Euler.CGPD_SGP4_Euler
np.savetxt("data/CGPD_RK4_Euler.txt", CGPD_RK4_Euler, fmt="%.10e")
y_Euler_iter_CGPD = all_results_CGPD_alt_Euler.y_euler_iter
CGPD_label_list_Euler = all_results_CGPD_alt_Euler.CGPD_label_list

# Report code and filenames of plots
plot_CGPD_alt_filenames = []
plot_CGPD_RK4_SGP4_filename = out_dir / "CGPD_RK4_SGP4.png"
plot_CGPD_alt_filenames.append(str(plot_CGPD_RK4_SGP4_filename))
plot_CGPD_RK4_Euler_filename = out_dir / "CGPD_RK4_Euler.png"
plot_CGPD_alt_filenames.append(str(plot_CGPD_RK4_Euler_filename))



dt_CGPD_alt = np.array(dt_CGPD_alt_SGP4)
idx_dt_CGPD_alt = np.argsort(dt_CGPD_alt)[::-1]

dt_CGPD_sorted_alt = dt_CGPD_alt[idx_dt_CGPD_alt]  # dt from maximum to minimum

if not pick("skip_plots"):
    plots.plot_log(
        dt_CGPD_sorted_alt,
        CGPD_RK4_SGP4[idx_dt_CGPD_alt],
        xlabel="dt [s]",
        ylabel="CGPD [km]",
        label_list=["CGPD for different dt"],
        title="CGPD between RK4 integrated and SGP4 orbit",
        filename=str(plot_CGPD_RK4_SGP4_filename),
        scatter=True,
        x_inverted=True
    )
    plots.plot_log(
        dt_CGPD_sorted_alt,
        CGPD_RK4_Euler[idx_dt_CGPD_alt],
        xlabel="dt [s]",
        ylabel="CGPD [km]",
        label_list=["CGPD for different dt"],
        title="CGPD between RK4 integrated and Euler orbit",
        filename=str(plot_CGPD_RK4_Euler_filename),
        scatter=True,
        x_inverted=True
    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")




# Report data
Answers['lf_AEX_AltCumGlobErrSteps_NUM_0.1'] = dt_CGPD_alt.tolist()

Answers['ls_AEX_AltCumGlobErr_PLOT_2.2'] = plot_CGPD_alt_filenames


######################################## Task AE 4: Analyse the differences ############################################
Answers['ss_AEX_AltObs_OIC_1.5'] = """|
- O1: Plot "GPD_RK4_Euler.png" shows that the GPD between RK4 and Euler grows to approximately 10^4 km for both the used 
time steps, and it is lower for a smaller dt. In "GPD_RK4_SGP4.png", the GPD between RK4 and SGP4 reaches ~100 km and is 
essentially identical for both step sizes, although for the first ~20 s the GPD for dt = 20.876 s is bigger.
- O2: In "Om_RK4_Euler.png" and "i_RK4_Euler.png", the RAAN and inclination differences stay at ~10^(-13). 
In "e_RK4_Euler.png", RK4 eccentricity remains effectively constant, with oscillations of amplitude ~4e-9 for the 
greatest dt used, whereas Euler orbit's eccentricity oscillates with a maximum amplitude of approximately 0.05 for the
greatest time step.
- O3: "CGPD_RK4_Euler.png" shows that the CGPD decreases with decreasing dt: values span from more than 10^4 km  down to 
less than ~4*10³ km. "CGPD_RK4_SGP4.png" instead plateaus near ~50 km for small dts, while the lowest value is obtained
for the largest dt, at ~42 m.
"""


Answers['ss_AEX_AltInt_OIC_2.5'] = """|
- I1: The strong growth of GPD in Euler cases originates from first-order truncation errors accumulating in the state, while
 RK4 remains consistent across the considered dts; the negligible differences in RAAN and inclination confirm that, 
 using a point-mass gravity model without any perturbation, these two keplerian elements are not affected by the 
 integration method chosen. By contrary, the differences in eccentricity between Euler and RK4 orbit show that the Euler
 method introduces a numerical oscillation and RK4 doesn't.
- I2: The CGPD trends reflect convergence of RK4 to the two-body solution with a point-mass model and no perturbations. Decreasing dt 
reduces RK4’s own numerical noise, but cannot diminish the model discrepancy with SGP4. The small dip at the coarsest 
dt is a local numerical artefact rather than an indicator of improved physical accuracy.
"""


Answers['ss_AEX_AltCon_OIC_4'] = """|
- To reduce the truncation error, it is more effective to increase the order of the integration model than to decrease 
the integration time. For this reason, an explicit first-order method is reliable only for time steps close to machine 
precision, whereas for a higher-order method, a much larger time step is sufficient to achieve a convergent solution.
"""

# Final answers:
Answers['ss_CEX_Explain_GEN_10'] = """|
- Developed a robust TLE caching and retrieval system with fallback logic, local storage, NORAD/name matching, 
malformed-response handling, and incremental caching.
- Vectorised the CartoKep and KeptoCar transformation, eliminating loops.
- Built a reusable plotting framework including matrix-based plotting with gradient colours, and generic plotting interfaces.
- Adopted dataclasses for all iteration and storage structures, producing clean, typed, self-documenting data.
- Implemented a layered configuration system (DEFAULTS -> CONFIG.yaml -> CLI arguments) with the `pick()` 
selector.
"""

Answers['ss_MDT_AI_GEN_0'] = """|
I used AI to get help with the syntax, the libraries and the yaml file generation. Also, 
plotting functions and secondary functions have been refined by AI for better clarity.
"""

Answers['sf_REG_WORKLOAD_GEN_0'] = 10
Answers['ss_REG_FEEDBACK_GEN_0'] = """|
No feedback
"""
########################################################################################################################
# YAML file writing #
with open("answer-sheet.yaml", "w") as f:
    dump(Answers, f, sort_keys=False, default_flow_style=False)

print("\n--- Running Sanity Check ---")

sanity_script = Path(__file__).parent / "answer-sheet-sanity.py"
yaml_file = Path(__file__).parent / "answer-sheet.yaml"

subprocess.run([sys.executable, str(sanity_script), str(yaml_file)])
