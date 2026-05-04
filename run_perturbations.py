################################################## CONFIGURATION #######################################################
# Libraries and files:
import numpy as np
from debug import pick
from yaml import dump
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "modules"))
from modules import utils

# Satellite
NORAD = pick("NORAD")
TLE_data = utils.get_tle_data(NORAD, "TLE_GONETSM24.txt")
if TLE_data is None or len(TLE_data) < 3:
    raise RuntimeError("TLE data not found or incomplete")

from modules import integrator, plots, errors, frame_transformation, getsource, sgp4_propagator, \
    disturbances, ODE

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

Answers['ls_MDT_TLE_GEN_0'] = TLE_data

#------------------------------------------- Task 0: Plotting capabilities --------------------------------------------#
Answers['ls_MDT_OrbKepRes_SRC_0'], kep_el_res_code = getsource.getsourcefunc(errors.kep_el_and_residuals)
_, kep_el_res_plot = getsource.getsourcefunc(plots.plot_comparison_residual)
Answers['ss_REG_OrbKepRes_CODE_2'] = str(kep_el_res_code + "\n\n" + kep_el_res_plot)

Answers['ls_MDT_MagPlot_SRC_0'], mag_comp_code = getsource.getsourcefunc(errors.mag_dist)
mag_plt_src, mag_plt_code = getsource.getsourcefunc(plots.plot_mag_acc)
Answers['ss_REG_MagPlot_CODE_1'] = str(mag_comp_code + "\n\n" + mag_plt_code)

#--------------------------------------------- Task 1: Update the orbit -----------------------------------------------#
sat = utils.create_satrec(TLE_data)
# USED :
# GONETS-M 24
# 1 54151U 22139B   25352.27746198  .00000012  00000+0  15277-4 0  9992
# 2 54151  82.4976  99.9088 0015715 264.2059  95.7229 12.42860404143175

mean_motion = sat.no_kozai / 60  # [rad/s]
T = 2 * np.pi / mean_motion  # Orbital period
rev = pick("rev")  # Number of propagated orbital periods
ti = pick("ti")  # Arbitrary
tf = rev * T  # Final time
N = pick("N")  # Number of time steps
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

#------------------------------------------- Task 1.2: Integrate the orbit --------------------------------------------#
Answers['ss_MDT_Integrator_GEN_0'] = "Runge-Kutta 4"
y_i_RK4 = np.concatenate((ri, vi))
N_RK4 = pick("N")
rev_RK4 = pick("rev")
tf_RK4 = rev_RK4 * T
dt_RK4 = (tf_RK4 - ti) / N_RK4
tf_RK4 = tf_RK4
time_RK4 = np.arange(ti, tf_RK4 + dt_RK4, dt_RK4)

RHS_RK4 = ODE.make_rhs(include_point_mass=True,
                       include_j2=False,
                       include_srp=False,
                       include_drag=False,
                       include_tbp=False,
                       params=None
                       )
y_RK4_u = integrator.RK4(y_i_RK4, dt_RK4, tf_RK4, RHS_RK4)

Answers['lf_REG_IntPosIni_NUM_0'] = (y_RK4_u[0, 0:3] * 1e3).tolist()  # Initial 3D position
Answers['lf_REG_IntVelIni_NUM_0'] = (y_RK4_u[0, 3:6] * 1e3).tolist()  # Initial 3D velocity
Answers['lf_REG_IntPosEnd_NUM_0.85'] = (y_RK4_u[-1, 0:3] * 1e3).tolist()  # Final 3D position
Answers['lf_REG_IntVelEnd_NUM_0.85'] = (y_RK4_u[-1, 3:6] * 1e3).tolist()  # Final 3D velocity

#-------------------------------------------- Task 1.3: Plot SGP4 Kep Els ---------------------------------------------#
# SGP4 propagation and transformation to keplerian elements
r_SGP4, v_SGP4 = sgp4_propagator.propagated_position_velocity(sat, time_SGP4)
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
        time=time_SGP4.tolist(),
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
        time=time_SGP4.tolist(),
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

(a_RK4, e_RK4, i_rad_RK4, i_deg_RK4,
 Om_rad_RK4, Om_deg_RK4, om_rad_RK4, om_deg_RK4,
 th_rad_RK4, th_deg_RK4, u_deg_RK4) = frame_transformation.cartesian_to_keplerian(y_RK4_u)

# Define the path and name for the file
plot_RK4_filenames = []
plot_aei_RK4_filename = out_dir / "a_e_i_RK4.png"
plot_RK4_filenames.append(str(plot_aei_RK4_filename))
plot_uth_RK4_filename = out_dir / "u_RAAN_RK4.png"
plot_RK4_filenames.append(str(plot_uth_RK4_filename))

# Call the function
if not pick("skip_plots"):
    plots.plot_three_scales(
        time=time_RK4.tolist(),
        # Data 1: Semi-major Axis
        data1=a_RK4,
        label1="Semi-major Axis ($a$)",
        ylabel1="a [km]",
        # Data 2: Eccentricity
        data2=e_RK4,
        label2="Eccentricity ($e$)",
        ylabel2="e [-]",
        # Data 3: Inclination
        data3=i_deg_RK4,
        label3="Inclination ($i$)",
        ylabel3="i [deg]",
        # General Settings
        title=f"Semi-major axis, eccentricity and inclination evolution of RK4 integrated orbit, dt = {dt_RK4:.3f} "
              f"s",
        filename=str(plot_aei_RK4_filename)
    )

    plots.plot_two_scales(
        time=time_RK4.tolist(),
        data1=Om_deg_RK4,
        label1=r"RAAN ($\Omega$)",
        ylabel1=r"$\Omega$ [deg]",
        data2=u_deg_RK4,
        label2=r"Argument of latitude ($u$ = $\omega$ + $\theta$)",
        ylabel2=r"u [deg]",

        title="Argument of latitude and RAAN of RK4 integrated orbit",
        filename=str(plot_uth_RK4_filename)

    )
else:
    print("[INFO] Plots skipped (--skip_plots active)")

Answers['ls_REG_EulerKepPlotFile_PLOT_1'] = plot_RK4_filenames

# Data save:
np.savetxt("data/SGP4_state.txt", y_SGP4, fmt="%.10e")
np.savetxt("data/SGP4_time.txt", time_SGP4, fmt="%.10e")

np.savetxt("data/RK4_state.txt", y_RK4_u, fmt="%.10e")
np.savetxt("data/RK4_time.txt", time_RK4, fmt="%.10e")

#--------------------------------------------- Task 2: Earth Oblateness -----------------------------------------------#
Answers['ls_MDT_J2_SRC_0'], Answers['ss_REG_J2_CODE_2.5'] = getsource.getsourcefunc(disturbances.j2_acc)

#-------------------------------------- Task 2.1: Earth Oblateness - integration --------------------------------------#
RHS_J2 = ODE.make_rhs(include_point_mass=True,
                      include_j2=True,
                      include_srp=False,
                      include_drag=False,
                      include_tbp=False,
                      params=None
                      )
y_J2 = integrator.RK4(y_i_RK4, dt_RK4, tf_RK4, RHS_J2)

np.savetxt("data/y_J2.txt", y_J2, fmt="%.10e")

Answers['lf_REG_J2PosIni_NUM_0'] = (y_J2[0, 0:3] * 1e3).tolist()  # Initial 3D position
Answers['lf_REG_J2VelIni_NUM_0'] = (y_J2[0, 3:6] * 1e3).tolist()  # Initial 3D velocity
Answers['lf_REG_J2PosEnd_NUM_0.85'] = (y_J2[-1, 0:3] * 1e3).tolist()  # Final 3D position
Answers['lf_REG_J2VelEnd_NUM_0.85'] = (y_J2[-1, 3:6] * 1e3).tolist()  # Final 3D velocity

#----------------------------------------- Task 2.2: Earth Oblateness - plot ------------------------------------------#
(a_J2, e_J2, i_rad_J2, i_deg_J2,
 Om_rad_J2, Om_deg_J2, om_rad_J2, om_deg_J2,
 th_rad_J2, th_deg_J2, u_deg_J2) = frame_transformation.cartesian_to_keplerian(y_J2)

plot_J2_filenames = []
plot_a_J2_filename = out_dir / "a_J2.png"
plot_J2_filenames.append(str(plot_a_J2_filename))
plot_e_J2_filename = out_dir / "e_J2.png"
plot_J2_filenames.append(str(plot_e_J2_filename))
plot_i_J2_filename = out_dir / "i_J2.png"
plot_J2_filenames.append(str(plot_i_J2_filename))
plot_RAAN_J2_filename = out_dir / "RAAN_J2.png"
plot_J2_filenames.append(str(plot_RAAN_J2_filename))
plot_u_J2_filename = out_dir / "u_J2.png"
plot_J2_filenames.append(str(plot_u_J2_filename))

# Undisturbed orbit
label_undisturbed = [r"$a_u$",
                     r"$e_u$",
                     r"$i_u$",
                     r"$RAAN_u$",
                     r"$u_u$"]
ylabel_undisturbed = [r"$a [km]$",
                      r"$e [-]$",
                      r"$i [deg]$",
                      r"$RAAN [deg]$",
                      r"$u [deg]$"]

# Orbit with J2 perturbation
label_J2 = [r"$a_d$",
            r"$e_d$",
            r"$i_d$",
            r"$RAAN_d$",
            r"$u_d$"]
ylabel_J2 = [r"$a_d [km]$",
             r"$e_d [-]$",
             r"$i_d [deg]$",
             r"$RAAN_d [deg]$",
             r"$u_d [deg]$"]
label_J2_diff = [r"$\Delta a_{d-u}$",
                 r"$\Delta e_{d-u}$",
                 r"$\Delta i_{d-u}$",
                 r"$\Delta RAAN_{d-u}$",
                 r"$\Delta u_{d-u}$"]
ylabel_J2_diff = [r"$\Delta a_{d-u} [km]$",
                  r"$\Delta e_{d-u} [-]$",
                  r"$i_{d-u} [deg]$",
                  r"$\Delta RAAN_{d-u} [deg]$",
                  r"$\Delta u_{d-u} [deg]$"]

# Figure title and name
fig_title_J2 = [r"Semi-major axis of unperturbed and J2 perturbed orbit",
                r"Eccentricity of unperturbed and J2 perturbed orbit",
                r"Inclination of unperturbed and J2 perturbed orbit",
                r"RAAN of unperturbed and J2 perturbed orbit",
                r"Argument of Latitude of unperturbed and J2 perturbed orbit"]
fig_name_J2 = plot_J2_filenames

# Plot keplerian elements of undisturbed and disturbed orbits
kep_el_u, kep_el_J2, kep_el_J2_diff = errors.kep_el_and_residuals(y_RK4_u, time_RK4, y_J2, time_RK4,
                                                                  label_undisturbed, ylabel_undisturbed,
                                                                  label_J2, ylabel_J2,
                                                                  label_J2_diff, ylabel_J2_diff,
                                                                  fig_title_J2, fig_name_J2)

# Plot magnitude of J2 acceleration
plot_J2_mag_filename = out_dir / "a_mag_J2.png"
J2_acc = disturbances.j2_acc_vec(y_RK4_u[:, :3])
J2_acc_mag = plots.plot_mag_acc(time_RK4, J2_acc * 1e6,
                                "Time [s]", r"$a_{J2}$ [$mm/s^2$]",
                                "Acceleration due to J2", "Acceleration magnitude due to J2 perturbation",
                                str(plot_J2_mag_filename))
Answers['ls_REG_J2Mag_PLOT_2'] = [str(plot_J2_mag_filename)]

#---------------------------------- Task 2.2b: Earth Oblateness - residuals wrt SGP4 ----------------------------------#
res_J2_SGP4 = (np.linalg.norm(y_J2[:, :3], axis=1) - np.linalg.norm(y_SGP4[:, :3], axis=1))
gpd_J2_SGP4 = errors.compute_gpd(y_J2[:, :3], y_SGP4[:, :3])
plot_gpd_J2_filename = out_dir / "GPD_J2.png"

plots.plot_three_scales(time_RK4,
                        np.linalg.norm(y_SGP4[:, :3], axis=1), "SGP4 position magnitude", "SGP4-orbit [km]",
                        np.linalg.norm(y_J2[:, :3], axis=1), "Position magnitude with J2 perturbation", "J2-orbit [km]",
                        gpd_J2_SGP4, "GPD",
                        "GPD [km]", "Global Position difference between SGP4 orbit and J2-perturbed orbit",
                        str(plot_gpd_J2_filename), ax1_big=True)

#-------------------------- Task 2.2c: Earth Oblateness - Keplerian residuals wrt SGP4 --------------------------#

# Unpack residuals (disturbed - undisturbed)
_, _, kep_el_J2_SGP4_diff = errors.kep_el_and_residuals(y_SGP4, time_RK4, y_J2, time_RK4,
                                                        label_undisturbed, ylabel_undisturbed,
                                                        label_J2, ylabel_J2,
                                                        label_J2_diff, ylabel_J2_diff,
                                                        fig_title_J2, fig_name_J2, plot=False)

Delta_a_J2_SGP4 = kep_el_J2_SGP4_diff[:, 0]
Delta_e_J2_SGP4 = kep_el_J2_SGP4_diff[:, 1]
Delta_i_J2_SGP4 = kep_el_J2_SGP4_diff[:, 2]
Delta_RAAN_J2_SGP4 = kep_el_J2_SGP4_diff[:, 3]
Delta_u_J2_SGP4 = kep_el_J2_SGP4_diff[:, 4]

# ---------- Plot Δa, Δe, Δi ----------
plot_kep_res_aei_J2_filename = out_dir / "a_e_i_res_J2_SGP4.png"
plot_J2_filenames.append(str(plot_kep_res_aei_J2_filename))

plot_kep_res_RAAN_u_J2_filename = out_dir / "RAAN_u_res_J2_SGP4.png"
plot_J2_filenames.append(str(plot_kep_res_RAAN_u_J2_filename))

if not pick("skip_plots"):
    plots.plot_three_scales(
        time=time_RK4.tolist(),
        data1=Delta_a_J2_SGP4,
        label1=r"$\Delta a_{J2-SGP4}$",
        ylabel1=r"$\Delta a$ [km]",
        data2=Delta_e_J2_SGP4,
        label2=r"$\Delta e_{J2-SGP4}$",
        ylabel2=r"$\Delta e$ [-]",
        data3=Delta_i_J2_SGP4,
        label3=r"$\Delta i_{J2-SGP4}$",
        ylabel3=r"$\Delta i$ [deg]",
        title="Residuals of semi-major axis, eccentricity and inclination (J2 vs SGP4)",
        filename=str(plot_kep_res_aei_J2_filename),
        ax1_big=True
    )

    # ---------- Plot ΔRAAN, Δu ----------
    plots.plot_two_scales(
        time=time_RK4.tolist(),
        data1=Delta_RAAN_J2_SGP4,
        label1=r"$\Delta \Omega_{J2-SGP4}$",
        ylabel1=r"$\Delta \Omega$ [deg]",
        data2=Delta_u_J2_SGP4,
        label2=r"$\Delta u_{J2-SGP4}$",
        ylabel2=r"$\Delta u$ [deg]",
        title="Residuals of RAAN and argument of latitude (J2 vs SGP4)",
        filename=str(plot_kep_res_RAAN_u_J2_filename)
    )

Answers['ls_REG_J2Orb_PLOT_5'] = plot_J2_filenames

#--------------------------------------- Task 2.3: Earth Oblateness - Analysis ----------------------------------------#
# Observations :
Answers['ss_REG_J2Obs_OIC_6'] = """ (I will refer to the orbit propagated with the SGP4 model as 'SGP4 orbit' and to the orbit integrated including the J2 perturbation as 'J2-perturbed orbit')
| 
- O1: Plot "a_e_i_res_J2_SGP4.png" shows the residuals in semi-major axis, eccentricity and inclination between the SGP4 
and the J2-perturbed orbits. Δa oscillates and reaches a maximum amplitude of about 0.056 km, while Δe and Δi remain of 
the order of 1e-5, although Δi shows an increasing oscillation amplitude. In the plot "a_e_i_RK4.png", the mean values 
of these Keplerian elements (respectively a, e, and i) are approximately 7878 km, 6.56e-4, and 82.5 deg.
- O2: Plot "a_J2.png" shows that the semi-major axis of the J2-perturbed orbit oscillates between approximately
7862.5 and 7877.5 km, with a maximum separation of approximately 16 km with respect to the unperturbed orbit, 
reached twice every orbital period.
- O3: Plot "a_mag_J2.png" shows that the magnitude of the acceleration due to the J2 perturbation follows a periodic 
pattern with values bounded between approximately 6 and 13 mm/s^2; two local maxima around 13 mm/s^2 occur per orbital period.
"""

Answers['ss_REG_J2Int_OIC_8'] = """|
- I1: From O1, the residuals between the SGP4 and the J2-perturbed orbits are small compared to the absolute values 
of the corresponding orbital elements because, over the considered time span, for the Keplerian elements analysed, and 
for this orbit, the perturbation due to J2 represents a dominant contribution included in the SGP4 propagation model.
- I2: From O2 and O3, the semi-major axis of the orbit perturbed by J2 oscillates periodically because the flattening 
of the Earth introduces a latitude-dependent gravitational potential. Specifically, this causes alternating increases and 
decreases in the force (and hence velocity, orbital energy) experienced by the satellite. The semi-major axis follows 
the resulting variation in orbital energy, leading to two extrema per orbital period.
"""

# Conclusion :
Answers['ss_REG_J2Con_OIC_10'] = """|
For short integration periods and first-order analyses, adding the J2 perturbation to the gravitational point-mass 
model yields values for the semi-major axis, eccentricity, and inclination that are comparable to those obtained 
through SGP4 propagation. Additionally, the J2 perturbation causes short-period variations in orbital energy, 
but not secular variations. 
"""

#---------------------------------------------- Task 3: Solar Position  -----------------------------------------------#
TLE_initial_date = utils.tle_days_to_date(sat, 0.0)
np.savetxt("data/TLE_initial_date.txt", TLE_initial_date, fmt="%.2f")

TLE_final_date = utils.tle_days_to_date(sat, (tf_RK4 + dt_RK4) / 86400)
np.savetxt("data/TLE_final_date.txt", TLE_final_date, fmt="%.2f")

Answers['lf_REG_SunPos_NUM_0.9'] = [-1.040487728604979e7 * 1e3, -1.347330452178150e8 * 1e3, -5.840477243536754e7 * 1e3]
sun_pos = np.array([-1.040487728604979e7, -1.347330452178150e8, -5.840477243536754e7])

#----------------------------------------- Task 3.1: Solar Radiation Pressure  ----------------------------------------#
Answers['ls_MDT_SRP_SRC_0'], Answers['ss_REG_SRP_CODE_2.5'] = getsource.getsourcefunc(disturbances.srp_acc)

#--------------------------------- Task 3.2: Solar Radiation Pressure - integration  ----------------------------------#
params_SRP = {
    "r_e_sun": sun_pos
}
RHS_SRP = ODE.make_rhs(include_point_mass=True,
                       include_j2=False,
                       include_srp=True,
                       include_drag=False,
                       include_tbp=False,
                       params=params_SRP
                       )
y_SRP = integrator.RK4(y_i_RK4, dt_RK4, tf_RK4, RHS_SRP)
np.savetxt("data/y_SRP.txt", y_SRP, fmt="%.10e")

Answers['lf_REG_SRPPosIni_NUM_0'] = (y_SRP[0, 0:3] * 1e3).tolist()  # Initial 3D position
Answers['lf_REG_SRPVelIni_NUM_0'] = (y_SRP[0, 3:6] * 1e3).tolist()  # Initial 3D velocity
Answers['lf_REG_SRPPosEnd_NUM_0.85'] = (y_SRP[-1, 0:3] * 1e3).tolist()  # Final 3D position
Answers['lf_REG_SRPVelEnd_NUM_0.85'] = (y_SRP[-1, 3:6] * 1e3).tolist()  # Final 3D velocity

#------------------------------------- Task 3.3: Solar Radiation Pressure - plot  -------------------------------------#
(a_SRP, e_SRP, i_rad_SRP, i_deg_SRP,
 Om_rad_SRP, Om_deg_SRP, om_rad_SRP, om_deg_SRP,
 th_rad_SRP, th_deg_SRP, u_deg_SRP) = frame_transformation.cartesian_to_keplerian(y_SRP)

plot_SRP_filenames = []
plot_a_SRP_filename = out_dir / "a_SRP.png"; plot_SRP_filenames.append(str(plot_a_SRP_filename))
plot_e_SRP_filename = out_dir / "e_SRP.png"; plot_SRP_filenames.append(str(plot_e_SRP_filename))
plot_i_SRP_filename = out_dir / "i_SRP.png"; plot_SRP_filenames.append(str(plot_i_SRP_filename))
plot_RAAN_SRP_filename = out_dir / "RAAN_SRP.png"; plot_SRP_filenames.append(str(plot_RAAN_SRP_filename))
plot_u_SRP_filename = out_dir / "u_SRP.png"; plot_SRP_filenames.append(str(plot_u_SRP_filename))

# Orbit with SRP perturbation
label_SRP = [r"$a_d$",
             r"$e_d$",
             r"$i_d$",
             r"$RAAN_d$",
             r"$u_d$"]
ylabel_SRP = [r"$a_d [km]$",
              r"$e_d [-]$",
              r"$i_d [deg]$",
              r"$RAAN_d [deg]$",
              r"$u_d [deg]$"]
label_SRP_diff = [r"$\Delta a_{d-u}$",
                  r"$\Delta e_{d-u}$",
                  r"$\Delta i_{d-u}$",
                  r"$\Delta RAAN_{d-u}$",
                  r"$\Delta u_{d-u}$"]
ylabel_SRP_diff = [r"$\Delta a_{d-u} [km]$",
                   r"$\Delta e_{d-u} [-]$",
                   r"$i_{d-u} [deg]$",
                   r"$\Delta RAAN_{d-u} [deg]$",
                   r"$\Delta u_{d-u} [deg]$"]

# Figure title and name
fig_title_SRP = [r"Semi-major axis of unperturbed and SRP perturbed orbit",
                 r"Eccentricity of unperturbed and SRP perturbed orbit",
                 r"Inclination of unperturbed and SRP perturbed orbit",
                 r"RAAN of unperturbed and SRP perturbed orbit",
                 r"Argument of Latitude of unperturbed and SRP perturbed orbit"]
fig_name_SRP = plot_SRP_filenames

# Plot keplerian elements of undisturbed and disturbed orbits
_, kep_el_SRP, kep_el_SRP_diff = errors.kep_el_and_residuals(y_RK4_u, time_RK4, y_SRP, time_RK4,
                                                             label_undisturbed, ylabel_undisturbed,
                                                             label_SRP, ylabel_SRP,
                                                             label_SRP_diff, ylabel_SRP_diff,
                                                             fig_title_SRP, fig_name_SRP)

# Plot magnitude of SRP acceleration
plot_SRP_mag_filename = out_dir / "a_mag_SRP.png"
SRP_acc = disturbances.srp_acc_vec(y_RK4_u[:, :3], sun_pos)
SRP_acc_mag = plots.plot_mag_acc(time_RK4, SRP_acc * 1e6,
                                 "Time [s]", r"$a_{SRP}$ [$mm/s^2$]",
                                 "Acceleration due to SRP", "Acceleration magnitude due to SRP perturbation",
                                 str(plot_SRP_mag_filename))

Answers['ls_REG_SRPMag_PLOT_2'] = [str(plot_SRP_mag_filename)]

#----------------------------------------- Task 3.3b: SRP - residuals wrt SGP4 ----------------------------------------#
res_SRP_U = (np.linalg.norm(y_SRP[:, :3], axis=1) - np.linalg.norm(y_RK4_u[:, :3], axis=1))
gpd_SRP_U = errors.compute_gpd(y_SRP[:, :3], y_RK4_u[:, :3])
plot_gpd_SRP_U_filename = out_dir / "GPD_SRP_U.png"
plot_SRP_filenames.append(str(plot_gpd_SRP_U_filename))

plots.plot_three_scales(time_RK4,
                        np.linalg.norm(y_RK4_u[:, :3], axis=1), "Unperturbed orbit position magnitude",
                        "Unperturbed orbit position [km]",
                        np.linalg.norm(y_SRP[:, :3], axis=1), "Position magnitude with SRP perturbation",
                        "SRP perturbed orbit position [km]",
                        gpd_SRP_U, "GPD",
                        "GPD [km]", "Global position difference between unperturbed orbit and"
                                    " orbit perturbed by SRP", str(plot_gpd_SRP_U_filename), ax1_big=True)

#----------------------------------- Task 3.3c: SRP - Keplerian residuals wrt U ------------------------------------#

_, _, kep_el_SRP_U_diff = errors.kep_el_and_residuals(y_RK4_u, time_RK4, y_SRP, time_RK4,
                                                      label_undisturbed, ylabel_undisturbed,
                                                      label_SRP, ylabel_SRP,
                                                      label_SRP_diff, ylabel_SRP_diff,
                                                      fig_title_SRP, fig_name_SRP, plot=False)

Delta_a_SRP_U = kep_el_SRP_U_diff[:, 0]
Delta_e_SRP_U = kep_el_SRP_U_diff[:, 1]
Delta_i_SRP_U = kep_el_SRP_U_diff[:, 2]
Delta_RAAN_SRP_U = kep_el_SRP_U_diff[:, 3]
Delta_u_SRP_U = kep_el_SRP_U_diff[:, 4]

# ---------- Plot Δa, Δe, Δi ----------
plot_kep_res_aei_SRP_filename = out_dir / "a_e_i_res_SRP_U.png"
plot_SRP_filenames.append(str(plot_kep_res_aei_SRP_filename))

plots.plot_three_scales(
    time=time_RK4.tolist(),
    data1=Delta_a_SRP_U,
    label1=r"$\Delta a_{SRP-U}$",
    ylabel1=r"$\Delta a$ [km]",
    data2=Delta_e_SRP_U,
    label2=r"$\Delta e_{SRP-U}$",
    ylabel2=r"$\Delta e$ [-]",
    data3=Delta_i_SRP_U,
    label3=r"$\Delta i_{SRP-U}$",
    ylabel3=r"$\Delta i$ [deg]",
    title="Residuals of semi-major axis, eccentricity and inclination (SRP vs Unperturbed)",
    filename=str(plot_kep_res_aei_SRP_filename),
    ax1_big=True
)

## Specific energy
plot_energy_SRP = out_dir / "specific_energy_SRP.png"
plot_SRP_filenames.append(str(plot_energy_SRP))

r_SRP = np.linalg.norm(y_SRP[:, :3], axis=1)
v2_SRP = np.sum(y_SRP[:, 3:6] ** 2, axis=1)
eps = 0.5 * v2_SRP - pick("mu_E") / r_SRP

plots.plot_normal(time_RK4, eps, "time [s]", r"Specific energy [$km^2/s^2$]",
                  ['Specific energy'], "Specific energy of the orbit perturbed by SRP",
                  filename=str(plot_energy_SRP))

Answers['ls_REG_SRPOrb_PLOT_5'] = plot_SRP_filenames

#----------------------------------- Task 3.4: Solar Radiation Pressure - analysis  -----------------------------------#
# Observations :
Answers['ss_REG_SRPObs_OIC_6'] = """ (I will refer to the orbit integrated including the SRP perturbation as 'SRP-perturbed orbit')
|
- O1: Plot "a_mag_SRP.png" shows that the magnitude of the acceleration due to solar radiation pressure forms a rectangular
  wave pattern, with values alternating between a maximum of about 9e-5 mm/s^2 and zero. Over each orbital period,
  the acceleration is non-zero for most of the time interval. 
- O2: Plot "a_e_i_res_SRP_U.png" shows the residuals in semi-major axis, eccentricity and inclination between the
  unperturbed orbit and the SRP-perturbed orbit. Δe shows a negative piecewise drift with values of order 1e-7, while Δi shows
  a positive piecewise drift with values of order 1e-6 deg. Δa exhibits a near-periodic pattern with no visible drift, reaching a
  maximum amplitude of approximately 35 cm every orbital period. In all cases, intervals of nearly constant residuals 
  alternate with intervals of non-constant residuals.
- O3: Plot "specific_energy_SRP.png" shows that the specific energy of the perturbed orbit alternates between intervals where
  it is constant and intervals where it is non-constant. During the non-constant intervals, the specific energy increases to a
  maximum value (less negative), with variations of order 1e-6 km^2/s^2, and returns to the minimum plateau. No secular drift is visible. 
"""

Answers['ss_REG_SRPInt_OIC_8'] = """|
- I1: From O1 and O2, the acceleration trend causes the residuals to exhibit piecewise behaviour because, when the 
  perturbation due to SRP is zero, the orbital dynamics are governed only by the point-mass gravitational model, and 
  the Keplerian elements therefore remain practically constant.
- I2: From O2 and O3, the semi-major axis exhibits bounded, near-periodic variations because the specific orbital 
  energy varies periodically without a secular drift over the considered time interval, and the two quantities are
  directly related through ε = −μ/(2a).
"""

# Conclusion :
Answers['ss_REG_SRPCon_OIC_10'] = """|
- The inclusion of the perturbation due to SRP, although the associated force is
  non-conservative, introduces secular drifts only in eccentricity and inclination (between the analyzed Keplerian
  elements). Instead, for the considered time interval (only a few orbital periods) it produces short-period variations
  in the orbital energy, which remains bounded.
"""

# ==================================================================================================================== #
#                                               Assignment excellence                                                  #
# ==================================================================================================================== #


#-------------------------------------------- Task AE1: Atmospheric Drag  ---------------------------------------------#
max_altitude = np.max(np.linalg.norm(y_RK4_u[:, :3], axis=1)) - pick("R_E")  # maximum altitude
min_altitude = np.min(np.linalg.norm(y_RK4_u[:, :3], axis=1)) - pick("R_E")  # minimum altitude
avg_altitude = np.average(np.linalg.norm(y_RK4_u[:, :3], axis=1)) - pick("R_E")  # Average altitude = 1507.3 km

ref_altitude = pick("ref_altitude")  # m
scale_height = pick("scale_height")  # m
mean_rho = pick("ref_rho")  # kg/m^3
Answers['sf_AEX_RefHeight_GEN_0.1'] = ref_altitude  # Reference height in implementation of drag, in meters
Answers['sf_AEX_ScaleHeight_GEN_0.1'] = scale_height  # Scale height  in implementation of drag, in meters
Answers['sf_AEX_MeanRho_GEN_0.1'] = mean_rho  # mean density in implementation of drag, in kg/m^3

#-------------------------------------- Task AE1.1: Atmospheric Drag - function  --------------------------------------#
Answers['ls_AEX_Drag_SRC_0.1'], Answers['ss_AEX_Drag_CODE_0.8'] = getsource.getsourcefunc(disturbances.drag_acc)

#------------------------------------- Task AE1.2: Atmospheric Drag - integration  ------------------------------------#
RHS_RK4_Drag = ODE.make_rhs(include_point_mass=True,
                            include_j2=False,
                            include_srp=False,
                            include_drag=True,
                            include_tbp=False,
                            params=None
                            )
y_RK4_Drag = integrator.RK4(y_i_RK4, dt_RK4, tf_RK4, RHS_RK4_Drag)

np.savetxt("data/y_Drag.txt", y_RK4_Drag, fmt="%.10e")

Answers['lf_AEX_DragPosIni_NUM_0'] = (y_RK4_Drag[0, 0:3] * 1e3).tolist()  # Initial 3D position
Answers['lf_AEX_DragVelIni_NUM_0'] = (y_RK4_Drag[0, 3:6] * 1e3).tolist()  # Initial 3D velocity
Answers['lf_AEX_DragPosEnd_NUM_0.2'] = (y_RK4_Drag[-1, 0:3] * 1e3).tolist()  # Final 3D position
Answers['lf_AEX_DragVelEnd_NUM_0.2'] = (y_RK4_Drag[-1, 3:6] * 1e3).tolist()  # Final 3D velocity

#------------------------------------- Task AE1.3: Atmospheric Drag - plot  ------------------------------------#
(a_Drag, e_Drag, i_rad_Drag, i_deg_Drag,
 Om_rad_Drag, Om_deg_Drag, om_rad_Drag, om_deg_Drag,
 th_rad_Drag, th_deg_Drag, u_deg_Drag) = frame_transformation.cartesian_to_keplerian(y_RK4_Drag)

plot_Drag_filenames = []
plot_a_Drag_filename = out_dir / "a_Drag.png"; plot_Drag_filenames.append(str(plot_a_Drag_filename))
plot_e_Drag_filename = out_dir / "e_Drag.png"; plot_Drag_filenames.append(str(plot_e_Drag_filename))
plot_i_Drag_filename = out_dir / "i_Drag.png"; plot_Drag_filenames.append(str(plot_i_Drag_filename))
plot_RAAN_Drag_filename = out_dir / "RAAN_Drag.png"; plot_Drag_filenames.append(str(plot_RAAN_Drag_filename))
plot_u_Drag_filename = out_dir / "u_Drag.png"; plot_Drag_filenames.append(str(plot_u_Drag_filename))

# Orbit with Drag perturbation
label_Drag = [r"$a_d$",
              r"$e_d$",
              r"$i_d$",
              r"$RAAN_d$",
              r"$u_d$"]
ylabel_Drag = [r"$a_d [km]$",
               r"$e_d [-]$",
               r"$i_d [deg]$",
               r"$RAAN_d [deg]$",
               r"$u_d [deg]$"]
label_Drag_diff = [r"$\Delta a_{d-u}$",
                   r"$\Delta e_{d-u}$",
                   r"$\Delta i_{d-u}$",
                   r"$\Delta RAAN_{d-u}$",
                   r"$\Delta u_{d-u}$"]
ylabel_Drag_diff = [r"$\Delta a_{d-u} [km]$",
                    r"$\Delta e_{d-u} [-]$",
                    r"$i_{d-u} [deg]$",
                    r"$\Delta RAAN_{d-u} [deg]$",
                    r"$\Delta u_{d-u} [deg]$"]

# Figure title and name
fig_title_Drag = [r"Semi-major axis of unperturbed and Drag perturbed orbit",
                  r"Eccentricity of unperturbed and Drag perturbed orbit",
                  r"Inclination of unperturbed and Drag perturbed orbit",
                  r"RAAN of unperturbed and Drag perturbed orbit",
                  r"Argument of Latitude of unperturbed and Drag perturbed orbit"]
fig_name_Drag = plot_Drag_filenames

# Plot keplerian elements of undisturbed and disturbed orbits
_, kep_el_Drag, kep_el_Drag_diff = errors.kep_el_and_residuals(y_RK4_u, time_RK4, y_RK4_Drag, time_RK4,
                                                               label_undisturbed, ylabel_undisturbed,
                                                               label_Drag, ylabel_Drag,
                                                               label_Drag_diff, ylabel_Drag_diff,
                                                               fig_title_Drag, fig_name_Drag)
Answers['ls_AEX_DragOrb_PLOT_1'] = plot_Drag_filenames

# Plot magnitude of Drag acceleration
plot_Drag_mag_filename = out_dir / "a_mag_Drag.png"
Drag_acc = disturbances.drag_acc_vec(y_RK4_u[:, :3], y_RK4_u[:, 3:6])
Drag_acc_mag = plots.plot_mag_acc(time_RK4, Drag_acc * 1e6,
                                  "Time [s]", r"$a_{Drag}$ [$mm/s^2$]",
                                  "Acceleration due to Drag", "Acceleration magnitude due to Drag perturbation",
                                  str(plot_Drag_mag_filename))

Answers['ls_AEX_DragMag_PLOT_0.5'] = [str(plot_Drag_mag_filename)]

#-------------------------------------- Task AE1.4: Atmospheric Drag - analysis  --------------------------------------#
# Observations :
Answers['ss_AEX_DragObs_OIC_1'] = """|
- O1: Plot "a_mag_Drag.png" shows that the magnitude of drag acceleration oscillates periodically between ~6.90×10^-8 and
 ~6.96×10^-8 mm/s^2, repeating approximately the same pattern over multiple orbits.
- O2: Plots "a_Drag.png" and "u_Drag.png" show that the residual Δa_d-u decreases over time and reaches ~(-5)×10^-9 km 
  at the end of integration, while Δu_d-u increases over time and reaches ~9×10^-10 deg.
- O3: Plots "i_Drag.png" and "RAAN_Drag.png" show that i_u and i_d are practically coincident and Δi_d-u drops to 
  ~(-8)×10^-13 deg; also RAAN_u and RAAN_d are visibly coincident and ΔRAAN_d-u remains confined about within 
  ±6×10^-14 deg with no clear trend.
"""

# Interpretations :
Answers['ss_AEX_DragInt_OIC_2'] = """|
- I1: From O1, the periodic oscillation of |a_drag| is consistent with the dependence of the drag on ρ(h) and |v_rel|^2; along
  the orbit these terms oscillate and thus the magnitude of the perturbation also varies with approximately orbital periodicity.
- I2: From O2 and O3, the fact that drift is observed on a and u, but not on i and RAAN, is consistent with a perturbation
  mainly in-plane (or along-track): it reduces the orbital energy (decrease in a, E becomes more negative) and cumulatively
  changes the orbital phase (growth of Δu), while producing a negligible contribution on the orbital plane.
"""

# Conclusion :
Answers['ss_AEX_DragCon_OIC_4'] = """|
- For relatively short integration periods (between two and seven orbital periods), atmospheric-drag perturbations tend to be most
  apparent in in-plane quantities, while out-of-plane quantities remain negligible unless the spacecraft
  aerodynamics introduce a sustained out-of-plane component (e.g., non-spherical shape or attitude-dependent lift). For a spherical satellite
  with purely drag-like acceleration, orbital plane variations are therefore expected to stay near to machine precision 
  values.
"""

#------------------------------------- Task AE2: Third body perturbations (TBP)  --------------------------------------#
moon_pos = np.array([-1.820028042697523e5, -3.176222003235988e5, -1.758111396194424e5])
Answers['lf_AEX_MoonPos_NUM_0.1'] = (moon_pos * 1e3).tolist()  # Moon position at start epoch in m
# Sun position is the same = sun_pos


#------------------------------------------- Task AE2.1:  TBP - function   --------------------------------------------#
Answers['ls_AEX_TBP_SRC_0.1'], Answers['ss_AEX_TBP_CODE_0.8'] = getsource.getsourcefunc(disturbances.tbp_acc)

#------------------------------------------ Task AE2.2:  TBP - integration   ------------------------------------------#
params_SRP = {
    "r_e_sun": sun_pos,
    "r_e_moon": moon_pos
}
RHS_RK4_TBP = ODE.make_rhs(include_point_mass=True,
                           include_j2=False,
                           include_srp=False,
                           include_drag=False,
                           include_tbp=True,
                           params=params_SRP
                           )

y_RK4_TBP = integrator.RK4(y_i_RK4, dt_RK4, tf_RK4, RHS_RK4_TBP)
np.savetxt("data/y_TBP.txt", y_RK4_TBP, fmt="%.10e")

Answers['lf_AEX_TBPPosIni_NUM_0'] = (y_RK4_TBP[0, 0:3] * 1e3).tolist()  # Initial 3D position
Answers['lf_AEX_TBPVelIni_NUM_0'] = (y_RK4_TBP[0, 3:6] * 1e3).tolist()  # Initial 3D velocity
Answers['lf_AEX_TBPPosEnd_NUM_0.2'] = (y_RK4_TBP[-1, 0:3] * 1e3).tolist()  # Final 3D position
Answers['lf_AEX_TBPVelEnd_NUM_0.2'] = (y_RK4_TBP[-1, 3:6] * 1e3).tolist()  # Final 3D velocity

#---------------------------------------------- Task AE1.3:  TBP - plot  ----------------------------------------------#
(a_TBP, e_TBP, i_rad_TBP, i_deg_TBP,
 Om_rad_TBP, Om_deg_TBP, om_rad_TBP, om_deg_TBP,
 th_rad_TBP, th_deg_TBP, u_deg_TBP) = frame_transformation.cartesian_to_keplerian(y_RK4_TBP)

plot_TBP_filenames = []
plot_a_TBP_filename = out_dir / "a_TBP.png"
plot_TBP_filenames.append(str(plot_a_TBP_filename))
plot_e_TBP_filename = out_dir / "e_TBP.png"
plot_TBP_filenames.append(str(plot_e_TBP_filename))
plot_i_TBP_filename = out_dir / "i_TBP.png"
plot_TBP_filenames.append(str(plot_i_TBP_filename))
plot_RAAN_TBP_filename = out_dir / "RAAN_TBP.png"
plot_TBP_filenames.append(str(plot_RAAN_TBP_filename))
plot_u_TBP_filename = out_dir / "u_TBP.png"
plot_TBP_filenames.append(str(plot_u_TBP_filename))

# Orbit with TBP perturbation
label_TBP = [r"$a_d$",
             r"$e_d$",
             r"$i_d$",
             r"$RAAN_d$",
             r"$u_d$"]
ylabel_TBP = [r"$a_d [km]$",
              r"$e_d [-]$",
              r"$i_d [deg]$",
              r"$RAAN_d [deg]$",
              r"$u_d [deg]$"]
label_TBP_diff = [r"$\Delta a_{d-u}$",
                  r"$\Delta e_{d-u}$",
                  r"$\Delta i_{d-u}$",
                  r"$\Delta RAAN_{d-u}$",
                  r"$\Delta u_{d-u}$"]
ylabel_TBP_diff = [r"$\Delta a_{d-u} [km]$",
                   r"$\Delta e_{d-u} [-]$",
                   r"$i_{d-u} [deg]$",
                   r"$\Delta RAAN_{d-u} [deg]$",
                   r"$\Delta u_{d-u} [deg]$"]

# Figure title and name
fig_title_TBP = [r"Semi-major axis of unperturbed and TBP perturbed orbit",
                 r"Eccentricity of unperturbed and TBP perturbed orbit",
                 r"Inclination of unperturbed and TBP perturbed orbit",
                 r"RAAN of unperturbed and TBP perturbed orbit",
                 r"Argument of Latitude of unperturbed and TBP perturbed orbit"]
fig_name_TBP = plot_TBP_filenames

# Plot keplerian elements of undisturbed and disturbed orbits
_, kep_el_TBP, kep_el_TBP_diff = errors.kep_el_and_residuals(y_RK4_u, time_RK4, y_RK4_TBP, time_RK4,
                                                             label_undisturbed, ylabel_undisturbed,
                                                             label_TBP, ylabel_TBP,
                                                             label_TBP_diff, ylabel_TBP_diff,
                                                             fig_title_TBP, fig_name_TBP)

# Plot magnitude of TBP acceleration
plot_TBP_mag_filename = out_dir / "a_mag_TBP.png"
TBP_acc = disturbances.tbp_acc_vec(y_RK4_u[:, :3], sun_pos, moon_pos)
TBP_acc_mag = plots.plot_mag_acc(time_RK4, TBP_acc * 1e6,
                                 "Time [s]", r"$a_{TBP}$ [$mm/s^2$]",
                                 "Acceleration due to TBP", "Acceleration magnitude due to TBP perturbation",
                                 str(plot_TBP_mag_filename))

Answers['ls_AEX_TBPMag_PLOT_0.5'] = [str(plot_TBP_mag_filename)]

# GPD between TBP-perturbed and undisturbed orbits (km)
GPD_TBP_km = errors.compute_gpd(y_RK4_TBP[:, :3], y_RK4_u[:, :3])

# Distances from satellite to Moon and Sun (km), using Earth-centered vectors
r_sat_TBP = y_RK4_TBP[:, :3]
d_moon_sat_km = np.linalg.norm(r_sat_TBP - moon_pos.reshape(1, 3), axis=1)
d_sun_sat_km = np.linalg.norm(r_sat_TBP - sun_pos.reshape(1, 3), axis=1)

# One plot with three scales: GPD, Moon distance, Sun distance
plot_TBP_gpd_dist_filename = out_dir / "GPD_TBP_SunMoonDist.png"
plot_TBP_filenames.append(str(plot_TBP_gpd_dist_filename))

plots.plot_three_scales(
    time_RK4,
    GPD_TBP_km, r"$GPD_{d-u}$", r"$[km]$",
    d_moon_sat_km, r"$|r_{m-sat}|$", r"$[km]$",
    d_sun_sat_km, r"$|r_{s-sat}|$", r"$[km]$",
    title="TBP: GPD and distances to Moon/Sun",
    filename=str(plot_TBP_gpd_dist_filename),
    ax1_big=True
)

# TBP acceleration components in ECI (km/s^2 -> mm/s^2)
TBP_acc_eci_km_s2 = disturbances.tbp_acc_vec(y_RK4_TBP[:, :3], sun_pos, moon_pos)
TBP_acc_eci_mm_s2 = TBP_acc_eci_km_s2 * 1e6

# plot_TBP_acc_eci_filename = out_dir / "a_TBP_ECI.png"
t_mat = np.tile(time_RK4, (3, 1))  # (3, N) to keep plot_normal happy

r_TBP = y_RK4_TBP[:, :3]
v_TBP = y_RK4_TBP[:, 3:6]

r_norm = np.linalg.norm(r_TBP, axis=1)
R_hat = r_TBP / r_norm[:, None]

h = np.linalg.cross(r_TBP, v_TBP)
h_norm = np.linalg.norm(h, axis=1)
C_hat = h / h_norm[:, None]

v_norm = np.linalg.norm(v_TBP, axis=1)
A_hat = v_TBP / v_norm[:, None]

# Project acceleration onto {R,T,L}
a_R = np.sum(TBP_acc_eci_km_s2 * R_hat, axis=1)
a_C = np.sum(TBP_acc_eci_km_s2 * C_hat, axis=1)
a_A = np.sum(TBP_acc_eci_km_s2 * A_hat, axis=1)

TBP_acc_rsw_mm_s2 = np.vstack((a_R, a_A, a_C)).T * 1e6

plot_TBP_acc_rsw_filename = out_dir / "a_TBP_RSW.png"
plot_TBP_filenames.append(str(plot_TBP_acc_rsw_filename))

plots.plot_normal(
    t_mat, TBP_acc_rsw_mm_s2.T,
    "Time [s]", r"$a_{TBP,RSW}$ [$mm/s^2$]",
    [r"$a_R$", r"$a_S$", r"$a_W$"],
    title="TBP acceleration components in satellite radial RF (Radial, Along-track, Cross-track) - RSW",
    filename=str(plot_TBP_acc_rsw_filename)
)

Answers['ls_AEX_TBPOrb_PLOT_1'] = plot_TBP_filenames

#-------------------------------------------- Task AE1.3:  TBP - analysis  --------------------------------------------#
# Observations :
Answers['ss_AEX_TBPObs_OIC_1'] = """|
- O1: Plot "a_mag_TBP.png" shows that |a_TBP| varies periodically between about 9e-4 and ~1.5e-3 mm/s^2; during each 
  orbital period, two local maxima and minima are present.
- O2: Plot "a_TBP_RSW.png" shows that the Radial, Along-track and Cross-track components have different dominant periods:
  a_W completes about one cycle per orbit, while a_R and a_S complete about two cycles per orbit.
- O3: Plot "GPD_TBP_SunMoonDist.png" shows that GPD_{d-u} increases to about 35 m after five orbital periods,
  with oscillations superimposed; in the same plot, |r_{m-sat}| and |r_{s-sat}| reach their local maxima/minima approximately at the
  same times (their oscillations are nearly in phase).
"""

# Interpretations :
Answers['ss_AEX_TBPInt_OIC_2'] = """|
- I1: From O3, the non-zero final GPD indicates that the third-body perturbation does not result in a purely periodic
  displacement that cancels after each revolution. Even though gravity is conservative, the third-body term modifies the
  satellite’s trajectory relative to the two-body (Earth-only) reference, so small orbit-to-orbit mismatches accumulate
  over time, producing a growing separation with superimposed short-period oscillations.
- I2: From O1 and O2, the ~2-per-orbit pattern in |a_TBP| and in the in-plane components (a_R and a_S) is caused by
  the tidal nature of third-body gravity: the perturbation depends on the satellite’s alignment with the Earth–third-body
  direction and is strongest when the satellite is approximately collinear with that direction (both on the near side and
  on the far side), and weakest near quadrature. This geometry repeats every half orbit, which yields two extrema per
  orbital period; in addition, the radial projection can keep the same sign in both collinear configurations because both
  the tidal acceleration and the radial unit vector reverse together on opposite sides of the orbit.
"""

# Conclusion :
Answers['ss_AEX_TBPCon_OIC_4'] = """|
- For short periods of integration, and assuming a constant position of the third body, the disturbance is approximately 
  a static tidal field; in that regime, the dominant short-period signature in the orbital-plane components typically
  repeats every half orbit (i.e., has a twice-per-orbit harmonic), while the out-of-plane component is typically 
  dominated by a 1-per-orbit term.
"""

# ==================================================================================================================== #
#                                               Code excellence and comments                                           #
# ==================================================================================================================== #


# Final answers:
Answers['ss_CEX_Explain_GEN_10'] = """|
- Developed a robust TLE caching and retrieval system with fallback logic, local storage, NORAD/name matching, 
malformed-response handling, and incremental caching.
- The code is structured in files with functions for specific or general tasks.
- Vectorised the CartoKep and KeptoCar transformation, eliminating loops.
- Built reusable plotting functions including matrix-based plotting and generic plotting interfaces.
- Implemented a layered configuration system (DEFAULTS -> CONFIG.yaml -> CLI arguments) with the `pick()` 
selector.
"""

Answers['ss_MDT_AI_GEN_0'] = """|
I used AI to get help with the syntax, the libraries and the yaml file generation. Also, 
plotting functions and secondary functions have been refined by AI for better clarity.
"""

Answers['sf_REG_WORKLOAD_GEN_0'] = 25
Answers['ss_REG_FEEDBACK_GEN_0'] = """|
Assignment useful to explore the different perturbations and their effects, although the analysis is limited to a certain range of orbits. 
"""

########################################################################################################################
# YAML file writing #
# with open("answer-sheet.yaml", "w") as f:
#     dump(Answers, f)
with open("answer-sheet.yaml", "w") as f:
    dump(Answers, f, sort_keys=False, default_flow_style=False)

print("\n--- Running Sanity Check ---")

sanity_script = Path(__file__).parent / "answer-sheet-sanity.py"
yaml_file = Path(__file__).parent / "answer-sheet.yaml"

subprocess.run([sys.executable, str(sanity_script), str(yaml_file)])
