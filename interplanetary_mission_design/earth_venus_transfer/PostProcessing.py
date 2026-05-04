import subprocess
import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.getcwd())

from Configuration.debug import pick
from Modules.utility import prepare_directories
from Modules.plotting import plot_3d_orbits, plot_normal, plot_three_vertical_subplots, plot_three_scales, plot_two_vertical_subplots, LABEL_SIZE, TITLE_SIZE, LEGEND_SIZE

from tudatpy import constants
from tudatpy.interface import spice


current_task = pick('task')
only_plotting = pick('only_plotting')

# Map short task identifiers to the actual script filenames
all_scripts = {
    "q1": "interplanetary_transfer_Q1.py",
    "q2": "interplanetary_transfer_Q2.py",
    "q3": "interplanetary_transfer_Q3.py",
    "q4": "interplanetary_transfer_Q4.py"
}

# Determine which tasks to process
if current_task == "all":
    task_keys = list(all_scripts.keys())
    print("Processing all tasks...")
elif current_task.lower() in all_scripts:
    task_keys = [current_task.lower()]
    print(f"Processing task: {current_task}")
else:
    # Try to match full filename if the key doesn't work
    matched_key = next((k for k, v in all_scripts.items() if v == current_task), None)
    if matched_key:
        task_keys = [matched_key]
    else:
        task_keys = []
        print(f"No valid task found for: {current_task}")

# 1. RUN SIMULATIONS
if not only_plotting:
    for key in task_keys:
        script = all_scripts[key]
        print(f"\n--- Running {script} (task {key}) ---")
        prepare_directories(key)
        
        # Execute script and pass the task flag so internal picks work correctly
        result = subprocess.run([sys.executable, script, "--task", key])
        
        if result.returncode != 0:
            print(f"WARNING: {script} encountered an error.")
        else:
            print(f"SUCCESS: {script} finished.")
else:
    print("\nOnly plotting mode active. Skipping simulations.")



###########################################################################
# PLOTTING LOGIC ##########################################################
###########################################################################

def plot_q1_results():
    
    print("\n--- Generating Plots for Q1 ---")
    output_dir = Path("SimulationOutput")
    num_file = output_dir / "Q1_numerical_states.dat"
    lambert_file = output_dir / "Q1_lambert_states.dat"

    if not num_file.exists() or not lambert_file.exists():
        print(f"Skipping Q1 plots: Data files not found in {output_dir}")
        return

    # Load data (Time, x, y, z, vx, vy, vz)
    num_data = np.loadtxt(num_file)
    lambert_data = np.loadtxt(lambert_file)

    time = num_data[:, 0]
    time_days = (time - time[0]) / 86400.0
    num_pos = num_data[:, 1:4]
    lam_pos = lambert_data[:, 1:4]

    # Plot 1: 3D Trajectory
    trajectories = {
        'Numerical Propagation': num_pos,
        'Lambert Arc': lam_pos
    }
    plot_3d_orbits(
        trajectories, 
        origin_name="Sun", 
        title="3D Interplanetary Trajectory", 
        filename="Q1_3d_trajectory.pdf", 
        task="q1"
    )

    # Plot 2: Cartesian Position Differences using plot_normal
    diff_pos = num_pos - lam_pos
    # diff_pos[diff_pos < 1e-12] = 'nan' # Use 1e-12 as a floor for better log clarity
    # plot_normal(
    #     time_days, 
    #     diff_pos, 
    #     xlabel="Time [days]", 
    #     ylabel="Position Difference [m]",
    #     title="Cartesian Position Differences: Numerical vs Lambert", 
    #     filename="Q1_pos_differences.pdf", 
    #     leg_labels=[r"$\Delta x$", r"$\Delta y$", r"$\Delta z$"],
    #     task="q1",
    #     logy=True
    # )
    plot_three_scales(
        time_days,
        diff_pos[:, 0], "x", r"$\Delta x$ [m]",
        diff_pos[:, 1], "y", r"$\Delta y$ [m]",
        diff_pos[:, 2], "z", r"$\Delta z$ [m]",
        "Cartesian Position Differences: Numerical vs Lambert", 
        "Q1_pos_differences.pdf", 
        ax1_big=False,
        task="q1"
    )
    
    print("Q1 plots generated in Plots/q1/")




def plot_q2_results():
    print("\n--- Generating Plots for Q2 ---")
    data_dir = Path("Data/q2")
    output_dir = Path("SimulationOutput")
    
    # 1. Plot residuals for the 4 plain cases
    cases = ["case_i", "case_ii", "case_iii", "case_iv"]
    
 
    for case in cases:
        # --- Residuals Plotting ---
        pos_file = data_dir / f"{case}_pos_residual.dat"
        vel_file = data_dir / f"{case}_vel_residual.dat"
        acc_file = data_dir / f"{case}_acc_residual.dat"
        
        if pos_file.exists() and vel_file.exists() and acc_file.exists():
            # Load data (Time, x, y, z)
            pos_data = np.loadtxt(pos_file)
            vel_data = np.loadtxt(vel_file)
            acc_data = np.loadtxt(acc_file)
            
            time = pos_data[:, 0]
            time_days = (time - time[0]) / 86400.0
            
            # Compute norms
            pos_norm = np.linalg.norm(pos_data[:, 1:], axis=1)
            vel_norm = np.linalg.norm(vel_data[:, 1:], axis=1)
            acc_norm = np.linalg.norm(acc_data[:, 1:], axis=1)

            # Avoid exact zeros for log scale
            # pos_norm[pos_norm < 1e-15] = 'nan'
            # vel_norm[vel_norm < 1e-15] = 'nan'
            # acc_norm[acc_norm < 1e-15] = 'nan'
            
            # Determine title
            plot_title = f"Residual Norms: {case}"
            if case == "case_iii":
                t_start = time[0]
                t_end = time[-1]
                dep_epoch_case_i = np.loadtxt(output_dir / "Q2_case_i_numerical_states.dat")[:, 0][0]
                # arr_epoch_case_i = np.loadtxt(output_dir / "Q2_case_i_numerical_states.dat")[:, 0][-1]

                diff_start = (t_start - dep_epoch_case_i) / 3600.0
                # diff_end = (t_end - arr_epoch_case_i) / 3600.0
                plot_title += f"\nStart: {diff_start:+.2f} h (vs Case I)"

            plot_three_vertical_subplots(
                    time_days,
                    pos_norm, r"$||\Delta \mathbf{r}||$ [m]",
                    vel_norm, r"$||\Delta \mathbf{v}||$ [m/s]",
                    acc_norm, r"$||\Delta \mathbf{a}||$ [m/s$^2$]",
                    title=plot_title,
                    filename=f"Q2_{case}_residuals.pdf",
                    task="q2",
                    logy=True
            )
        else:
            print(f"Skipping residuals for {case}: Data files not found in {data_dir}")

        # --- 3D Trajectory Plotting ---
        num_file = output_dir / f"Q2_{case}_numerical_states.dat"
        lam_file = output_dir / f"Q2_{case}_lambert_states.dat"

        if num_file.exists() and lam_file.exists():
            num_data = np.loadtxt(num_file)
            lam_data = np.loadtxt(lam_file)
            
            trajectories = {
                'Numerical': num_data[:, 1:4],
                'Lambert': lam_data[:, 1:4]
            }
            
            plot_3d_orbits(
                trajectories, 
                origin_name="Sun", 
                title=f"3D Trajectory - {case}", 
                filename=f"Q2_{case}_3d_trajectory.pdf", 
                task="q2"
            )
        else:
             print(f"Skipping 3D plot for {case}: State files not found in {output_dir}")

    # 2. Compare mass impact on Case IV
    # We compare case_iv (1000kg) vs case_iv_500 (500kg) and case_iv_250 (250kg)
    base_file = output_dir / "Q2_case_iv_numerical_states.dat"
    file_500 =  output_dir / "Q2_case_iv_500_numerical_states.dat"
    file_250 =  output_dir / "Q2_case_iv_250_numerical_states.dat"

    if base_file.exists() and file_500.exists() and file_250.exists():
        base_data = np.loadtxt(base_file)
        data_500 = np.loadtxt(file_500)
        data_250 = np.loadtxt(file_250)



        # Calculate norm of position difference
        diff_500 = np.linalg.norm(data_500[:, 1:] - base_data[:, 1:], axis=1)
        diff_250 = np.linalg.norm(data_250[:, 1:] - base_data[:, 1:], axis=1)

        print(diff_250[-1]/diff_500[-1])

        plot_two_vertical_subplots(
            time_days,
            diff_500, r"$||\Delta \mathbf{r}_{500-1000}||$ [m]",
            diff_250, r"$||\Delta \mathbf{r}_{250-1000}||$ [m]",
            title="Impact of Spacecraft Mass on Trajectory (Case IV)",
            filename="Q2_mass_impact.pdf",
            task="q2",
            logy=True
        )

        # Figure of Merit computation: Ratio of position error norms
        diff_500[diff_500 == 0] = 'nan'
        diff_250[diff_250 == 0] = 'nan'
        FoM = diff_250/diff_500

        plot_normal(
            time_days, 
            FoM, 'Time [days]', r'FoM = $\frac{||\Delta \mathbf{r}_{250-1000}||}{||\Delta \mathbf{r}_{500-1000}||}$ ',
            title="Figure of Merit: Impact of Spacecraft Mass on Trajectory (Case IV)", 
            filename="Q2_FoM.pdf", 
            task="q2"
        )
        print("Q2 plots generated in Plots/q2/")    
    

def plot_q3_results():


    print("\n--- Generating Plots for Q3 ---")
    output_dir = Path("SimulationOutput")
    
    number_of_arcs = 10
    
    # Cases to plot: (a) uncorrected, (c) single correction, (d) iterative correction
    cases = ["Q3a", "Q3c", "Q3d"]
    case_labels = ["Uncorrected Arcwise Propagation (a)", 
                   "Single-Step Corrected Arcwise Propagation (c)", 
                   "Iteratively Corrected Arcwise Propagation (d)"]
    filenames = ["Q3a_pos_error_norm.pdf", "Q3c_pos_error_norm.pdf", "Q3d_pos_error_norm.pdf"]
    
    # Get the global departure epoch from the first arc's data to normalize time
    first_arc_file = output_dir / "Q3a_arc_0_numerical_states.dat"
    if not first_arc_file.exists():
        print("ERROR: Cannot generate Q3 plots. Data for arc 0 not found.")
        return
        
    departure_epoch = np.loadtxt(first_arc_file)[0, 0]

    for idx, case in enumerate(cases):
        case_time_days = []
        case_error_norm = []
        case_num_pos = []
        case_lam_pos = []
        
        arc_initial_times = []
        arc_initial_errors = []
        arc_final_times = []
        arc_final_errors = []

        # Loop through all arcs to collect and stitch data
        for i in range(number_of_arcs):
            file_num = output_dir / f"{case}_arc_{i}_numerical_states.dat"
            file_lam = output_dir / f"{case}_arc_{i}_lambert_states.dat"
            
            if file_num.exists() and file_lam.exists():
                num_data = np.loadtxt(file_num)
                lam_data = np.loadtxt(file_lam)
                
                # Calculate time in days relative to the start of the entire transfer
                t = (num_data[:, 0] - departure_epoch) / 86400.0
                
                # Calculate the norm of the position error vector
                pos_error = num_data[:, 1:4] - lam_data[:, 1:4]
                error_norm = np.linalg.norm(pos_error, axis=1)
                error_norm[error_norm < 1e-12] = 1e-12 # Use 1e-12 as a floor for better log clarity

                # Track start and end points of each arc for dots
                arc_initial_times.append(t[0])
                arc_initial_errors.append(error_norm[0])
                arc_final_times.append(t[-1])
                arc_final_errors.append(error_norm[-1])

                # Append data, avoiding duplicate junction points
                if i < number_of_arcs - 1:
                    case_time_days.extend(t[:-1])
                    case_error_norm.extend(error_norm[:-1])
                    case_num_pos.append(num_data[:-1, 1:4])
                    case_lam_pos.append(lam_data[:-1, 1:4])
                else:  # For the last arc, include all points
                    case_time_days.extend(t)
                    case_error_norm.extend(error_norm)
                    case_num_pos.append(num_data[:, 1:4])
                    case_lam_pos.append(lam_data[:, 1:4])
            else:
                print(f"WARNING: Data for {case} arc {i} not found.")

        if case_time_days:

            np.savetxt(f'./Data/q3/{case}_residual.txt', np.column_stack((case_time_days, case_error_norm)), delimiter=',')
            
            scatter_list = [
                (arc_initial_times, arc_initial_errors, r"Arc Initial Error ($0\rightarrow 10^{-12}$)", "green", "o"),
                (arc_final_times, arc_final_errors, "Arc Final Error", "red", "s")
            ]
            
            # 1. Position Error Norm Plot
            plot_normal(
                np.array(case_time_days), 
                np.array(case_error_norm), 
                xlabel="Time [days]", 
                ylabel=r"Position Error Norm $||\Delta\mathbf{r}(t)||$ [m]",
                title=f"Position Error Norm: {case_labels[idx]}", 
                filename=filenames[idx], 
                task="q3",
                logy=True,
                scatter_list=scatter_list
            )
            
            # 2. 3D Orbit Plot
            trajectories = {
                'Numerical': np.vstack(case_num_pos),
                'Lambert': np.vstack(case_lam_pos)
            }
            plot_3d_orbits(
                trajectories, 
                origin_name="Sun", 
                title=f"3D Trajectory: {case_labels[idx]}", 
                filename=f"{case}_3d_trajectory.pdf", 
                task="q3"
            )
            print(f"SUCCESS: Generated plots for {case} in Plots/q3/")
        else:
            print(f"ERROR: No data found for {case} to generate plot.")

def plot_q4_results():
    """
    Generates plots for Question 4, showing the residual norm between numerical 
    and Lambert states for single and two-arc propagations, plus Monte Carlo results.
    """
    print("\n--- Generating Plots for Q4 ---")
    output_dir = Path("SimulationOutput")
    
    # --- Plotting 4b (single arc) ---
    num_file = output_dir / "Q4_single_arc_numerical_states.dat"
    lambert_file = output_dir / "Q4_single_arc_lambert_states.dat"

    if num_file.exists() and lambert_file.exists():
        num_data = np.loadtxt(num_file)
        lambert_data = np.loadtxt(lambert_file)
        time = num_data[:, 0]
        time_days = (time - time[0]) / 86400.0
        pos_error = num_data[:, 1:4] - lambert_data[:, 1:4]
        error_norm = np.linalg.norm(pos_error, axis=1)
        
        plot_normal(
            time_days, 
            error_norm, 
            xlabel="Time [days]", 
            ylabel=r"Position Error Norm $||\Delta\mathbf{r}(t)||$ [m]",
            title="Position Error Norm (Q4): Single Arc with RSW Acceleration", 
            filename="Q4_pos_error_norm.pdf", 
            task="q4",
            logy=False
        )
        
        trajectories = {
            'Numerical (with low-thrust)': num_data[:, 1:4],
            'Lambert': lambert_data[:, 1:4]
        }
        plot_3d_orbits(
            trajectories, 
            origin_name="Sun", 
            title="3D Trajectory (Q4): Single Arc RSW Correction", 
            filename="Q4_3d_trajectory.pdf", 
            task="q4"
        )
    
    # --- Plotting 4c (two-arc) ---
    num_file_2 = output_dir / "Q4_two_arc_numerical_states.dat"
    lambert_file_2 = output_dir / "Q4_two_arc_lambert_states.dat"
    if num_file_2.exists() and lambert_file_2.exists():
        num_data_2 = np.loadtxt(num_file_2)
        lambert_data_2 = np.loadtxt(lambert_file_2)
        time_2 = num_data_2[:, 0]
        time_days_2 = (time_2 - time_2[0]) / 86400.0
        pos_error_2 = num_data_2[:, 1:4] - lambert_data_2[:, 1:4]
        error_norm_2 = np.linalg.norm(pos_error_2, axis=1)
        
        plot_normal(
            time_days_2, 
            error_norm_2, 
            xlabel="Time [days]", 
            ylabel=r"Position Error Norm $||\Delta\mathbf{r}(t)||$ [m]",
            title="Position Error Norm (Q4): Two-Arc with RSW Acceleration", 
            filename="Q4_two_arc_pos_error_norm.pdf", 
            task="q4",
            logy=True
        )
        
        trajectories_2 = {
            'Numerical (Two-Arc)': num_data_2[:, 1:4],
            'Lambert': lambert_data_2[:, 1:4]
        }
        plot_3d_orbits(
            trajectories_2, 
            origin_name="Sun", 
            title="3D Trajectory (Q4): Two Arc RSW Correction", 
            filename="Q4_two_arc_3d_trajectory.pdf", 
            task="q4"
        )

    # --- Plotting 4d (Monte Carlo) ---
    mc_file = output_dir / "Q4d_mc_results.dat"
    if mc_file.exists():
        mc_data = np.loadtxt(mc_file)
        deviation = mc_data[:, 0]
        avg_thrust = mc_data[:, 1]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(deviation, avg_thrust, alpha=0.6, edgecolors='none')
        plt.xlabel(r"Deviation from initial guess $||\mathbf{p}_1 - \mathbf{p}^{(b)}||$ [m/s$^2$]", fontsize=LABEL_SIZE)
        plt.ylabel(r"Average Thrust Level $(||\mathbf{p}_1|| + ||\mathbf{p}_2||)/2$ [m/s$^2$]", fontsize=LABEL_SIZE)
        plt.title("Monte Carlo Analysis: Average Thrust vs. Deviation (Q4d)", fontsize=TITLE_SIZE)
        plt.semilogy()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(Path("Plots/q4") / "Q4d_MC_scatter.pdf")
        plt.close()

    # --- Plotting 4d (Thrust components over time) ---
    thrust_file = output_dir / "Q4d_thrust_profiles.dat"
    times_file = output_dir / "Q4d_times.dat"
    if thrust_file.exists() and times_file.exists():
        thrust_profiles = np.loadtxt(thrust_file)
        times = np.loadtxt(times_file)
        
        p_b = thrust_profiles[0]
        p1_opt = thrust_profiles[1]
        p2_opt = thrust_profiles[2]
        
        t0, tmid, tE = times
        t0_days = 0.0
        tmid_days = (tmid - t0) / 86400.0
        tE_days = (tE - t0) / 86400.0
        
        fig, axs = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
        components = ['R', 'S', 'W']
        colors_b = ['b', 'b', 'b']
        colors_opt = ['r', 'r', 'r']
        
        for i in range(3):
            # Model b (single arc)
            axs[i].step([t0_days, tE_days], [p_b[i], p_b[i]], where='post', 
                       label='Model (b) - Single Arc', color=colors_b[i], linestyle='--')
            
            # Model d (optimal two-arc)
            axs[i].step([t0_days, tmid_days, tE_days], [p1_opt[i], p2_opt[i], p2_opt[i]], where='post', 
                       label='Optimal Model (d) - Two Arc', color=colors_opt[i])
            
            axs[i].set_ylabel(f"{components[i]} Acc. [m/s$^2$]", fontsize=LABEL_SIZE)
            axs[i].grid(True, linestyle='--', alpha=0.6)
            if i == 0:
                axs[i].legend(fontsize=LEGEND_SIZE)
                
        axs[-1].set_xlabel("Time [days]", fontsize=LABEL_SIZE)
        fig.suptitle("Thrust Acceleration Components in RSW Frame (Q4)", fontsize=TITLE_SIZE)
        plt.tight_layout()
        plt.savefig(Path("Plots/q4") / "Q4d_thrust_components.pdf")
        plt.close()

    print("Q4 plots generated in Plots/q4/")

# Run plotting functions for selected tasks
if "q1" in task_keys:
    plot_q1_results()

if "q2" in task_keys:
    plot_q2_results()

if "q3" in task_keys:
    plot_q3_results()

if "q4" in task_keys:
    plot_q4_results()
