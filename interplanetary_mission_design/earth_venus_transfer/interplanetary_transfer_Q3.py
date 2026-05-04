""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from interplanetary_transfer_helper_functions import *
from Modules.utility import save_to_row
import pandas as pd

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 3 #################################################
###########################################################################

if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    ##############################################################
    # 0. FIND START AND END TIMES (CASE III)
    ##############################################################
    
    # SOI Radii
    mu_earth = bodies.get('Earth').gravitational_parameter
    mu_venus = bodies.get('Venus').gravitational_parameter
    mu_sun = bodies.get('Sun').gravitational_parameter

    # Earth/Venus states at departure/arrival to get orbital radius for SOI calculation
    earth_sun_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Earth", observer_body_name="Sun",
        reference_frame_name=global_frame_orientation, aberration_corrections="NONE", ephemeris_time=departure_epoch
    )
    venus_sun_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Venus", observer_body_name="Sun",
        reference_frame_name=global_frame_orientation, aberration_corrections="NONE", ephemeris_time=arrival_epoch
    )

    R_soi_earth = (mu_earth / mu_sun) ** (2.0 / 5.0) * np.linalg.norm(earth_sun_state[:3])
    R_soi_venus = (mu_venus / mu_sun) ** (2.0 / 5.0) * np.linalg.norm(venus_sun_state[:3])

    #Sample the Lambert arc to find the crossing
    test_dep_epochs = np.arange(departure_epoch, departure_epoch+2*86400.0, 1.0)
    lambert_pos_dep = np.array([lambert_arc_ephemeris.cartesian_state(t)[:3] for t in test_dep_epochs])
    earth_pos = np.array([spice.get_body_cartesian_state_at_epoch("Earth", "Sun", global_frame_orientation, "NONE", t)[:3] for t in test_dep_epochs])
    dist_earth = np.linalg.norm(lambert_pos_dep - earth_pos, axis=1)
    earth_soi_crossing_time = test_dep_epochs[np.argmin(np.abs(dist_earth - R_soi_earth))]


    # test_arr_epochs = np.arange(arrival_epoch - 2*86400.0, arrival_epoch, 1.0)
    # lambert_pos_arr = np.array([lambert_arc_ephemeris.cartesian_state(t)[:3] for t in test_arr_epochs])
    # venus_pos = np.array([spice.get_body_cartesian_state_at_epoch("venus", "Sun", global_frame_orientation, "NONE", t)[:3] for t in test_arr_epochs])
    # R_soi_venus = (mu_venus / mu_sun) ** (2.0 / 5.0) * np.linalg.norm(venus_pos[:, :3])
    # dist_venus = np.linalg.norm(lambert_pos_arr - venus_pos, axis=1)
    # venus_soi_crossing_time = test_arr_epochs[np.argmin(np.abs(dist_venus - R_soi_venus))]
    # print(venus_soi_crossing_time)
    # exit()

    # Find Venus SOI crossing time by propagating Case iii
    # Termination: Distance to Venus < SOI or time > arrival_epoch
    termination_settings_iii = propagation_setup.propagator.hybrid_termination(
        [
            propagation_setup.propagator.time_termination(arrival_epoch),
            propagation_setup.propagator.dependent_variable_termination(
                dependent_variable_settings=propagation_setup.dependent_variable.relative_distance("Spacecraft", "Venus"),
                limit_value=R_soi_venus,
                use_as_lower_limit=True
            )
        ],
        fulfill_single_condition=True
    )
    
    # Note: Case iii start state is Lambert at earth_soi_crossing_time
    # initial_state_correction = Lambert(t_start) - Lambert(t_0)
    # But propagate_trajectory takes initial_time and uses Lambert(initial_time) + correction
    # So if we want to start at Lambert(t_start), correction is 0.
    
    dynamics_simulator_iii = propagate_trajectory(
        earth_soi_crossing_time,
        termination_settings_iii,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        initial_state_correction=np.zeros(6)
    )
    case_iii_epochs = sorted(list(dynamics_simulator_iii.propagation_results.state_history.keys()))
    t_start = case_iii_epochs[0]
    t_end = case_iii_epochs[-1]

    # print(f"Arc start time: {t_start}")
    # print(f"Arc end time:   {t_end}")
    # print(f"Total duration: {(t_end - t_start)/86400.0:.2f} days")

    ##############################################################
    # 1. DIVIDE INTO 10 ARCS
    ##############################################################
    number_of_arcs = 10
    # Select indices from the case_iii epoch history to ensure each boundary is a multiple of 3600s
    # from the start time, as the integrator used a fixed step size of 3600s.
    indices = np.linspace(0, len(case_iii_epochs) - 1, number_of_arcs + 1).astype(int)
    arc_times = np.array([case_iii_epochs[i] for i in indices])
    print(f"Arc boundaries: {arc_times}")
     
    arc_results_a = []
    arc_results_c = []
    arc_results_d = []
    
    iteration_counts = []
    total_delta_v = 0.0
    
    # Store results for Save File 
    rows_to_save = {}

    ##############################################################
    # 2. PROPAGATE ARCS
    ##############################################################
    
    delta_v_maneuvers = []

    for arc_index in range(number_of_arcs):
        t0_arc = arc_times[arc_index]
        tf_arc = arc_times[arc_index + 1]
        print(f"\nProcessing Arc {arc_index}: {t0_arc} to {tf_arc}")

        # Target final position (Lambert)
        target_final_state = lambert_arc_ephemeris.cartesian_state(tf_arc)
        target_final_pos = target_final_state[:3]

        # Standard time termination for most cases
        term_a = propagation_setup.propagator.time_termination(tf_arc)


        ###########################################################################
        # QUESTION 3a: Arcwise propagation with Lambert initial state
        ###########################################################################

        sim_a = propagate_trajectory(t0_arc, term_a, bodies, lambert_arc_ephemeris, use_perturbations=True)
        write_propagation_results_to_file(sim_a, lambert_arc_ephemeris, f"Q3a_arc_{arc_index}", output_directory)
        
        # Save File 1 rows
        state_history_a = sim_a.propagation_results.state_history
        epochs_a = sorted(list(state_history_a.keys()))
        if arc_index == 0:
            rows_to_save[9] = [epochs_a[0]] + list(state_history_a[epochs_a[0]])
            rows_to_save[10] = [epochs_a[-1]] + list(state_history_a[epochs_a[-1]])
        elif arc_index == 4:
            rows_to_save[11] = [epochs_a[0]] + list(state_history_a[epochs_a[0]])
            rows_to_save[12] = [epochs_a[-1]] + list(state_history_a[epochs_a[-1]])


        ###########################################################################
        # QUESTION 3c: Single-step correction
        ###########################################################################
        # We need the STM. We propagate variational equations starting from Lambert.
        var_sim = propagate_variational_equations(t0_arc, term_a, bodies, lambert_arc_ephemeris)
        stm_history = var_sim.state_transition_matrix_history
        state_history_var = var_sim.state_history
        epochs_var = sorted(list(state_history_var.keys()))


        
        final_epoch_var = epochs_var[-1]
        final_stm = stm_history[final_epoch_var]
        final_state_num = state_history_var[final_epoch_var]
        
        # Target final position is Lambert at the ACTUAL final epoch of propagation
        target_final_pos_actual = lambert_arc_ephemeris.cartesian_state(final_epoch_var)[:3]
        pos_error = final_state_num[:3] - target_final_pos_actual
        phi_rv = final_stm[0:3, 3:6]
        dv_corr = -np.linalg.inv(phi_rv) @ pos_error
        
        corr_c = np.zeros(6)
        corr_c[3:] = dv_corr
        
        sim_c = propagate_trajectory(t0_arc, term_a, bodies, lambert_arc_ephemeris, use_perturbations=True, initial_state_correction=corr_c)
        write_propagation_results_to_file(sim_c, lambert_arc_ephemeris, f"Q3c_arc_{arc_index}", output_directory)

        ###########################################################################
        # QUESTION 3d: Iterative correction
        ###########################################################################
        
        # Determine termination settings for Case D
        if arc_index == number_of_arcs - 1:
            termination_d = termination_settings_iii
        else:
            termination_d = term_a

        current_corr = corr_c.copy()
        iters = 1
        state_history_c = sim_c.propagation_results.state_history
        epochs_c = sorted(list(state_history_c.keys()))
        final_epoch_c = epochs_c[-1]
        pos_error_norm = np.linalg.norm(state_history_c[final_epoch_c][:3] - lambert_arc_ephemeris.cartesian_state(final_epoch_c)[:3])
        
        while pos_error_norm > 1.0:
            # Re-run variational equations from the current corrected initial state
            var_sim_iter = propagate_variational_equations(t0_arc, termination_d, bodies, lambert_arc_ephemeris, initial_state_correction=current_corr)
            state_history_iter_var = var_sim_iter.state_history
            epochs_iter_var = sorted(list(state_history_iter_var.keys()))
            final_epoch_iter = epochs_iter_var[-1]
            
            final_stm_iter = var_sim_iter.state_transition_matrix_history[final_epoch_iter]
            final_pos_iter = state_history_iter_var[final_epoch_iter][:3]
            
            error_iter = final_pos_iter - lambert_arc_ephemeris.cartesian_state(final_epoch_iter)[:3]
            phi_rv_iter = final_stm_iter[0:3, 3:6]
            dv_update = -np.linalg.inv(phi_rv_iter) @ error_iter
            
            current_corr[3:] += dv_update
            
            # Check convergence
            sim_iter = propagate_trajectory(t0_arc, termination_d, bodies, lambert_arc_ephemeris, use_perturbations=True, initial_state_correction=current_corr)
            state_history_iter = sim_iter.propagation_results.state_history
            epochs_iter = sorted(list(state_history_iter.keys()))
            final_epoch_iter_check = epochs_iter[-1]
            pos_error_norm = np.linalg.norm(state_history_iter[final_epoch_iter_check][:3] - lambert_arc_ephemeris.cartesian_state(final_epoch_iter_check)[:3])
            print(f"  Iteration {iters}: Error Norm = {pos_error_norm:.4e} m")
            iters += 1
            if iters > 1000: break # Safety break

        iteration_counts.append(iters)
        sim_d = propagate_trajectory(t0_arc, termination_d, bodies, lambert_arc_ephemeris, use_perturbations=True, initial_state_correction=current_corr)
        write_propagation_results_to_file(sim_d, lambert_arc_ephemeris, f"Q3d_arc_{arc_index}", output_directory)

        ###########################################################################
        # QUESTION 3e: Delta V calculation
        ###########################################################################
        state_history_d = sim_d.propagation_results.state_history
        epochs_d = sorted(list(state_history_d.keys()))
        
        # Initial maneuver: correction to Lambert initial state
        dv_initial = np.linalg.norm(current_corr[3:])
        
        # Final maneuver: from numerical end velocity to Lambert end velocity
        v_end_arc = state_history_d[epochs_d[-1]][3:]
        v_lambert_end = lambert_arc_ephemeris.cartesian_state(epochs_d[-1])[3:]
        dv_final = np.linalg.norm(v_end_arc - v_lambert_end)
        
        dv_total_arc = dv_initial + dv_final
        total_delta_v += dv_total_arc

        delta_v_maneuvers.append(dv_total_arc)


    # Final arrival maneuver at Venus?
    # The question asks for Delta V to fly the trajectory. 
    # Usually, this includes the arrival maneuver if we want to "arrive" (match velocity).
    # But often in these assignments, it's just the sum of correction maneuvers + departure.
    # Let's check the target velocity.
    # v_target_final = spice.get_body_cartesian_state_at_epoch("Venus", "Sun", global_frame_orientation, "NONE", t_end)[3:]
    # dv_arrival = np.linalg.norm(prev_final_velocity - v_target_final)
    # total_delta_v += dv_arrival

    print(f"\nTotal Delta V: {total_delta_v:.2f} m/s")
    print("Iteration counts per arc:")
    print(iteration_counts)

    # Save to file 1
    filename_save1 = "CartesianResults_AE4868_2026_2_6541151.dat"
    for row, data in rows_to_save.items():
        save_to_row(filename_save1, data, row)

    # Export iteration table (optional, but requested in d)
    iter_df = pd.DataFrame({"Arc": range(number_of_arcs), "Iterations": iteration_counts, "Delta V": delta_v_maneuvers})
    print(iter_df.to_string(index=False))
