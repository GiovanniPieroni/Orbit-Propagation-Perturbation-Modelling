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

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 4 #################################################
###########################################################################

if __name__ == "__main__":

    rsw_acceleration_magnitude = [0, 0, 0]

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    ###########################################################################
    # RUN CODE FOR QUESTION 4b ################################################
    ###########################################################################

    # Set start and end times of full trajectory
    departure_epoch_with_buffer = np.loadtxt(output_directory + "Q2_case_iii_numerical_states.dat")[0, 0]
    arrival_epoch_with_buffer = np.loadtxt(output_directory + "Q2_case_iii_numerical_states.dat")[-1, 0]


    # Solve for state transition matrix on current arc
    termination_settings = propagation_setup.propagator.time_termination(
        arrival_epoch_with_buffer 
    )

    variational_equations_solver = propagate_variational_equations(
        departure_epoch_with_buffer,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_rsw_acceleration = True)

    sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
    state_history_var = variational_equations_solver.state_history
    lambert_history_var = get_lambert_arc_history(lambert_arc_ephemeris, state_history_var)

    # dep_var = variational_equations_solver.dependent_variable_history
    # rsw_to_i = dep_var[:, 1:10].reshape(dep_var.shape[0], 3, 3)
    # Compute low-thrust RSW acceleration to meet required final position
    rsw_acceleration_magnitude = np.linalg.inv(sensitivity_matrix_history[arrival_epoch_with_buffer][:3, :3]) @ (lambert_history_var[arrival_epoch_with_buffer][:3] - state_history_var[arrival_epoch_with_buffer][:3])
    # Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
    # input to the propagate_trajectory function
    
    print(f"rsw_acceleration_magnitude (p_b): {rsw_acceleration_magnitude}")
    dynamics_simulator = propagate_trajectory(
        departure_epoch_with_buffer,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        initial_state_correction=np.zeros(6),
        use_rsw_acceleration=True,
        rsw_acceleration_magnitude=rsw_acceleration_magnitude
    )
    
    # Calculate error in 4b
    state_4b = dynamics_simulator.propagation_results.state_history[arrival_epoch_with_buffer]
    target_4b = lambert_arc_ephemeris.cartesian_state(arrival_epoch_with_buffer)
    error_4b = np.linalg.norm(target_4b[:3] - state_4b[:3])
    print(f"Error in 4b: {error_4b} m")

    write_propagation_results_to_file(
            dynamics_simulator,
            lambert_arc_ephemeris,
            "Q4_" + 'single_arc',
            output_directory,
        )

    # Save to row 13
    filename_save1 = "CartesianResults_AE4868_2026_2_6541151.dat"
    data_13 = [arrival_epoch_with_buffer] + list(state_4b)
    save_to_row(filename_save1, data_13, 13)

    state_history = dynamics_simulator.propagation_results.state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)


    ###########################################################################
    # RUN CODE FOR QUESTION 4c ################################################
    ###########################################################################

    # Define mid-epoch
    mid_epoch = (departure_epoch_with_buffer + arrival_epoch_with_buffer) / 2

    # Propagate first arc [t0, tmid] with p1 = p_b
    termination_settings_1 = propagation_setup.propagator.time_termination(mid_epoch)
    dynamics_simulator_1 = propagate_trajectory(
        departure_epoch_with_buffer,
        termination_settings_1,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        initial_state_correction=np.zeros(6),
        use_rsw_acceleration=True,
        rsw_acceleration_magnitude=rsw_acceleration_magnitude
    )

    state_history_1 = dynamics_simulator_1.propagation_results.state_history
    final_state_arc_1 = state_history_1[mid_epoch]

    # Initialize p2 with p_b
    p2 = rsw_acceleration_magnitude.copy()
    
    # Iterate to find p2 such that final position matches lambert arc
    tolerance = 1.0 # 1 meter
    max_iterations = 50
    for i in range(max_iterations):
        # Propagate Arc 2 [tmid, tE] with current p2
        termination_settings_2 = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)
        
        initial_state_correction_2 = final_state_arc_1 - lambert_arc_ephemeris.cartesian_state(mid_epoch)
        
        variational_equations_solver_2 = propagate_variational_equations(
            mid_epoch,
            termination_settings_2,
            bodies,
            lambert_arc_ephemeris,
            initial_state_correction=initial_state_correction_2,
            use_rsw_acceleration=True,
            rsw_acceleration_magnitude=p2
        )
        
        state_history_2 = variational_equations_solver_2.state_history
        sensitivity_matrix_history_2 = variational_equations_solver_2.sensitivity_matrix_history
        
        r_final = state_history_2[arrival_epoch_with_buffer][:3]
        r_target = lambert_arc_ephemeris.cartesian_state(arrival_epoch_with_buffer)[:3]
        
        error = r_target - r_final
        error_norm = np.linalg.norm(error)
        
        print(f"Iteration {i}: Error norm = {error_norm} m")
        
        if error_norm < tolerance:
            break
            
        S2 = sensitivity_matrix_history_2[arrival_epoch_with_buffer][:3, :3]
        p2 += np.linalg.inv(S2) @ error

    # Propagate full trajectory for plotting and saving
    dynamics_simulator_2 = propagate_trajectory(
        mid_epoch,
        termination_settings_2,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        initial_state_correction=initial_state_correction_2,
        use_rsw_acceleration=True,
        rsw_acceleration_magnitude=p2
    )
    
    # Combine state histories
    full_state_history = state_history_1.copy()
    full_state_history.update(dynamics_simulator_2.propagation_results.state_history)
    
    # Save results for 4c
    save2txt(full_state_history, output_directory + "Q4_two_arc_numerical_states.dat")
    
    lambert_history_full = get_lambert_arc_history(lambert_arc_ephemeris, full_state_history)
    save2txt(lambert_history_full, output_directory + "Q4_two_arc_lambert_states.dat")

    # Add to save file 1: Row 14
    final_epoch = arrival_epoch_with_buffer
    final_state = full_state_history[final_epoch]
    data_14 = [final_epoch] + list(final_state)
    save_to_row(filename_save1, data_14, 14)

    print(f"p2 value: {p2}")
    print(f"p_b value: {rsw_acceleration_magnitude}")

    ###########################################################################
    # RUN CODE FOR QUESTION 4d ################################################
    ###########################################################################
    print("\n--- Running Monte Carlo Analysis (Q4d) ---")
    np.random.seed(42) # Set seed for reproducibility
    num_samples = 1000
    p_b_norm = np.linalg.norm(rsw_acceleration_magnitude)
    p1_samples = np.random.normal(loc=rsw_acceleration_magnitude, scale=0.4 * p_b_norm, size=(num_samples, 3))
    
    mc_results = []
    
    optimal_p1 = None
    optimal_p2 = None
    min_avg_thrust = float('inf')
    
    for idx in range(num_samples):
        # if idx % 100 == 0:
        print(f"Running sample {idx}/{num_samples}")
            
        p1 = p1_samples[idx]
        
        # Propagate Arc 1 [t0, tmid] with current p1
        termination_settings_1 = propagation_setup.propagator.time_termination(mid_epoch)
        sim_1_mc = propagate_trajectory(
            departure_epoch_with_buffer,
            termination_settings_1,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations=True,
            initial_state_correction=np.zeros(6),
            use_rsw_acceleration=True,
            rsw_acceleration_magnitude=p1
        )
        final_state_arc_1_mc = sim_1_mc.propagation_results.state_history[mid_epoch]
        
        # Initialize p2 for root finding
        p2_mc = p1.copy()
        
        # Iterate to find p2_mc
        initial_state_correction_2_mc = final_state_arc_1_mc - lambert_arc_ephemeris.cartesian_state(mid_epoch)
        
        for i in range(max_iterations):
            termination_settings_2 = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)
            var_sim_2_mc = propagate_variational_equations(
                mid_epoch,
                termination_settings_2,
                bodies,
                lambert_arc_ephemeris,
                initial_state_correction=initial_state_correction_2_mc,
                use_rsw_acceleration=True,
                rsw_acceleration_magnitude=p2_mc
            )
            r_final_mc = var_sim_2_mc.state_history[arrival_epoch_with_buffer][:3]
            r_target = lambert_arc_ephemeris.cartesian_state(arrival_epoch_with_buffer)[:3]
            error_mc = r_target - r_final_mc
            error_norm_mc = np.linalg.norm(error_mc)
            
            if error_norm_mc < tolerance:
                break
                
            S2_mc = var_sim_2_mc.sensitivity_matrix_history[arrival_epoch_with_buffer][:3, :3]
            p2_mc += np.linalg.inv(S2_mc) @ error_mc
            
        avg_thrust = (np.linalg.norm(p1) + np.linalg.norm(p2_mc)) / 2.0
        deviation = np.linalg.norm(p1 - rsw_acceleration_magnitude)
        mc_results.append([deviation, avg_thrust])
        
        if avg_thrust < min_avg_thrust:
            min_avg_thrust = avg_thrust
            optimal_p1 = p1
            optimal_p2 = p2_mc

    print(f"Optimal p1: {optimal_p1}")
    print(f"Optimal p2: {optimal_p2}")
    print(f"Minimum average thrust: {min_avg_thrust}")
    
    # Save MC results for plotting
    np.savetxt(output_directory + "Q4d_mc_results.dat", np.array(mc_results))
    
    # Save thrust profiles for plotting
    thrust_profiles = np.array([
        rsw_acceleration_magnitude,  # p_b
        optimal_p1,                  # optimal p1
        optimal_p2                   # optimal p2
    ])
    np.savetxt(output_directory + "Q4d_thrust_profiles.dat", thrust_profiles)
    
    times = np.array([departure_epoch_with_buffer, mid_epoch, arrival_epoch_with_buffer])
    np.savetxt(output_directory + "Q4d_times.dat", times)
