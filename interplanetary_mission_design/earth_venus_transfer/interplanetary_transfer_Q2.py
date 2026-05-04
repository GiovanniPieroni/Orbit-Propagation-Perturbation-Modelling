""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import os

from interplanetary_transfer_helper_functions import *
from Modules.utility import save_to_row

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

def load_simulation_results(file_path):
    """Loads simulation results from a .dat file into a dictionary."""
    data = np.loadtxt(file_path)
    # Column 0 is time, columns 1-7 are the state vector
    return {row[0]: row[1:] for row in data}

###########################################################################
# RUN CODE FOR QUESTION 2 #################################################
###########################################################################

if __name__ == "__main__":

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    # Grab the Q1 results so we can sample the Lambert arc at the exact same epochs
    q1_file = output_directory + "Q1_lambert_states.dat"
    if os.path.exists(q1_file):
        simulation_result_dict = load_simulation_results(q1_file)
        
        # Reconstruct the Lambert history from Q1 timestamps
        lambert_arc_state_history = np.hstack((
            np.vstack(list(get_lambert_arc_history(lambert_arc_ephemeris, simulation_result_dict).keys())),
            np.vstack(list(get_lambert_arc_history(lambert_arc_ephemeris, simulation_result_dict).values()))))
    else:
        print(f"Warning: {q1_file} not found. Run Q1 first.")


    
    """
    case_i: The initial and final propagation time equal to the initial and final times of the Lambert arc.
    case_ii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour.
    case_iii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t such that we start/end on the sphere of influence
    case_iv: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour. The propagation is started from the middle point in time of the Lambert arc and propagated forward and backward in time.

    """
    # List cases to iterate over. STUDENT NOTE: feel free to modify if you see fit
    cases = ["case_i", "case_ii", "case_iii", "case_iv", "case_iv_500", "case_iv_250"]


    #  Sphere of Influence (SOI) Calculations 
    mu_earth = bodies.get('Earth').gravitational_parameter
    mu_venus = bodies.get('Venus').gravitational_parameter
    mu_sun = bodies.get('Sun').gravitational_parameter

    earth_sun_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name=global_frame_origin,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch,
    )
    venus_sun_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Venus",
        observer_body_name=global_frame_origin,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch,
    )

    # Calculate SOI radii
    R_soi_earth = (mu_earth / mu_sun) ** (2.0 / 5.0) * np.linalg.norm(earth_sun_state[:3])
    R_soi_venus = (mu_venus / mu_sun) ** (2.0 / 5.0) * np.linalg.norm(venus_sun_state[:3])


    # Sample the Lambert arc to find the crossing
    test_epochs = np.arange(departure_epoch, departure_epoch+10*86400.0, 1.0)
    lambert_state = np.array([lambert_arc_ephemeris.cartesian_state(t) for t in test_epochs])
    
    earth_state = np.array([spice.get_body_cartesian_state_at_epoch("Earth", "Sun", global_frame_orientation, "NONE", t) for t in test_epochs])
    dist_earth = np.linalg.norm(lambert_state[:, :3]- earth_state[:, :3], axis=1)
    R_soi_earth = (mu_earth / mu_sun) ** (2.0 / 5.0) * np.linalg.norm(earth_state[:, :3], axis=1)# [:, np.newaxis] * lambert_state[:, 3:6]/np.linalg.norm(lambert_state[:, 3:6], axis=1)[:, np.newaxis]
    earth_soi_crossing_time = test_epochs[np.argmin(np.abs(dist_earth - R_soi_earth))]
    # print(earth_soi_crossing_time)
    # # Find the exact moment we cross Earth's SOI (minimizing distance to SOI surface)
    # merit_earth = np.linalg.norm(lambert_arc_state_history[:, 1:4] - earth_sun_state[:3], axis=1) - R_soi_earth
    # # earth_soi_crossing_time = lambert_arc_state_history[np.argmin(np.abs(merit_earth)), 0]

 

    # Define start and end times for each scenario
    time_definitions = {
        "case_i": (departure_epoch, arrival_epoch),
        "case_ii": (departure_epoch + 3600.0, arrival_epoch - 3600.0),
        "case_iii": (
            earth_soi_crossing_time,
            arrival_epoch
        ),
        "case_iv": (
            departure_epoch + 3600.0, arrival_epoch - 3600.0 
        ),
        "case_iv_500": (
            departure_epoch + 3600.0, arrival_epoch - 3600.0 
        ),
        "case_iv_250": (
            departure_epoch + 3600.0, arrival_epoch - 3600.0    
        )
    }

    # Calculate initial state corrections if we aren't starting at t=0
    # This aligns the state with the shifted start time
    # lambert_arc_initial_correction = {
    #     'case_i': np.zeros(6),
    #     'case_ii': lambert_arc_state_history[lambert_arc_state_history[:, 0] == time_definitions['case_ii'][0], 1:7] - lambert_arc_state_history[0, 1:7],
    #     'case_iii': lambert_arc_state_history[lambert_arc_state_history[:, 0] == time_definitions['case_iii'][0], 1:7] - lambert_arc_state_history[0, 1:7],
    #     'case_iv': lambert_arc_state_history[lambert_arc_state_history[:, 0] == time_definitions['case_iv'][0], 1:7] - lambert_arc_state_history[0, 1:7],
    #     'case_iv_500': lambert_arc_state_history[lambert_arc_state_history[:, 0] == time_definitions['case_iv_500'][0], 1:7] - lambert_arc_state_history[0, 1:7],
    #     'case_iv_250': lambert_arc_state_history[lambert_arc_state_history[:, 0] == time_definitions['case_iv_250'][0], 1:7] - lambert_arc_state_history[0, 1:7],
    # }


    # Run propagation for each of cases i-iii
    for case in cases:

        # Define the initial and final propagation time for the current case
        departure_epoch_with_buffer = time_definitions[case][0]
        arrival_epoch_with_buffer = time_definitions[case][1]
        print(f"\nRunning {case} with departure epoch: {departure_epoch_with_buffer} and arrival epoch: {arrival_epoch_with_buffer}")
        
        termination_settings  = propagation_setup.propagator.time_termination(arrival_epoch_with_buffer)

        if case == 'case_iv_500':
            # Case IV: Start from the middle and propagate fwd and bwd
            bodies.get_body('Spacecraft').mass = 500.0
        elif case == 'case_iv_250': 
            bodies.get_body('Spacecraft').mass = 250.0
        else:
            bodies.get_body('Spacecraft').mass = pick('vehicle_mass')


        if case == 'case_iii':
            # Case III: Stop when we reach Venus SOI (or time runs out)
            # We use a hybrid termination to check both conditions
            time_termination_settings = propagation_setup.propagator.time_termination(
                arrival_epoch_with_buffer
            )

            # Stop if distance to Venus < SOI
            termination_variable = propagation_setup.dependent_variable.relative_distance(
                "Spacecraft", "Venus"
            )
            altitude_termination_settings = (
                propagation_setup.propagator.dependent_variable_termination(
                    dependent_variable_settings=termination_variable,
                    limit_value=R_soi_venus,
                    use_as_lower_limit=True,
                )
            )

            termination_settings = propagation_setup.propagator.hybrid_termination(
                [time_termination_settings, altitude_termination_settings], 
                fulfill_single_condition=True
            )

        if case == 'case_iv' or case == 'case_iv_500' or case == 'case_iv_250':
            # Case IV: Start from the middle and propagate both Forward and Backward
            termination_time_fwd = time_definitions[case][1]
            time_termination_settings_fwd = propagation_setup.propagator.time_termination(
                termination_time_fwd
            )

            termination_time_bwd = time_definitions[case][0]
            time_termination_settings_bwd = propagation_setup.propagator.time_termination(
                termination_time_bwd
            )

            termination_settings = propagation_setup.propagator.non_sequential_termination(
                time_termination_settings_fwd,
                time_termination_settings_bwd
            )
            
            # Shift start time to the midpoint
            departure_epoch_with_buffer = departure_epoch_with_buffer + int(termination_time_fwd - termination_time_bwd)/2

        
       
        # Run the simulation 
        dynamics_simulator = propagate_trajectory(
            departure_epoch_with_buffer,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations=True,
            # initial_state_correction=lambert_arc_initial_correction[case]
        )
        write_propagation_results_to_file(
            dynamics_simulator,
            lambert_arc_ephemeris,
            "Q2_" + str(case),
            output_directory,
        )

        state_history = dynamics_simulator.propagation_results.state_history
        acc_history = dynamics_simulator.propagation_results.dependent_variable_history

        lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)
        
        # Extract arrays for vectorized calculation
        epochs = sorted(list(state_history.keys()))
        state_arr = np.vstack([state_history[t] for t in epochs])
        lambert_arr = np.vstack([lambert_history[t] for t in epochs])
        acc_arr = np.vstack([acc_history[t] for t in epochs])

        # Calculate Lambert acceleration
        # a = -mu * r / |r|^3
        r_vec = lambert_arr[:, :3]
        r_norm = np.linalg.norm(r_vec, axis=1).reshape(-1, 1)
        lambert_acc = - mu_sun * r_vec / (r_norm**3)

        # Calculate residuals and convert to dictionaries
        pos_residual = dict(zip(epochs, state_arr[:, :3] - lambert_arr[:, :3]))
        vel_residual = dict(zip(epochs, state_arr[:, 3:6] - lambert_arr[:, 3:6]))
        acc_residual = dict(zip(epochs, acc_arr - lambert_acc))

        save2txt(
            solution=pos_residual,
            filename= str(case) + "_pos_residual.dat",
            directory="./Data/q2",
        )
        save2txt(
            solution=vel_residual,
            filename= str(case) + "_vel_residual.dat",
            directory="./Data/q2",
        )
        save2txt(
            solution=acc_residual,
            filename=str(case) + "_acc_residual.dat",
            directory="./Data/q2",
        )
        




        # power = 1367.0 * 4 * np.pi * constants.ASTRONOMICAL_UNIT**2

        # acc_rad_sun_analyt = power * 100 * 1.2 / (4 * np.pi * constants.SPEED_OF_LIGHT * 2e3) * r_sun_juice / r_sun_juice_norm[:, None]**3               

        # np.savetxt(f'./Data/{current_task}/acc_rad_sun_analyt.txt', acc_rad_sun_analyt)



        # Save Cartesian states if requested
        if pick('save_cat_state'):
            filename = "CartesianResults_AE4868_2026_2_6541151.dat"
            
            # initial_time = min(state_history.keys())
            initial_time = epochs[0]
            initial_state = state_history[initial_time]
            # final_time = max(state_history.keys())
            final_time = epochs[-1]
            final_state = state_history[final_time]
            
            if case == 'case_i':
                save_to_row(filename, [initial_time] + list(initial_state), 3)
                save_to_row(filename, [final_time] + list(final_state), 4)
            elif case == 'case_ii':
                save_to_row(filename, [initial_time] + list(initial_state), 5)
                save_to_row(filename, [final_time] + list(final_state), 6)
            elif case == 'case_iii':
                save_to_row(filename, [initial_time] + list(initial_state), 7)
                save_to_row(filename, [final_time] + list(final_state), 8)