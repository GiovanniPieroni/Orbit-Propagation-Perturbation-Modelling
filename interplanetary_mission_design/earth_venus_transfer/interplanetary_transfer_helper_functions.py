""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import numpy as np
from tudatpy import constants
from tudatpy.astro import element_conversion, two_body_dynamics
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, environment, propagation_setup, propagation, parameters_setup, parameters, simulator


# My imports
from Configuration.debug import pick


# Define departure/arrival epoch - in seconds since J2000
departure_epoch = pick('departure_epoch')
time_of_flight = pick('time_of_flight')
arrival_epoch = departure_epoch + time_of_flight
target_body = pick('target_body') or "Venus"
global_frame_orientation = "ECLIPJ2000"
global_frame_origin = "Sun"
fixed_step_size = 3600.0




################ HELPER FUNCTIONS: DO NOT MODIFY ########################################


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def write_propagation_results_to_file(
    dynamics_simulator: simulator.SingleArcSimulator,
    lambert_arc_ephemeris: environment.Ephemeris,
    file_output_identifier: str,
    output_directory: str,
) -> None:
    """
    This function will write the results of a numerical propagation, as well as the Lambert arc states at the epochs of the
    numerical state history, to a set of files. Two files are always written when calling this function (numerical state history, a
    and Lambert arc state history). If any dependent variables are saved during the propagation, those are also saved to a file

    Parameters
    ----------
    dynamics_simulator : simulator.SingleArcSimulator
        Object that was used to propagate the dynamics, and which contains the numerical state and dependent variable results

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    file_output_identifier : str
        Name that will be used to correctly save the output data files

    output_directory : str
        Directory to which the files will be written

    Files written
    -------------

    <output_directory/file_output_identifier>_numerical_states.dat
    <output_directory/file_output_identifier>_dependent_variables.dat
    <output_directory/file_output_identifier>_lambert_states.dat


    Return
    ------
    None

    """

    propagation_results = dynamics_simulator.propagation_results

    # Save numerical states
    state_history = propagation_results.state_history
    save2txt(
        solution=state_history,
        filename=output_directory + file_output_identifier + "_numerical_states.dat",
        directory="./",
    )

    # Save dependent variables
    dependent_variables = propagation_results.dependent_variable_history
    if len(dependent_variables.keys()) > 0:
        save2txt(
            solution=dependent_variables,
            filename=output_directory
            + file_output_identifier
            + "_dependent_variables.dat",
            directory="./",
        )

    # Save Lambert arc states
    lambert_arc_states = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    save2txt(
        solution=lambert_arc_states,
        filename=output_directory + file_output_identifier + "_lambert_states.dat",
        directory="./",
    )

    return


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_problem_result(
    bodies: environment.SystemOfBodies,
    target_body: str,
    departure_epoch: float,
    arrival_epoch: float,
) -> environment.Ephemeris:
    """
    This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
    a target body (at arrival epoch), with the states of Earth and the target body defined
    by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
    assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    target_body : str
        The name (string) of the body to which the Lambert arc is to be computed

    departure_epoch : float
        Epoch at which the departure from Earth's center of mass is to take place

    arrival_epoch : float
        Epoch at which the arrival at he target body's center of mass is to take place

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory. This Keplerian trajectory defines the transfer
    from Earth to the target body according to the inputs to this function. Note that this Ephemeris object
    is valid before the departure epoch, and after the arrival epoch, and simply continues (forwards and backwards)
    the unperturbed Sun-centered orbit, as fully defined by the unperturbed transfer arc
    """

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body(
        "Sun"
    ).gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch,
    )

    final_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch,
    )

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3],
        final_state[:3],
        arrival_epoch - departure_epoch,
        central_body_gravitational_parameter,
    )

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(
        lambert_arc_initial_state, central_body_gravitational_parameter
    )

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(
            lambert_arc_keplerian_elements,
            departure_epoch,
            central_body_gravitational_parameter,
        ),
        "",  # for keplerian ephemeris, this argument does not have an effect
    )

    return kepler_ephemeris


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_arc_history(
    lambert_arc_ephemeris: environment.Ephemeris, simulation_result: dict
) -> dict:
    """
    This function extracts the state history (as a dict with time as keys, and Cartesian states as values)
    from an Ephemeris object defined by a lambert solver. This function takes a dictionary of states (simulation_result)
    as input, iterates over the keys of this dict (which represent times) to ensure that the times
    at which this function returns the states of the lambert arcs are identical to those at which the
    simulation_result has (numerically calculated) states


    Parameters
    ----------
    lambert_arc_ephemeris : environment.Ephemeris
        Ephemeris object from which the states are to be extracted

    simulation_result : dict
        Dictionary of (numerically propagated) states, from which the keys
        are used to determine the times at which this function is to extract states
        from the lambert arc

    Return
    ------
    Dictionary of Cartesian states of the lambert arc, with the keys (epochs) being the same as those of the input
    simulation_result and the corresponding Cartesian states of the Lambert arc.
    """

    lambert_arc_states = dict()
    for epoch in simulation_result:
        lambert_arc_states[epoch] = lambert_arc_ephemeris.cartesian_state(epoch)

    return lambert_arc_states


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_trajectory(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    use_perturbations: bool,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0,0,0])
) -> simulator.SingleArcSimulator:
    """
    This function will be repeatedly called throughout the assignment. Propagates the trajectory based
    on several input parameters

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    use_perturbations : bool
        Boolean to indicate whether a perturbed (True) or unperturbed (False) trajectory
        is propagated

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Dynamics simulator object from which the state- and dependent variable history can be extracted

    """

    # Compute initial state along Lambert arc (and apply correction if needed)
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings for perturbed/unperturbed forwards/backwards arcs
    if use_perturbations:
        propagator_settings = get_perturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, termination_condition, use_rsw_acceleration, rsw_acceleration_magnitude
        )

    else:
        propagator_settings = get_unperturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, termination_condition
        )

    # Propagate dynamics with required settings
    dynamics_simulator = simulator.create_dynamics_simulator(
        bodies, propagator_settings
    )

    return dynamics_simulator


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_variational_equations(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0,0,0])
) -> simulator.SingleArcVariationalSimulator:
    """
    Propagates the variational equations for a given range of epochs for a perturbed trajectory.

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    # Compute initial state along Lambert arc
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings
    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        termination_condition,
        use_rsw_acceleration,
        rsw_acceleration_magnitude
    )

    # Define parameters for variational equations
    sensitivity_parameters = get_sensitivity_parameter_set(propagator_settings, bodies, use_rsw_acceleration)

    # Propagate variational equations
    variational_equations_solver = (
        simulator.create_variational_equations_solver(
            bodies, propagator_settings, sensitivity_parameters
        )
    )

    return variational_equations_solver


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_sensitivity_parameter_set(
    propagator_settings: propagation_setup.propagator.PropagatorSettings,
    bodies: environment.SystemOfBodies,
    use_rsw_acceleration: bool = False
) -> parameters.EstimatableParameterSet:
    """
    Function creating the parameters for which the variational equations are to be solved.

    Parameters
    ----------
    propagator_settings : propagation_setup.propagator.PropagatorSettings
        Settings used for the propagation of the dynamics

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    use_rsw_acceleration : Boolean denoting whether the sensitivity to an RSW acceleration is to be
                           included. Note that this can only be used (set to True) is the acceleration models
                           in propagator_settings contain an empirical acceleration

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """
    parameter_settings = parameters_setup.initial_states(
        propagator_settings, bodies
    )

    if use_rsw_acceleration:
        parameter_settings.append(
            parameters_setup.constant_empirical_acceleration_terms(
                "Spacecraft", "Sun"
            )
        )

    return parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )


################ HELPER FUNCTIONS: MODIFY ########################################


# STUDENT CODE TASK - full function (except signature and return)
def get_unperturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for an unperturbed trajectory.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : np.ndarray
        Cartesian initial state of the vehicle in the simulation

    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """

    # Define bodies that are propagated
    bodies_to_propagate = ["Spacecraft"]
    

    # Define accelerations acting on vehicle.
    acceleration_settings_on_vehicle = dict(
        Sun=[propagation_setup.acceleration.point_mass_gravity()]
    )

    # Create global accelerations dictionary.
    acceleration_settings = {"Spacecraft": acceleration_settings_on_vehicle}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, [global_frame_origin]
    )


    # Create numerical integrator settings.
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        [global_frame_origin],
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        initial_time,
        integrator_settings,
        termination_condition,
    )

    return propagator_settings


# STUDENT CODE TASK - full function (except signature and return)
def get_perturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0,0,0])
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for a perturbed trajectory.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : np.ndarray
        Cartesian initial state of the vehicle in the simulation

    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Propagation settings of the perturbed trajectory.
    """
    

    # bodies.get_body('Spacecraft').mass = pick('vehicle_mass')

    ###########################################################################
    # CREATE ACCELERATIONS ####################################################
    ###########################################################################

    # Define bodies that are propagated, and their central bodies of propagation.
    bodies_to_propagate = ["Spacecraft"]
    central_bodies = [global_frame_origin]

    # Define accelerations acting on vehicle.
    acceleration_settings_on_vehicle = dict(
        Venus=[propagation_setup.acceleration.point_mass_gravity()],
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Moon=[propagation_setup.acceleration.point_mass_gravity()],
        Mars=[propagation_setup.acceleration.point_mass_gravity()],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()],

        Sun=[propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.radiation_pressure()
            ],

        Saturn  =[propagation_setup.acceleration.point_mass_gravity()],
    )

    # DO NOT MODIFY, and keep AFTER creation of acceleration_settings_on_spacecraft, but before
    # call to function create_acceleration_models
    # (line is added for compatibility with question 4)
    if use_rsw_acceleration:
        acceleration_settings_on_vehicle["Sun"].append(
            propagation_setup.acceleration.empirical(rsw_acceleration_magnitude))

    # Create global accelerations dictionary.
    acceleration_settings_on_spacecraft = {"Spacecraft": acceleration_settings_on_vehicle}
        

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_on_spacecraft, bodies_to_propagate, central_bodies
    )


    # Create numerical integrator settings.
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    if pick('task') == 'q2':
        dependent_variables_to_save = [propagation_setup.dependent_variable.total_acceleration('Spacecraft')]
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state.flatten(),
            initial_time,
            integrator_settings,
            termination_condition,
            output_variables=dependent_variables_to_save
        )

    if pick('task') == 'q3':
        bodies.get('Spacecraft').mass = pick('vehicle_mass')
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state.flatten(),
            initial_time,
            integrator_settings,
            termination_condition,
        )

    if pick('task') == 'q4':
        bodies.get('Spacecraft').mass = pick('vehicle_mass')
        dependent_variables_to_save = [propagation_setup.dependent_variable.rsw_to_inertial_rotation_matrix('Spacecraft', 'Sun')]
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state.flatten(),
            initial_time,
            integrator_settings,
            termination_condition,
            output_variables=dependent_variables_to_save
        )

    # propagator_settings.print_settings.print_initial_and_final_conditions = True

    
    return propagator_settings


# STUDENT CODE TASK - full function (except signature and return)
# NOTE: Keep this function the same for each question (it does no harm if bodies are
# added that are not used)
def create_simulation_bodies() -> environment.SystemOfBodies:
    """
    Creates the body objects required for the simulation, using the
    environment_setup.create_system_of_bodies for natural bodies,
    and manual definition for vehicles

    Parameters
    ----------
    none

    Return
    ------
    Body objects required for the simulation.

    """

    bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Jupiter", "Saturn", target_body]
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )
    
    # Adding Spacecraft
    body_settings.add_empty_settings("Spacecraft")
    body_settings.get("Spacecraft").constant_mass = pick('vehicle_mass')


    # Sun irradiance definition
    irradiance_at_1AU = 1367.0  # W/m^2, Vallado 2013

    luminosity_model_settings = (
        environment_setup.radiation_pressure.irradiance_based_constant_luminosity(
            irradiance_at_1AU, constants.ASTRONOMICAL_UNIT
        )
    )
    radiation_source_settings_sun = (
        environment_setup.radiation_pressure.isotropic_radiation_source(
            luminosity_model_settings
        )
    )

    body_settings.get("Sun").radiation_source_settings = radiation_source_settings_sun


    body_settings.get('Spacecraft').radiation_pressure_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area = 20.0,
        radiation_pressure_coefficient = 1.2, 
    )


    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies
