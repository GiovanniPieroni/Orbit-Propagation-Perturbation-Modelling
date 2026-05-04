from modules import utils
import yaml

# SETUP folders and files:

if __name__ == "__main__":
    utils.prepare_directories()
try:
    with open("CONFIG.yaml", "r") as yaml_file:
        CONFIG = yaml.safe_load(yaml_file)
except FileNotFoundError:
    CONFIG = {}


# Default parameters and modes:
DEFAULTS = {
    # Parameters:
    "mu_E": 398600.8,
    "R_E": 6371,
    "R_S": 695700,
    "mu_S": 132712440041.93938,
    "mu_M": 4902.800066,
    "J_2": 1.083e-3,

    "initial_date": '2025-12-29',
    "final_date": '2025-12-30',

    # SRP
    "rad_press_ball_coeff": 0.02,  # m^2/kg
    "flux": 1361,  # w/m^2
    "light_speed": 299792458,  # m/s

    # Drag
    "ball_coeff": 0.01,  # m^2/kg
    "ref_altitude": 1500000,  # m
    "scale_height": 516000,  # m
    "ref_rho": 2.79e-16,  # kg/m^3
    "w_E": 7.27220521664304e-05,

    # TBP
    "include_TBP": False,

    "N": 100,
    "ti": 0,
    "rev": 2.5,

    "opt_max_iter": 100,

    # Integration

    # Satellite
    "name": "GONETS-M 24",
    "NORAD": 54151,

    # Different modes:
    "debug": False,
    "TLE": "TLE_data.txt",
    "skip_plots": False
}






