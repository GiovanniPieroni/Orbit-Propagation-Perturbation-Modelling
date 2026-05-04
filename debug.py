import argparse
from config import CONFIG, DEFAULTS

# Argument parser:
parser = argparse.ArgumentParser()

parser.add_argument("--dt", type=float)
parser.add_argument("--N", type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--skip_plots", action="store_true")
parser.add_argument("--NORAD", type=int)
parser.add_argument("--rev", type=int)
parser.add_argument("--rev_GPD", type=int)
parser.add_argument("--rev_GPD_alt", type=int)
parser.add_argument("--rev_euler", type=int)
parser.add_argument("--rev_RK4", type=int)
parser.add_argument("--ti", type=float)
parser.add_argument("--N_euler", type=float)
parser.add_argument("--N_GPD_exp_max", type=int)
parser.add_argument("--N_GPD_exp_min", type=int)
parser.add_argument("--N_GPD_length", type=int)
parser.add_argument("--N_CGPD_exp_max", type=int)
parser.add_argument("--N_CGPD_length", type=int)
parser.add_argument("--N_GPD_exp_min_alt", type=int)
parser.add_argument("--N_GPD_exp_max_alt", type=int)
parser.add_argument("--N_GPD_length_alt", type=int)
parser.add_argument("--N_CGPD_exp_min_alt", type=int)
parser.add_argument("--N_CGPD_exp_max_alt", type=int)
parser.add_argument("--N_CGPD_length_alt", type=int)
parser.add_argument("--N_RK4", type=int)

parser.add_argument("--max_int_time", type=int)
parser.add_argument("--opt_max_iter", type=int)

parser.add_argument("--mu_E", type=float)
parser.add_argument("--mu_S", type=float)
parser.add_argument("--mu_M", type=float)

parser.add_argument("--R_E", type=int)
parser.add_argument("--R_S", type=int)

parser.add_argument("--J_2", type=float)

parser.add_argument("--initial_date", type=float)
parser.add_argument("--final_date", type=float)

# SRP
parser.add_argument("--rad_press_ball_coeff", type=float)
parser.add_argument("--flux", type=float)
parser.add_argument("--light_speed", type=float)

# Drag
parser.add_argument("--ball_coeff", type=float)
parser.add_argument("--ref_altitude", type=float)
parser.add_argument("--scale_height", type=float)
parser.add_argument("--ref_rho", type=float)
parser.add_argument("--w_E", type=float)

# TBP
parser.add_argument("--include_TBP", type=bool)

CLI = parser.parse_args()


# Merging argparse options:
def pick(param):
    if getattr(CLI, param) not in [None, False]:
        return getattr(CLI, param)
    if param in CONFIG:
        return CONFIG[param]
    return DEFAULTS[param]

