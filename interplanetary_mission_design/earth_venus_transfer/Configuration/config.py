import yaml
import os

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load YAML CONFIG 
try:
    with open(os.path.join(BASE_DIR, "Configuration", "CONFIG.yaml"), "r") as yaml_file:
        CONFIG = yaml.safe_load(yaml_file)
except FileNotFoundError:
    CONFIG = {}

# Current active task
current_task = CONFIG.get("task", "interplanetary_transfer_Q1")
print(f"\n Active task: {current_task}. If the active task to be runned is not {current_task}, change it accordingly inside Configuration/CONFIG.yaml -> task")

# Defaults for every task in Assignment 2
TASKS_DEFAULTS = {
    "q1": {
        'departure_epoch': 2132.212895 * 86400.0, 
        'time_of_flight': 157.8635921 * 86400.0,
        'vehicle_mass': 0,
        'save_cat_state': True,
    },
    "q2": {
        'departure_epoch': 2132.212895 * 86400.0,
        'time_of_flight': 157.8635921 * 86400.0,
        'vehicle_mass': 1000,
        'save_cat_state': True,
        
    },
    "q3": {
        'target_body': 'Venus',
        'departure_epoch': 2132.212895 * 86400.0,
        'time_of_flight': 157.8635921 * 86400.0,
        'vehicle_mass': 1000,
        'save_cat_state': True,
    },
    "q4": {
        'target_body': 'Venus',
        'vehicle_mass': 1000,
        'save_cat_state': True,
        'departure_epoch': 2132.212895 * 86400.0,
        'time_of_flight': 157.8635921 * 86400.0,
    }
}

# Extracting only the defaults of the current task
DEFAULTS = TASKS_DEFAULTS.get(current_task, TASKS_DEFAULTS["q3"])
