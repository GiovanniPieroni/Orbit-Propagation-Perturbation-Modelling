import argparse
import sys
import os

# Add parent directory to sys.path so we can import from Configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Configuration.config import CONFIG, TASKS_DEFAULTS

parser = argparse.ArgumentParser()

# Global flags
parser.add_argument("--task", type=str) 
parser.add_argument("--only_plotting", type=bool) 
parser.add_argument("--debug", type=bool)
parser.add_argument("--save_cat_state", type=bool)

# Task-specific parameters
parser.add_argument("--target_body", type=str) 
parser.add_argument("--departure_epoch", type=float) 
parser.add_argument("--time_of_flight", type=float) 

CLI, _ = parser.parse_known_args()

def pick(param, override_task=None):
    active_task = override_task or CLI.task or CONFIG.get("task", "interplanetary_transfer_Q1")

    # 1. CLI arguments have priority
    if hasattr(CLI, param) and getattr(CLI, param) not in [None, False]:
        return getattr(CLI, param)
    
    # 2. YAML config file
    if param in CONFIG:
        return CONFIG[param]
    
    # 3. Code defaults in config.py
    if active_task in TASKS_DEFAULTS and param in TASKS_DEFAULTS[active_task]:
        return TASKS_DEFAULTS[active_task][param]
        
    return None
