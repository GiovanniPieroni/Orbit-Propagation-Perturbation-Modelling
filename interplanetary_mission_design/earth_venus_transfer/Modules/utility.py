from pathlib import Path
from datetime import datetime
import os

# Base directory relative to this file
BASE_DIR = Path(__file__).parent.parent

def prepare_directories(task="default"):
    """Creates Plots and Data folders for the specific task."""
    (BASE_DIR / "Plots" / task).mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "Data" / task).mkdir(parents=True, exist_ok=True)

def plot_filename(base_name, task="default", extension="pdf"):
    """Creates a timestamped filename for plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(BASE_DIR / "Plots" / task / f"{base_name}_{timestamp}.{extension}")

def save_to_row(filename, data, row_number):
    """Saves an array to a specific row of a file (1-indexed)."""
    file_path = BASE_DIR / filename

    # Read existing content if file exists
    lines = []
    if file_path.exists():
        with open(file_path, "r") as f:
            lines = f.readlines()

    # Ensure we have enough lines
    while len(lines) < row_number:
        lines.append("\n")

    # Format data as space-separated string
    data_str = " ".join(map(str, data)) + "\n"

    # Replace the specific row (row_number is 1-indexed)
    lines[row_number - 1] = data_str

    # Write back to file
    with open(file_path, "w") as f:
        f.writelines(lines)

