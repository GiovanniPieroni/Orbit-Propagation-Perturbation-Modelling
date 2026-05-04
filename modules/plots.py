import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import utils


formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 2))

# Constants for consistent plotting styles
LABEL_SIZE = 16
TITLE_SIZE = 18
TICK_SIZE = 14
LEGEND_SIZE = 'medium'

with open("TLE_GONETSM24.txt", "r") as f:
    TLE_data = f.read().splitlines()

sat = utils.create_satrec(TLE_data)

mean_motion = sat.no_kozai / 60  # [rad/s]
period = 2 * np.pi / mean_motion  # Orbital period


# ----------------------------------------------------- A3 ----------------------------------------------------------- #

def mask_discontinuity(data, threshold=179):
    """
    Replaces values with NaN where the jump between points exceeds threshold.
    Useful for removing vertical lines in -180/180 wrapping angles.
    """
    # Create a copy so we don't modify the original data array
    clean_data = data.copy()

    # Calculate difference between consecutive points
    # prepend=data[0] keeps the shape consistent
    diff = np.abs(np.diff(clean_data, prepend=clean_data[0]))

    # Find indices where the jump is too big (wrapping)
    # Threshold 300 deg is safe for standard orbits
    jump_indices = diff > threshold

    # Set those points to NaN to break the line
    clean_data[jump_indices] = np.nan
    return clean_data





def plot_three_scales(time,
                      data1, label1, ylabel1,
                      data2, label2, ylabel2,
                      data3, label3, ylabel3,
                      title="Orbital Elements",
                      filename="plot_3_axes.png"):
    # --- Setup & Figure Size ---
    max_time = np.nanmax(np.array(time)) if len(time) > 0 else 0
    orbital_lines = []

    # Calculate lines
    if period is not None and max_time > period:
        orbital_lines = np.arange(period, max_time+1, period, dtype=float)
        fig, ax1 = plt.subplots(figsize=(10 + len(orbital_lines), 9))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 7))

    fig.subplots_adjust(right=0.75, top=0.85)

    try:
        # --- Masking Logic ---
        if "Latitude" in title or "Argument" in title:
            data1 = mask_discontinuity(data1)
            data2 = mask_discontinuity(data2)

        color1, color2, color3 = 'tab:blue', 'tab:green', 'tab:orange'

        # --- Axis 1 (Left) ---
        if "Latitude" in title or "Argument" in title:
            ax1.plot(time, data1, color=color1, label=label1, linestyle='-', linewidth=6, zorder=1)
        else:
            ax1.plot(time, data1, color=color1, label=label1, linestyle='-', linewidth=2, zorder=0)

        ax1.set_xlabel("Time [s]", fontsize=LABEL_SIZE)

        # FIX 1: Massive padding to make room for the offset text on the left
        ax1.set_ylabel(ylabel1, color=color1, fontsize=LABEL_SIZE)

        # Ticks
        ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax1.tick_params(axis='both', which='major', length=7, width=1.2, labelsize=TICK_SIZE)
        ax1.tick_params(axis='both', which='minor', length=4, width=0.8)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, which='major', linestyle=':', alpha=0.6)

        # FIX 2: Move Scientific Notation (Offset Text) FAR LEFT
        # (-0.25 means 25% of the plot width to the left of the y-axis)
        ax1.yaxis.get_offset_text().set_horizontalalignment('right')
        ax1.yaxis.get_offset_text().set_position((-0.25, 1.0))

        # --- Vertical Lines (Orbital Period) ---
        if period is not None and max_time > period:
            for line_x in orbital_lines:
                ax1.axvline(x=line_x, color='black', linestyle='-.', alpha=0.4, zorder=0)

                # FIX 3: Place text using Axis Coordinates (y=1.01) so it sits just above the frame
                # This separates it vertically from the scientific notation if they are close
                label_text = rf"{int(line_x / period)} T $\approx$ {int(period * (line_x / period))} [s]"
                ax1.text(line_x, 1.04, label_text,
                         transform=ax1.get_xaxis_transform(),  # <--- Key fix: uses axis y-coords (0 to 1)
                         ha='center', va='bottom', fontsize=12, color='gray')

        # --- Axis 2 (Right) ---
        if "Latitude" in title or "Argument" in title:
            ax2 = ax1.twinx()
            ax2.plot(time, data2, color=color2, label=label2, linestyle='--', linewidth=4, zorder=0)
            ax2.set_ylabel(ylabel2, color=color2, fontsize=LABEL_SIZE)
            ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
            ax2.yaxis.get_offset_text().set_x(1.18)
        else:
            ax2 = ax1.twinx()
            ax2.plot(time, data2, color=color2, label=label2, linestyle='-', linewidth=6, zorder=0)
            ax2.set_ylabel(ylabel2, color=color2, fontsize=LABEL_SIZE)
            ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
            ax2.yaxis.get_offset_text().set_x(1.1)

        # --- Axis 3 (Right Offset) ---
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.15))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        ax3.plot(time, data3, color=color3, label=label3, linestyle='--', linewidth=4)
        ax3.set_ylabel(ylabel3, color=color3, fontsize=LABEL_SIZE)
        ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax3.tick_params(axis='y', labelcolor=color3, labelsize=TICK_SIZE)

        # Formatter
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 2))
        ax3.yaxis.set_major_formatter(formatter)
        ax3.yaxis.get_offset_text().set_x(1.18)

        # --- Legend & Title ---
        lines = [ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0]]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.08),
                   ncol=3, borderaxespad=0., frameon=True, fontsize=LEGEND_SIZE)
        ax1.set_title(title, y=1.15, fontsize=TITLE_SIZE)

        # --- Save ---
        try:
            fig.savefig(filename, bbox_inches="tight")
        except ValueError:
            print(f"Warning: Infinite data detected in {filename}. Saving without 'tight' layout.")
            fig.savefig(filename)

    finally:
        plt.close(fig)

    return str(filename)



def plot_two_scales(time, data1, label1, ylabel1, data2, label2, ylabel2,
                    title="Generic title - insert another", filename="plot_2_axes.png"):
    """
    Plots 2 quantities with different scales on a single plot using 2 Y-axes.
    """

    if "Latitude" in title or "Argument" in title:
        data2 = mask_discontinuity(data2)

    fig, ax1 = plt.subplots(figsize=(10, 8))  # Taller figure
    # Adjust top margin significantly to separate Title, Legend, and 1T labels
    fig.subplots_adjust(top=0.80)

    color1, color2 = 'tab:blue', 'tab:red'

    # --- Axis 1 ---
    ax1.plot(time, data1, color=color1, label=label1, linestyle='-')
    ax1.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax1.set_ylabel(ylabel1, color=color1, fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Axis 2 ---
    ax2 = ax1.twinx()
    ax2.plot(time, data2, color=color2, label=label2, linestyle='-')
    ax2.set_ylabel(ylabel2, color=color2, fontsize=LABEL_SIZE)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)

    # --- Legend & Title ---
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    labels = [l.get_label() for l in lines]

    # --- ADD ORBITAL PERIOD LABEL ---
    # lines.append(Line2D([0], [0], color='none', label='1T = one orbital period'))
    # labels.append('1T = one orbital period')

    # Place legend ABOVE the plot area but BELOW the title
    # y=1.08 puts it clear of the x-axis top frame where "1T" labels sit
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.08),
               ncol=2, borderaxespad=0., frameon=True, fontsize=LEGEND_SIZE)

    # Push title way up
    ax1.set_title(title, y=1.20, fontsize=TITLE_SIZE)

    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    return str(filename)





