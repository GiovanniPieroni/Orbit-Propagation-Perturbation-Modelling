import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.lines import Line2D  # <--- Make sure this is imported at the top

import os
from pathlib import Path

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 2))

# Constants for consistent plotting styles
LABEL_SIZE = 14
TITLE_SIZE = 16
TICK_SIZE = 12
LEGEND_SIZE = 'large'

current_directory = os.getcwd()
# plotting_directory = Path("Plots/")
# plotting_directory.mkdir(exist_ok=True, parents=True)

#####################################################################################################################################################
# General plotting functions
#####################################################################################################################################################

def plot_normal(x: np.ndarray, y: np.ndarray,
                xlabel: str, ylabel: str,
                title="Generic title", filename="Generic_filename.png", leg_labels=None,
                task="default",
                xlim=None, ylim=None,
                lwidth=1,
                logy=False,
                scatter_list=None):
    """
    Args:
        x: (N,) or (#args, N) array
        y: (N,) or (#args, N) array
        label_list: List of label strings
        filename: Figure filename
    """

    # filename = plotting_directory / f'{filename}'
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    # --- 3. Figure Setup ---
    max_time = np.nanmax(x)
    orbital_lines = []
    # Calculate lines
   

    fig, ax = plt.subplots(figsize=(7, 5))  
    
    if logy:
        ax.set_yscale('log')

    # --- Force numpy arrays ---
    x = np.asarray(x)
    y = np.asarray(y)

    # --- Ensure at least 2D ---
    if x.ndim == 1:
        x = x[:, np.newaxis]  # (N, 1)
    elif x.ndim != 2:
        raise ValueError("x must be 1D or 2D")

    if y.ndim == 1:
        y = y[:, np.newaxis]  # (N, 1)
    elif y.ndim != 2:
        raise ValueError("y must be 1D or 2D")

    n_args = y.shape[1]
    
    # --- Plot ---
    if leg_labels is not None or scatter_list is not None:
         # Increase top margin to give room for labels, legend, and title stack
        fig.subplots_adjust(top=0.80)

    if leg_labels is not None:
        for k in range(n_args):
            plt.plot(x, y[:, k], linewidth=lwidth, label=leg_labels[k])
    else:
        for k in range(n_args):
            plt.plot(x, y[:, k], linewidth=lwidth)
         
    if scatter_list is not None:
        for sx, sy, slabel, scolor, smarker in scatter_list:
            plt.scatter(sx, sy, label=slabel, color=scolor, marker=smarker, s=30, zorder=5)
   

    plt.xlabel(xlabel, fontsize=LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=LABEL_SIZE)

    # Adjust title height based on legend presence
    title_y = 1.20 if (leg_labels is not None or scatter_list is not None) else 1.08
    plt.title(title, fontsize=TITLE_SIZE, y=title_y, fontweight='bold')
    
    if logy:
        ax.set_yscale('log')
        # Use a higher density of major/minor ticks for a finer grid
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.grid(True, which='major', linestyle='-', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    else:
        ax.grid(True, linestyle=':', alpha=0.6)

    ax.tick_params(labelsize=TICK_SIZE)


    # --- ADD ORBITAL PERIOD LABEL ---
    handles, labels = ax.get_legend_handles_labels()
    # Legend ABOVE axis, sitting between title and plot
    if leg_labels is not None or scatter_list is not None:
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.08),
             ncol=max(1, len(labels)), fontsize=LEGEND_SIZE)
        
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(filepath)


def plot_two_scales(time, data1, label1, ylabel1, data2, label2, ylabel2,
                    title="Generic title - insert another", filename="plot_2_axes.png", task="default"):
    """
    Plots 2 quantities with different scales on a single plot using 2 Y-axes.
    """
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    

    fig, ax1 = plt.subplots(figsize=(7, 5))  # Taller figure
    # Adjust top margin significantly to separate Title, Legend, and 1T labels
    fig.subplots_adjust(top=0.90)

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
    ax2.plot(time, data2, color=color2, label=label2, linestyle='--', linewidth=2)
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

    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)


def plot_three_scales(time,
                      data1, label1, ylabel1,
                      data2, label2, ylabel2,
                      data3, label3, ylabel3,
                      title="Orbital Elements",
                      filename="plot_3_axes.pdf",
                      ax1_big=False,
                      task="default"):
    """
    Plots 3 quantities with different scales on a single plot using 3 Y-axes.
    """
    # --- Setup & Filepath ---
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename
    
    # Increase figure width to accommodate the third axis
    fig, ax1 = plt.subplots(figsize=(12, 6))  

    color1, color2, color3 = 'tab:blue', 'tab:green', 'tab:orange'

    # --- Axis 1 (Left) ---
    l1, = ax1.plot(time, data1, color=color1, label=label1, 
                   linestyle='-', linewidth=3 if ax1_big else 2, zorder=3)
    ax1.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
    ax1.set_ylabel(ylabel1, color=color1, fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax1.tick_params(axis='x', labelsize=TICK_SIZE)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # Ensure scientific notation is placed reasonably
    ax1.yaxis.get_offset_text().set_fontsize(TICK_SIZE - 2)

    # --- Axis 2 (Right) ---
    ax2 = ax1.twinx()
    l2, = ax2.plot(time, data2, color=color2, label=label2, 
                   linestyle='--', linewidth=2, zorder=2)
    ax2.set_ylabel(ylabel2, color=color2, fontsize=LABEL_SIZE)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
    ax2.yaxis.get_offset_text().set_fontsize(TICK_SIZE - 2)

    # --- Axis 3 (Offset Right) ---
    ax3 = ax1.twinx()
    # Offset the right spine of ax3
    ax3.spines["right"].set_position(("axes", 1.15))
    l3, = ax3.plot(time, data3, color=color3, label=label3, 
                   linestyle=':', linewidth=2, zorder=1)
    ax3.set_ylabel(ylabel3, color=color3, fontsize=LABEL_SIZE)
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=TICK_SIZE)
    ax3.yaxis.get_offset_text().set_fontsize(TICK_SIZE - 2)

    # Use standard formatter for ax3
    ax3.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax3.yaxis.get_offset_text().set_x(1.15)

    # --- Legend & Title ---
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    
    # Place legend outside above the plot area
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.12),
               ncol=3, frameon=True, fontsize=LEGEND_SIZE)

    # Title pushed higher to avoid legend overlap
    ax1.set_title(title, y=1.25, fontsize=TITLE_SIZE, fontweight='bold')

    # Save with tight_layout or adjusted margins
    try:
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
    except Exception as e:
        print(f"Warning: Could not save {filename} with tight layout: {e}")
        plt.savefig(filepath, dpi=300)
    
    plt.close(fig)
    return str(filepath)

    return str(filename)


def plot_six_subplots(time, state_vector, title="Six subplots", filename='Generic_Filename.pdf',
                      task='default'):
    """
    
    state_vector:  (N, 6) -> [x, y, z, vx, vy, vz]
    """
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    if state_vector.shape[1] != 6:
        raise ValueError("state vector mus be (N, 6)")

    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight='bold')

    labels = [[r'$\Delta$ x [m]', r'$\Delta$ y [m]', r'$\Delta$ z [m]'], 
              [r'$\Delta v_x$ [m/s]', r'$\Delta v_y$ [m/s]', r'$\Delta v_z$ [m/s]']]
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for row in range(2):
        for col in range(3):
            idx = row * 3 + col  
            ax = axs[row, col]
            
            ax.plot(time, state_vector[:, idx], color=colors[col], linewidth=1)
            ax.set_ylabel(labels[row][col], fontsize=LABEL_SIZE)
            ax.grid(True, linestyle=':', alpha=0.7)
            
            if row == 1:
                ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)

    ax.tick_params(labelsize=TICK_SIZE)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(filepath)


def plot_mixed_subplots(time, 
                         data1_top, label1_top, ylabel1_top, 
                         data2_top, label2_top, ylabel2_top,
                         data_bot, label_bot, ylabel_bot, 
                         lwidth_top=6,
                         lwidth_bot=0.5,
                         title="Generic Title", filename="mixed_subplots.pdf", 
                         task="default"):
    """
    Creates a single figure with 2 vertical subplots. 
    TOP: Two Y-axes (twinx) for comparing two scales.
    BOTTOM: Normal single Y-axis plot.
    """
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    # CHANGED: figsize from (8, 10) to (12, 6) for a wider, shorter plot
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6))
    
    # hspace adds room between plots; top leaves room for suptitle/legend
    fig.subplots_adjust(hspace=0.4, top=0.85) # slightly reduced top to fit the new aspect ratio

    color1, color2 = 'tab:blue', 'tab:red'

    # --- TOP SUBPLOT (Two Scales) ---
    ax_top.plot(time, data1_top, color=color1, label=label1_top, linewidth=lwidth_top-lwidth_top/3, zorder=1)
    ax_top.set_ylabel(ylabel1_top, color=color1, fontsize=LABEL_SIZE)
    ax_top.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax_top.grid(True, linestyle=':', alpha=0.6)
    
    ax_top_twin = ax_top.twinx()
    ax_top_twin.plot(time, data2_top, color=color2, label=label2_top, linestyle=':', linewidth=lwidth_top)
    ax_top_twin.set_ylabel(ylabel2_top, color=color2, fontsize=LABEL_SIZE)
    ax_top_twin.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
    
    lines_t1, labels_t1 = ax_top.get_legend_handles_labels()
    lines_t2, labels_t2 = ax_top_twin.get_legend_handles_labels()
    ax_top.legend(lines_t1 + lines_t2, labels_t1 + labels_t2, loc='lower center', 
                  bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=LEGEND_SIZE, frameon=True)

    # --- BOTTOM SUBPLOT (Normal Plot) ---
    ax_bot.plot(time, data_bot, color='blue', label=label_bot, linewidth=lwidth_bot)
    ax_bot.set_xlabel("Time [days]", fontsize=LABEL_SIZE)
    ax_bot.set_ylabel(ylabel_bot, fontsize=LABEL_SIZE)
    ax_bot.tick_params(axis='both', labelsize=TICK_SIZE)
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    
    ax_bot.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), 
                  fontsize=LEGEND_SIZE, frameon=True)

    # Global Title
    fig.suptitle(title, fontsize=TITLE_SIZE, y=1.02, fontweight='bold')

    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return str(filepath)


def plot_three_vertical_subplots(time, 
                                data_top, ylabel_top,
                                data_mid, ylabel_mid,
                                data_bot, ylabel_bot, 
                                lwidth=1.5,
                                title="Generic Title", filename="three_compact_subplots.pdf", 
                                task="default",
                                logy=False):
    """
    Creates a compact figure with 3 vertical subplots sharing the same X-axis.
    No legends, shared x-label at the bottom.
    """
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    
    fig.subplots_adjust(hspace=0.15, top=0.90, bottom=0.1) 

    # --- TOP SUBPLOT ---
    ax_top.plot(time, data_top, color='tab:blue', linewidth=lwidth)
    ax_top.set_ylabel(ylabel_top, fontsize=LABEL_SIZE)
    ax_top.tick_params(axis='both', labelsize=TICK_SIZE)
    if logy:
        ax_top.set_yscale('log')
        ax_top.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax_top.grid(True, which='major', linestyle='-', alpha=0.5)
        ax_top.grid(True, which='minor', linestyle=':', alpha=0.3)
    else:
        ax_top.grid(True, linestyle=':', alpha=0.6)

    # --- MIDDLE SUBPLOT ---
    ax_mid.plot(time, data_mid, color='tab:orange', linewidth=lwidth)
    ax_mid.set_ylabel(ylabel_mid, fontsize=LABEL_SIZE)
    ax_mid.tick_params(axis='both', labelsize=TICK_SIZE)
    if logy:
        ax_mid.set_yscale('log')
        ax_mid.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax_mid.grid(True, which='major', linestyle='-', alpha=0.5)
        ax_mid.grid(True, which='minor', linestyle=':', alpha=0.3)
    else:
        ax_mid.grid(True, linestyle=':', alpha=0.6)

    # --- BOTTOM SUBPLOT ---
    ax_bot.plot(time, data_bot, color='tab:green', linewidth=lwidth)
    ax_bot.set_ylabel(ylabel_bot, fontsize=LABEL_SIZE)
    ax_bot.set_xlabel("Time [days]", fontsize=LABEL_SIZE)
    ax_bot.tick_params(axis='both', labelsize=TICK_SIZE)
    if logy:
        ax_bot.set_yscale('log')
        ax_bot.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax_bot.grid(True, which='major', linestyle='-', alpha=0.5)
        ax_bot.grid(True, which='minor', linestyle=':', alpha=0.3)
    else:
        ax_bot.grid(True, linestyle=':', alpha=0.6)

    # Global Title
    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.98, fontweight='bold')

    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return str(filepath)


def plot_two_vertical_subplots(time, 
                                data_top, ylabel_top,
                                data_bot, ylabel_bot, 
                                lwidth=1.5,
                                title="Generic Title", filename="two_compact_subplots.pdf", 
                                task="default",
                                logy=False):
    """
    Creates a compact figure with 2 vertical subplots sharing the same X-axis.
    No legends, shared x-label at the bottom.
    """
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    fig.subplots_adjust(hspace=0.15, top=0.90, bottom=0.1) 

    # --- TOP SUBPLOT ---
    ax_top.plot(time, data_top, color='tab:blue', linewidth=lwidth)
    ax_top.set_ylabel(ylabel_top, fontsize=LABEL_SIZE)
    ax_top.tick_params(axis='both', labelsize=TICK_SIZE)
    if logy:
        ax_top.set_yscale('log')
        ax_top.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax_top.grid(True, which='major', linestyle='-', alpha=0.5)
        ax_top.grid(True, which='minor', linestyle=':', alpha=0.3)
    else:
        ax_top.grid(True, linestyle=':', alpha=0.6)

    # --- BOTTOM SUBPLOT ---
    ax_bot.plot(time, data_bot, color='tab:orange', linewidth=lwidth)
    ax_bot.set_ylabel(ylabel_bot, fontsize=LABEL_SIZE)
    ax_bot.set_xlabel("Time [days]", fontsize=LABEL_SIZE)
    ax_bot.tick_params(axis='both', labelsize=TICK_SIZE)
    if logy:
        ax_bot.set_yscale('log')
        ax_bot.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax_bot.grid(True, which='major', linestyle='-', alpha=0.5)
        ax_bot.grid(True, which='minor', linestyle=':', alpha=0.3)
    else:
        ax_bot.grid(True, linestyle=':', alpha=0.6)

    # Global Title
    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.98, fontweight='bold')

    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return str(filepath)


def plot_two_subplots_two_scales(time, 
                                 data1_top, label1_top, ylabel1_top, 
                                 data2_top, label2_top, ylabel2_top,
                                 data1_bot, label1_bot, ylabel1_bot, 
                                 data2_bot, label2_bot, ylabel2_bot,
                                 title="Generic Title", filename="two_subplots_two_scales.pdf", 
                                 task="default"):
    """
    Creates a single figure with 2 vertical subplots. 
    Each subplot has two Y-axes (twinx).
    """
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    # Create figure with 2 rows, 1 column
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10))
    fig.subplots_adjust(hspace=0.4, top=0.90) # hspace adds room between plots

    color1, color2 = 'tab:blue', 'tab:red'

    # --- TOP SUBPLOT ---
    # Axis 1 (Left)
    ax_top.plot(time, data1_top, color=color1, label=label1_top)
    ax_top.set_ylabel(ylabel1_top, color=color1, fontsize=LABEL_SIZE)
    ax_top.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax_top.grid(True, linestyle=':', alpha=0.6)
    
    # Axis 2 (Right)
    ax_top_twin = ax_top.twinx()
    ax_top_twin.plot(time, data2_top, color=color2, label=label2_top, linestyle='--')
    ax_top_twin.set_ylabel(ylabel2_top, color=color2, fontsize=LABEL_SIZE)
    ax_top_twin.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
    
    # Legend for Top
    lines_t, labels_t = ax_top.get_legend_handles_labels()
    lines_t2, labels_t2 = ax_top_twin.get_legend_handles_labels()
    ax_top.legend(lines_t + lines_t2, labels_t + labels_t2, loc='lower center', 
                  bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=LEGEND_SIZE)

    # --- BOTTOM SUBPLOT ---
    # Axis 1 (Left)
    ax_bot.plot(time, data1_bot, color=color1, label=label1_bot)
    ax_bot.set_xlabel("Time [days]", fontsize=LABEL_SIZE) # X-label only on bottom
    ax_bot.set_ylabel(ylabel1_bot, color=color1, fontsize=LABEL_SIZE)
    ax_bot.tick_params(axis='y', labelcolor=color1, labelsize=TICK_SIZE)
    ax_bot.tick_params(axis='x', labelsize=TICK_SIZE)
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    
    # Axis 2 (Right)
    ax_bot_twin = ax_bot.twinx()
    ax_bot_twin.plot(time, data2_bot, color=color2, label=label2_bot, linestyle='--')
    ax_bot_twin.set_ylabel(ylabel2_bot, color=color2, fontsize=LABEL_SIZE)
    ax_bot_twin.tick_params(axis='y', labelcolor=color2, labelsize=TICK_SIZE)
    
    # Legend for Bottom
    lines_b, labels_b = ax_bot.get_legend_handles_labels()
    lines_b2, labels_b2 = ax_bot_twin.get_legend_handles_labels()
    ax_bot.legend(lines_b + lines_b2, labels_b + labels_b2, loc='lower center', 
                  bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=LEGEND_SIZE)

    # Global Title
    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.98)

    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)





#####################################################################################################################################################
# Specific plotting functions
#####################################################################################################################################################
def plot_kep_state_separate(time: np.ndarray, y: np.ndarray,
                             spacecraft="Generic spacecraft", 
                             CentralBody="Generic Central Body", 
                             model="Point Mass gravitational model",
                             task="default"
                            ):
    """
    Docstring for plot_kep_state_pointmass
    
    :param time: Description
    :type time: np.ndarray
    :param y: Description
    :type y: np.ndarray
    :param spacecraft: Description
    :param CentralBody: Description
    :param title: Description
    :param filename: Description
    """
    y_plot = y.copy()
    ylabels=['a [m]', 
             'e [-]', 
             'i [rad]', 
             r'$\omega$ [rad]', 
             r'$\Omega$ [rad]', 
             r'$\theta$ [rad]']

    titles=[f"Semi-major Axis of {spacecraft}'s orbit w.r.t {CentralBody}, \n using {model}", 
            f"Eccentricity of {spacecraft}'s orbit w.r.t {CentralBody}, \n using {model}",
            f"Inclination of {spacecraft}'s orbit w.r.t {CentralBody}, \n using {model}",
            f"Argument of Periapsis of {spacecraft}'s orbit w.r.t {CentralBody}, \n using {model}",
            f"RAAN of {spacecraft}'s orbit w.r.t {CentralBody}, \n using {model}",
            f"True Anomaly of {spacecraft}'s orbit w.r.t {CentralBody}, \n using {model}"]

    filenames=[f'sma_{spacecraft}_PM.pdf',
               f'ecc_{spacecraft}_PM.pdf',
               f'inc_{spacecraft}_PM.pdf',
               f'om_{spacecraft}_PM.pdf',
               f'RAAN_{spacecraft}_PM.pdf',
               f'theta_{spacecraft}_PM.pdf']

    y_plot[:, 3] = np.unwrap(y_plot[:, 3])
    y_plot[:, 5] = np.unwrap(y_plot[:, 5])

    for idx in range(0, 6):

        
        plot_normal(time, y_plot[:, idx], 'Time [days]', ylabels[idx], titles[idx], filenames[idx], task=task)


def plot_kep_state_error_separate(time: np.ndarray, y: np.ndarray,
                             spacecraft="Generic spacecraft", 
                             CentralBody="Generic Central Body", 
                             gravity_model="Point Mass gravitational model",
                             filename_number=None,
                             task="default"
                            ):
    """
    Docstring for plot_kep_state_pointmass
    
    :param time: Description
    :type time: np.ndarray
    :param y: Description
    :type y: np.ndarray
    :param spacecraft: Description
    :param CentralBody: Description
    :param title: Description
    :param filename: Description
    """

    ylabels=[r'$\Delta$ a [m]', 
             r'$\Delta$ e [-]', 
             r'$\Delta$ i [rad]', 
             r'$\Delta  \omega$ [rad]', 
             r'$\Delta \Omega$ [rad]', 
             r'$\Delta \theta$  [rad]']

    titles=[f'Semi-major Axis residual, \n'+ r'$a_{\text{numeric}}-a_{\text{analytical}}$, '+f'\n({spacecraft} S/C w.r.t {CentralBody}, {gravity_model})',
            f'Eccentricity residual, \n'+ r'$e_{\text{numeric}}-e_{\text{analytical}}$, '+f'\n({spacecraft} S/C w.r.t {CentralBody}, {gravity_model})',
            f'Inclination residual, \n'+ r'$i_{\text{numeric}}-i_{\text{analytical}}$, '+f'\n({spacecraft} S/C w.r.t {CentralBody}, {gravity_model})',
            f'Argument of Periapsis residual, \n'+ r'$\omega_{\text{numeric}}-\omega_{\text{analytical}}$, '+f'\n({spacecraft} S/C w.r.t {CentralBody}, {gravity_model})',
            f'Right Ascension of the Ascending Node residual, \n'+ r'$\Omega_{\text{numeric}}-\Omega_{\text{analytical}}$, '+f'\n({spacecraft} S/C w.r.t {CentralBody}, {gravity_model})',
            f'True Anomaly residual, \n'+ r'$\theta_{\text{numeric}}-\theta_{\text{analytical}}$, '+f'\n({spacecraft} S/C w.r.t {CentralBody}, {gravity_model})']




    if filename_number is not None:
            filenames=[
                f'sma_err_{filename_number}_{spacecraft}_PM.pdf',
                f'ecc_err_{filename_number}_{spacecraft}_PM.pdf',
                f'inc_err_{filename_number}_{spacecraft}_PM.pdf',
                f'om_err_{filename_number}_{spacecraft}_PM.pdf',
                f'RAAN_err_{filename_number}_{spacecraft}_PM.pdf',
                f'theta_err_{filename_number}_{spacecraft}_PM.pdf']
    else:
            filenames=[
               f'sma_err_{spacecraft}_PM.pdf',
               f'ecc_err_{spacecraft}_PM.pdf',
               f'inc_err_{spacecraft}_PM.pdf',
               f'om_err_{spacecraft}_PM.pdf',
               f'RAAN_err_{spacecraft}_PM.pdf',
               f'theta_err_{spacecraft}_PM.pdf']

    

    for idx in range(0, 6):

        plot_normal(time, y[:, idx], 'Time [days]', ylabels[idx], titles[idx], filenames[idx], task=task)


def plot_keplerian_state(time_days: np.ndarray, keplerian_states: np.ndarray,
                         task: str='Default', filename: str='kep_state.pdf'):

    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename


    fig = plt.figure(figsize=(12,6))

    # semi-major axis
    ax = fig.add_subplot(231)
    ax.plot(time_days, (keplerian_states[:,0])/1.0e3, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel('Semi-major axis [km]', fontsize=LABEL_SIZE)
    ax.grid()

    # eccentricity
    ax = fig.add_subplot(232)
    ax.set_title(fr'Propagated orbital elements of JUICE, w.r.t those at $t_0$', fontsize=TITLE_SIZE)
    ax.plot(time_days, keplerian_states[:,1], linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel('Eccentricity [-]', fontsize=LABEL_SIZE)
    ax.grid()

    # inclination
    ax = fig.add_subplot(233)
    ax.plot(time_days, keplerian_states[:,2]/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel('Inclination [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    # argument of periapsis
    ax = fig.add_subplot(234)
    ax.plot(time_days, (keplerian_states[:,3])/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel('Argument of perigee [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    # right ascension of the ascending node
    ax = fig.add_subplot(235)
    ax.plot(time_days, keplerian_states[:,4]/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel('RAAN [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    # true anomaly
    ax = fig.add_subplot(236)
    ax.plot(time_days, keplerian_states[:,5]/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel('True anomaly [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    fig.tight_layout()

    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)


def plot_keplerian_state_error(time_days: np.ndarray, keplerian_states: np.ndarray,
                         task: str='Default', filename: str='kep_state.pdf', title: str='Generic Title'):

    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename


    fig = plt.figure(figsize=(12,6))

    # semi-major axis
    ax = fig.add_subplot(231)
    ax.plot(time_days, (keplerian_states[:,0])/1.0e3, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\Delta$ Semi-major axis [km]', fontsize=LABEL_SIZE)
    ax.grid()

    # eccentricity
    ax = fig.add_subplot(232)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.plot(time_days, keplerian_states[:,1], linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\Delta$ Eccentricity [-]', fontsize=LABEL_SIZE)
    ax.grid()

    # inclination
    ax = fig.add_subplot(233)
    ax.plot(time_days, keplerian_states[:,2]/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\Delta$ Inclination [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    # argument of periapsis
    ax = fig.add_subplot(234)
    ax.plot(time_days, (keplerian_states[:,3])/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\Delta$ Argument of perigee [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    # right ascension of the ascending node
    ax = fig.add_subplot(235)
    ax.plot(time_days, keplerian_states[:,4]/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\Delta$ RAAN [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    # true anomaly
    ax = fig.add_subplot(236)
    ax.plot(time_days, keplerian_states[:,5]/np.pi*180, linestyle='-')
    ax.set_xlabel('Time [days]', fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\Delta$ True anomaly [deg]', fontsize=LABEL_SIZE)
    ax.grid()

    fig.tight_layout()

    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)


def plot_car_state_error(time: np.ndarray, y: np.ndarray,
                             spacecraft="Generic spacecraft", 
                             CentralBody="Generic Central Body", 
                             gravity_model="Point Mass gravitational model",
                             filename_number=None,
                             task="default"
                            ):
    """
    Docstring for plot_car_state_pointmass
    
    :param time: Description
    :type time: np.ndarray
    :param y: Description
    :type y: np.ndarray
    :param spacecraft: Description
    :param CentralBody: Description
    :param title: Description
    :param filename: Description
    """

    title = r'Cartesian state residual, $(\mathbf{x}_{Gs})_{\text{numeric}}-(\mathbf{x}_{Gs})_{\text{analytical}}$, '+f'\n({spacecraft} S/C w.r.t {CentralBody}, {gravity_model})'

    filename = 'cartesian_residual_PM.pdf'

    plot_six_subplots(time, y, title, filename, task=task)


def plot_acceleration_norm(time: np.ndarray, y: np.ndarray, 
                           spacecraft="Generic spacecraft", 
                           exerting_body='Generic exerting body',
                           CentralBody="Generic Central Body", 
                           gravity_model="Point Mass gravitational model",
                           filename_number=None,
                           task="default"):
    
    
    # plot normal inputs:
    # x: np.ndarray, y: np.ndarray,
    #             xlabel: str, ylabel: str,
    #             title="Generic title", filename="Generic_filename.png", leg_labels=None,
    #             task="default

    acc_norm = np.linalg.norm(y, axis=1)

    if exerting_body=='Io':
        acronym = 'I'
    elif exerting_body=='Sun':
        acronym='S'

    xlabel = 'Time [days]'
    ylabel = fr"$||\mathbf{{a}}_{{{acronym},s}}|| \ [m/s^2]$"
    title = f"{exerting_body}'s gravitational acceleration on {spacecraft} S/C, \n using a {gravity_model}"
    if filename_number is not None:
        filename = f"{exerting_body}_grav_acc_{filename_number}.pdf"
    else:
        filename = f"{exerting_body}_grav_acc.pdf"

    leg_labels = None

    task = task
    plot_normal(time, acc_norm, xlabel, ylabel, title, filename, leg_labels, task)

    return 0


def plot_3d_orbits(trajectories: dict, origin_name="Origin", title="3D Orbits", 
                   filename="3d_orbits.pdf", task="default"):
    """
    Plots multiple 3D trajectories on the same graph, with a central marker at (0,0,0).
    trajectories: dictionary of { 'Label': (N, 3) numpy array }
    """
    task_dir = Path(f"Plots/{task}")
    task_dir.mkdir(exist_ok=True, parents=True)
    filepath = task_dir / filename

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Adjust margins to give labels room - more conservative margins
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    # Zoom out further to keep labels inside
    ax.dist = 12

    # Calculate equal limits for all axes to ensure 1:1:1 data scale
    all_pos = np.vstack(list(trajectories.values()))
    max_range = np.array([all_pos[:,0].max()-all_pos[:,0].min(), 
                         all_pos[:,1].max()-all_pos[:,1].min(), 
                         all_pos[:,2].max()-all_pos[:,2].min()]).max()
    mid_x = (all_pos[:,0].max()+all_pos[:,0].min()) * 0.5
    mid_y = (all_pos[:,1].max()+all_pos[:,1].min()) * 0.5
    mid_z = (all_pos[:,2].max()+all_pos[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)

    # Ensure proportional 3D box aspect ratio
    ax.set_box_aspect([1, 1, 1])

    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']

    # Loop through the dictionary and plot each trajectory
    for idx, (label, pos) in enumerate(trajectories.items()):
        style = '-' if idx % 2 == 0 else '--'
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                label=label, color=colors[idx % len(colors)], linewidth=4.0, linestyle=style)

    # Plot origin body
    ax.scatter([0], [0], [0], color='tab:orange', s=100, label=origin_name, zorder=5)

    # Labeling - increased padding to avoid tick overlap
    ax.set_xlabel('X [m]', labelpad=20, fontsize=TICK_SIZE)
    ax.set_ylabel('Y [m]', labelpad=20, fontsize=TICK_SIZE)
    ax.set_zlabel('Z [m]', labelpad=20, fontsize=TICK_SIZE)
    ax.tick_params(labelsize=10)

    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Title and Legend
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=30)
    ax.legend(loc='upper left', fontsize=LEGEND_SIZE, bbox_to_anchor=(-0.2, 1.0))

    # Save without tight_layout as it often breaks 3D label placement
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(filepath)

