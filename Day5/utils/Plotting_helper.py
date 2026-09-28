import matplotlib.pyplot as plt

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator

color_palette = {'current': '#737373',

                 'PC_cell': '#2ca02c',       # green
                 'GrC_cell': '#d62728',      # red
                 'GoC_cell': '#2171b5',      # blue
                 'MLI_cell': '#ff7f0e',      # orange
                 
                 'stellate_cell': '#ff970e',  # lighter orange
                 'basket_cell': '#ff660e',    # darker orange
            
                 'DCNi_cell': '#e7298a',      # magenta
                 'DCNp_cell': '#6a51a3',      # purple
                 
                 'mossy_cell': '#000000'      # black
                }

#-------------------------------------------------------------------

#-------------------------------------------------------------------
def add_patches(ax, p_start_inh, p_end_inh, p_start_exc, p_end_exc):

    # Adding "inhibition" and "excitation" boxes

    # Ensure ax is iterable
    axes = np.atleast_1d(ax)
    
    for panel in axes:
        panel.add_patch(
            patches.Rectangle(
                (p_start_inh, panel.get_ylim()[0]),
                p_end_inh  - p_start_inh,
                panel.get_ylim()[1] - panel.get_ylim()[0],
                facecolor='red',
                alpha=0.2,
                label='Inhibition' if panel==axes[0] else None
            )
        )
        panel.add_patch(
            patches.Rectangle(
                (p_start_exc, panel.get_ylim()[0]),         # bottom left corner (time, y_min)
                p_end_exc - p_start_exc,                    # width
                panel.get_ylim()[1] - panel.get_ylim()[0],  # height spanning full y axis
                facecolor='green',
                alpha=0.2,
                label='Excitation' if panel==axes[0] else None
            )
        )
    return ax

#-------------------------------------------------------------------
def network_raster_plot(
                        pops,
                        N_pops,
                        pop_names,
                        markersize=None,
                        x_lim=None
                        ):

    # --- Safety checks ---
    assert len(pops) == len(N_pops) == len(pop_names), \
        "pops, N_pops, and pop_names must have same length"

    m_size = 1 if markersize is None else markersize

    fig = plt.figure(figsize=(10, 6))
    offset = 0

    for pop, N, name in zip(pops, N_pops, pop_names):

        if pop is None:
            continue

        # --- Get color ---
        color_key = f"{name}_cell"
        if color_key in color_palette:
            color = color_palette[color_key]
        else:
            print(f"Warning: {color_key} not in color_palette, using default")
            color = 'black'

        # --- Plot ---
        plt.plot(
            pop.t / b2.second,
            pop.i + offset,
            '.',
            color=color,
            label=name,
            markersize=m_size,
            rasterized = True
        )

        offset += N

    # --- Axis settings ---
    if x_lim is not None:
        plt.xlim(x_lim)

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index (with offsets)')
    plt.title('Network Raster Plot')

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(markerscale=5)

    return fig

#-------------------------------------------------------------------
def network_raster_plot_N_neurons_per_type(
                                pops,
                                N_pops,
                                pop_names,
                                markersize=None,
                                x_lim=None,
                                ax=None
                            ):

    # --- Safety checks ---
    assert len(pops) == len(N_pops) == len(pop_names), \
        "pops, N_pops, and pop_names must have same length"

    m_size = 1 if markersize is None else markersize

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        
    offset = 0

    min_N = np.min(N_pops)
   
    for pop, N, name in zip(pops, N_pops, pop_names):

        if pop is None:
            continue

        # --- Get color ---
        color_key = f"{name}_cell"
        if color_key in color_palette:
            color = color_palette[color_key]
        else:
            print(f"Warning: {color_key} not in color_palette, using default")
            color = 'black'

        # Random selection of min_N neurons
        selected_neurons = np.random.choice(N, size=min_N, replace=False)
        
        # Create mapping: original neuron id -> new compact index
        neuron_map = {nid: idx for idx, nid in enumerate(selected_neurons)}
        mask = np.isin(pop.i, selected_neurons)        
        # Apply remapping
        new_i = np.array([neuron_map[i] for i in pop.i[mask]])
        
        ax.plot(
            pop.t[mask] / b2.second,
            new_i + offset,
            '.',
            color=color,
            label=name,
            markersize=m_size,
            rasterized=True
        )

        offset += min_N

    # --- Axis settings ---
    if x_lim is not None:
        ax.set_xlim(x_lim)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron index (with offsets)')
    ax.set_title('Network Raster Plot')

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(markerscale=5)

    return fig
    
#-------------------------------------------------------------------
def plotting_pop_freq_and_std(pops,
                              N_pops,
                              pop_names,
                              sim_duration,
                              bin_size=None,
                              ax=None):

    # --- Safety checks ---
    assert len(pops) == len(N_pops) == len(pop_names), \
        "pops, N_pops, and pop_names must have same length"

    # --- Parameters ---
    if bin_size is None:
        bin_size = 0.1  # seconds

    bin_edges = np.arange(0, sim_duration / b2.second + bin_size, bin_size)
    time_bins = bin_edges[:-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        
    # --- Loop over populations ---
    for pop, N, name in zip(pops, N_pops, pop_names):

        if pop is None:
            continue

        # --- Build spike matrix ---
        spike_matrix = np.zeros((N, len(time_bins)))

        for i, t in zip(pop.i, pop.t / b2.second):
            bin_idx = int(t // bin_size)
            if bin_idx < len(time_bins):
                spike_matrix[i, bin_idx] += 1

        # --- Convert to rate (Hz) ---
        spike_matrix /= bin_size

        # --- Compute stats ---
        mean_rate = np.mean(spike_matrix, axis=0)
        std_rate = np.std(spike_matrix, axis=0)

        # --- Color handling ---
        color_key = f"{name}_cell"
        if color_key in color_palette:
            color = color_palette[color_key]
        else:
            print(f"Warning: {color_key} not in color_palette, using default")
            color = 'black'

        # --- Plot mean ---
        ax.plot(
            time_bins,
            mean_rate,
            label=f'avg {name} freq',
            color=color
        )

        # --- Plot std ---
        ax.fill_between(
            time_bins,
            np.clip(mean_rate - std_rate, 0, None),
            mean_rate + std_rate,
            color=color,
            alpha=0.3,
            label=f'± {name} std'
        )

    # --- Labels ---
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Firing rate (Hz, bin={int(bin_size*1000)} ms)')
    if ax is not None:
        ax.set_title(f'{name} firing rate ± std')
    else:
        ax.set_title(f'Population firing rate ± std')


    ax.legend()
    if ax is not None:
        ax.legend().remove()
        
    plt.tight_layout()

    return fig
