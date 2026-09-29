import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import networkx as nx



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize



def plot_bold_signal_4(empBOLD, simBOLD_time, simBOLD_m, bold_ww, bold_wc, bold_go, regions_name, outdir, labels_size=10):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('BOLD signals')

    # Colormap for unique color-coding
    cmap = plt.get_cmap('tab20')
    num_regions = len(regions_name)
    colors = [cmap(i) for i in np.linspace(0, 1, num_regions)]

    # Plot empirical BOLD data
    for i, region_data in enumerate(empBOLD.T):
        axs[0, 0].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[0, 0].set_title('Empirical BOLD', fontsize=labels_size + 2)
    axs[0, 0].set_xlabel('Time [s]', fontsize=labels_size)
    axs[0, 0].set_ylabel('BOLD', fontsize=labels_size)

    # Plot legend in the top-right plot
    axs[0, 1].axis('off')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend = axs[0, 1].legend(handles, regions_name, loc='center', ncol = 3, fontsize=labels_size-4)
    legend.set_title('Regions', prop={'size': labels_size + 2})
    axs[0, 1].add_artist(legend)

    # Plot simulated BOLD data - MEAN FIELD
    for i, region_data in enumerate(simBOLD_m.T):
        axs[1, 0].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[1, 0].set_title('CRBL MF', fontsize=labels_size + 2)
    axs[1, 0].set_xlabel('Time [s]', fontsize=labels_size)
    axs[1, 0].set_ylabel('BOLD', fontsize=labels_size)

    # Plot simulated BOLD data - WW
    for i, region_data in enumerate(bold_ww.T):
        axs[1, 1].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[1, 1].set_title('WW', fontsize=labels_size + 2)
    axs[1, 1].set_xlabel('Time [s]', fontsize=labels_size)
    axs[1, 1].set_ylabel('BOLD', fontsize=labels_size)

    # Plot simulated BOLD data - WC
    for i, region_data in enumerate(bold_wc.T):
        axs[2, 0].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[2, 0].set_title('WC', fontsize=labels_size + 2)
    axs[2, 0].set_xlabel('Time [s]', fontsize=labels_size)
    axs[2, 0].set_ylabel('BOLD', fontsize=labels_size)

    # Plot simulated BOLD data - GO
    for i, region_data in enumerate(bold_go.T):
        axs[2, 1].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[2, 1].set_title('GO', fontsize=labels_size + 2)
    axs[2, 1].set_xlabel('Time [s]', fontsize=labels_size)
    axs[2, 1].set_ylabel('BOLD', fontsize=labels_size)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend and title
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plt.savefig(outdir + 'BOLD_1subject.png')
    #plt.show()

def plot_bold_signal_wb(empBOLD, simBOLD_time, simBOLD_m, bold_ww, regions_name, outdir, labels_size=10):
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BOLD signals')

    # Colormap for unique color-coding
    cmap = plt.get_cmap('tab20')
    num_regions = len(regions_name)
    colors = [cmap(i) for i in np.linspace(0, 1, num_regions)]

    # Plot empirical BOLD data
    for i, region_data in enumerate(empBOLD.T):
        axs[0, 0].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[0, 0].set_title('Empirical BOLD', fontsize=labels_size + 2)
    axs[0, 0].set_xlabel('Time [s]', fontsize=labels_size)
    axs[0, 0].set_ylabel('BOLD', fontsize=labels_size)

    # Plot legend in the top-right plot
    axs[0, 1].axis('off')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend = axs[0, 1].legend(handles, regions_name, loc='center', ncol = 4, fontsize=labels_size)
    legend.set_title('Regions', prop={'size': labels_size + 2})
    axs[0, 1].add_artist(legend)

    # Plot simulated BOLD data - MEAN FIELD
    for i, region_data in enumerate(simBOLD_m.T):
        axs[1, 0].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[1, 0].set_title('Multi-model TVB', fontsize=labels_size + 2)
    axs[1, 0].set_xlabel('Time [s]', fontsize=labels_size)
    axs[1, 0].set_ylabel('BOLD', fontsize=labels_size)

    # Plot simulated BOLD data - WW
    for i, region_data in enumerate(bold_ww.T):
        axs[1, 1].plot(simBOLD_time, region_data - np.mean(region_data), label=regions_name[i], color=colors[i], alpha=0.3)

    axs[1, 1].set_title('Standard TVB', fontsize=labels_size + 2)
    axs[1, 1].set_xlabel('Time [s]', fontsize=labels_size)
    axs[1, 1].set_ylabel('BOLD', fontsize=labels_size)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend and title
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plt.savefig(outdir + 'BOLDWB_1subject.png')
    #plt.show()


def compare_distributions(emp_data, ww_data, mf_data, outdir):
    """
    Compare distributions of empirical and simulated data using KDE plot.

    Parameters:
    - emp_data: Empirical data array
    - ww_data: Data array for the first simulation
    - mf_data: Data array for the second simulation
    - output_file: File path to save the plot

    Returns:
    - None
    """

    # Flatten the data for each dataset
    emp_data_flat = emp_data.flatten()
    ww_data_flat = ww_data.flatten()
    mf_data_flat = mf_data.flatten()

    # Set seaborn style
    sns.set(style="whitegrid")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot KDE for empirical data
    sns.kdeplot(emp_data_flat, color='grey', label='Empirical BOLD', linewidth=2)

    # Plot KDE for first simulation data
    sns.kdeplot(ww_data_flat, color='#6fa8dc', label='WW model-simulated BOLD', linewidth=2)

    # Plot KDE for second simulation data
    sns.kdeplot(mf_data_flat, color='#ede708', label='Cerebellar MF-simulated BOLD', linewidth=2)

    # Add labels and title
    plt.xlabel('Amplitude [-]')
    plt.ylabel('Density')
    plt.title('BOLD amplitude distribution')

    # Show legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(outdir + 'BOLDs_ampl_DDP.png')


def compare_fourier_distributions(emp_data, ww_data, mf_data, outdir):
    """
    Compare distributions of empirical and simulated data Fourier coefficients using KDE plot.

    Parameters:
    - emp_data: Empirical data array
    - ww_data: Data array for the first simulation
    - mf_data: Data array for the second simulation
    - outdir: Directory where the plot will be saved

    Returns:
    - None
    """

    # Compute the Fourier transform of each dataset
    emp_fft = np.fft.fft(emp_data.flatten())
    ww_fft = np.fft.fft(ww_data.flatten())
    mf_fft = np.fft.fft(mf_data.flatten())

    # Normalize the Fourier coefficients
    emp_fft_norm = np.abs(emp_fft) / np.sum(np.abs(emp_fft))
    ww_fft_norm = np.abs(ww_fft) / np.sum(np.abs(ww_fft))
    mf_fft_norm = np.abs(mf_fft) / np.sum(np.abs(mf_fft))

    # Set seaborn style
    sns.set(style="whitegrid")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot KDE for Fourier transform of empirical data
    sns.kdeplot(emp_fft_norm, color='grey', label='Empirical BOLD', linewidth=2)

    # Plot KDE for Fourier transform of first simulation data
    sns.kdeplot(ww_fft_norm, color='#6fa8dc', label='WW model-simulated BOLD', linewidth=2)

    # Plot KDE for Fourier transform of second simulation data
    sns.kdeplot(mf_fft_norm, color='#ede708', label='Cerebellar MF-simulated BOLD', linewidth=2)

    # Add labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Normalized Density')
    plt.title('BOLD frequencies distribution')

    plt.xlim(0.0, 0.01)
    # Show legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(outdir + 'BOLDs_freq_DDP.png')


def plot_mae_with_error_bars_sem(mae_mf_all_subjects, mae_ww_all_subjects, mae_mf_stds, mae_ww_stds, subject_ids, outdir):
    # Convert subject_ids to range for plotting
    subjects = range(len(subject_ids))
    #subject_labs = [1,2,3,4,5,6,7,8]
    subject_labs = list(range(len(mae_mf_all_subjects)))

    # Calculate SEM (standard error of the mean)
    mae_mf_sem = mae_mf_stds / np.sqrt(len(mae_mf_all_subjects))
    mae_ww_sem = mae_ww_stds / np.sqrt(len(mae_ww_all_subjects))

    plt.figure(figsize=(6.51, 3.50))

    # Plot the shaded areas instead of error bars
    plt.fill_between(subjects, mae_mf_all_subjects - mae_mf_sem, mae_mf_all_subjects + mae_mf_sem, color='#ede708', alpha=0.15)  # Cerebellar MF
    plt.fill_between(subjects, mae_ww_all_subjects - mae_ww_sem, mae_ww_all_subjects + mae_ww_sem, color='#6fa8dc', alpha=0.09)  # WW model

    # Plot the means as points or lines
    plt.plot(subjects, mae_mf_all_subjects, 'o-', color='#ede708', label='cerebellar MF')
    plt.plot(subjects, mae_ww_all_subjects, 'o-', color='#6fa8dc', label='WW model')

    # Set plot labels and title
    plt.xlabel('Subjects')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error across subjects')
    plt.xticks(subjects, subject_labs, rotation=0)
    plt.ylim(0.0, 0.6)
    #plt.legend()
    plt.grid(False)

    # Remove the box by hiding the spines (top, right, left, bottom)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Show only the x and y axes
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig(outdir + 'overallMAE_acrosssubject_no_box.png', dpi = 300)
    plt.savefig(outdir + 'overallMAE_acrosssubject_no_box.pdf', dpi = 300)
    plt.close()



def plot_mae_with_error_bars(mae_mf_all_subjects, mae_ww_all_subjects, mae_mf_stds, mae_ww_stds, subject_ids, outdir):
    # Convert subject_ids to range for plotting
    subjects = range(len(subject_ids))

    plt.figure(figsize=(10, 6))

    # Plot MF data
    plt.errorbar(subjects, mae_mf_all_subjects, yerr=mae_mf_stds, fmt='o-', label='cerebellar MF', capsize=5, color='#ede708')
    
    # Plot WW data
    plt.errorbar(subjects, mae_ww_all_subjects, yerr=mae_ww_stds, fmt='o-', label='WW model', capsize=5, color='#6fa8dc')
    
    # Set plot labels and title
    plt.xlabel('Subjects')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error across subjects')
    plt.xticks(subjects, subject_ids, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig(outdir+'overallMAE_acrosssubject.png')

def plot_mae_without_error_bars(mae_mf_all_subjects, mae_ww_all_subjects, subject_ids, outdir):
    # Convert subject_ids to range for plotting
    subjects = range(len(subject_ids))

    plt.figure(figsize=(5.51, 3.50)) #2/3 of a A4

    # Plot MF data
    plt.plot(subjects, mae_mf_all_subjects, 'o-', label='CRBL MF', color='#ede708')
    
    # Plot WW data
    plt.plot(subjects, mae_ww_all_subjects, 'o-', label='WW', color='#6fa8dc')
    
    # Set plot labels and title
    plt.xlabel('Subjects',fontsize = 16)
    plt.ylabel('MAE',fontsize = 16)
    plt.title('Mean Absolute Error across subjects',fontsize = 16)
    plt.xticks(subjects, subject_ids, rotation=45)
    plt.legend(fontsize = 14)
    plt.grid(False)

    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylim(0, 0.60)

    plt.tight_layout()

    # Show the plot
    plt.savefig(outdir+'overallMAE_acrosssubject_NOSD.png', dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_NOSD.eps', dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_NOSD.pdf', dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_NOSD.svg', dpi = 300)


def plot_mae_with_error_bars_4_sem(mae_mf_all_subjects, mae_ww_all_subjects, mae_wc_all_subjects, mae_go_all_subjects, mae_mf_stds, mae_ww_stds, mae_wc_stds, mae_go_stds, subject_ids, outdir):
    # Convert subject_ids to range for plotting
    subjects = range(len(subject_ids))
    subjects_lab = [1,2,3,4,5,6,7,8] 

    # Calculate SEM (standard error of the mean)
    mae_mf_sem = mae_mf_stds / np.sqrt(len(mae_mf_all_subjects))
    mae_ww_sem = mae_ww_stds / np.sqrt(len(mae_ww_all_subjects))
    mae_wc_sem = mae_wc_stds / np.sqrt(len(mae_wc_all_subjects))
    mae_go_sem = mae_go_stds / np.sqrt(len(mae_go_all_subjects))

    # Create the plot
    plt.figure(figsize=(6.51, 3.50))

    # Plot the shaded areas instead of error bars
    plt.fill_between(subjects, mae_mf_all_subjects - mae_mf_sem, mae_mf_all_subjects + mae_mf_sem, color='#ede708', alpha=0.15)  # Cerebellar MF
    plt.fill_between(subjects, mae_ww_all_subjects - mae_ww_sem, mae_ww_all_subjects + mae_ww_sem, color='#6fa8dc', alpha=0.09)  # WW model
    plt.fill_between(subjects, mae_wc_all_subjects - mae_wc_sem, mae_wc_all_subjects + mae_wc_sem, color='#ff5733', alpha=0.06)  # WC model
    plt.fill_between(subjects, mae_go_all_subjects - mae_go_sem, mae_go_all_subjects + mae_go_sem, color='#ff33e6', alpha=0.06)  # GO model

    # Plot the means as points or lines
    plt.plot(subjects, mae_mf_all_subjects, 'o-', color='#ede708', label='cerebellar MF')
    plt.plot(subjects, mae_ww_all_subjects, 'o-', color='#6fa8dc', label='WW model')
    plt.plot(subjects, mae_wc_all_subjects, 'o-', color='#ff5733', label='WC model')
    plt.plot(subjects, mae_go_all_subjects, 'o-', color='#ff33e6', label='GO model')

    # Set plot labels and title
    plt.xlabel('Subjects', fontsize = 12)
    plt.ylabel('MAE', fontsize = 12)
    plt.title('', fontsize = 12)
    plt.xticks(subjects, subjects_lab, rotation=0)
    #plt.legend(fontsize = 16)
    plt.grid(False)

    # Remove the box by hiding the spines (top, right, left, bottom)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Show only the x and y axes
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(outdir + 'MAE_across_subjects_MFWWWCGO.png', dpi = 300)
    plt.savefig(outdir + 'MAE_across_subjects_MFWWWCGO.pdf', dpi = 300)
    plt.close()


   
def plot_mae_with_error_bars_4(mae_mf_all_subjects, mae_ww_all_subjects, mae_wc_all_subjects, mae_go_all_subjects, mae_mf_stds, mae_ww_stds, mae_wc_stds, mae_go_stds, subject_ids, outdir):
    # Convert subject_ids to range for plotting
    subjects = range(len(subject_ids))

    plt.figure(figsize=(5.51, 3.50))

    # Plot MF data
    plt.errorbar(subjects, mae_mf_all_subjects, yerr=mae_mf_stds, fmt='o-', label='cerebellar MF', capsize=10, color='#ede708')
    
    # Plot WW data
    plt.errorbar(subjects, mae_ww_all_subjects, yerr=mae_ww_stds, fmt='o-', label='WW model', capsize=10, color='#6fa8dc')

    # Plot WC data
    plt.errorbar(subjects, mae_wc_all_subjects, yerr=mae_wc_stds, fmt='o-', label='WC model', capsize=10, color='#ff5733')

    # Plot GO data
    plt.errorbar(subjects, mae_go_all_subjects, yerr=mae_go_stds, fmt='o-', label='GO model', capsize=10, color='#ff33e6')
    
    # Set plot labels and title
    plt.xlabel('Subjects', fontsize = 16)
    plt.ylabel('MAE', fontsize = 16)
    plt.title('', fontsize = 16)
    plt.xticks(subjects, subject_ids, rotation=45)
    plt.legend(fontsize = 16)
    plt.grid(False)


    plt.ylim(0,0.60)
    plt.tight_layout()

    # Show the plot
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.png', dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.eps', dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.pdf',dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.svg', dpi = 300)


def plot_mae_without_error_bars_4(mae_mf_all_subjects, mae_ww_all_subjects, mae_wc_all_subjects, mae_go_all_subjects, subject_ids, outdir):
    # Convert subject_ids to range for plotting
    subjects = range(len(subject_ids))

    plt.figure(figsize=(5.51, 3.50))

    # Plot MF data
    plt.plot(subjects, mae_mf_all_subjects, 'o-', label='CRBL MF', color='#ede708')
    
    # Plot WW data
    plt.plot(subjects, mae_ww_all_subjects, 'o-', label='WW', color='#6fa8dc')

    # Plot WC data
    plt.plot(subjects, mae_wc_all_subjects, 'o-', label='WC', color='#ff5733')

    # Plot GO data
    plt.plot(subjects, mae_go_all_subjects, 'o-', label='GO', color='#ff33e6')
    
    # Set plot labels and title
    plt.xlabel('Subjects', fontsize = 16)
    plt.ylabel('MAE',fontsize = 16)
    plt.title('',fontsize = 16)
    plt.xticks(subjects, subject_ids, rotation=45)
    plt.legend(ncol=2, fontsize=12, loc='best')
    plt.grid(False)

    plt.ylim(0,0.60)
    plt.tight_layout()

    # Show the plot
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.png', dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.eps', dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.pdf',dpi = 300)
    plt.savefig(outdir+'overallMAE_acrosssubject_MFWWWCGO.svg', dpi = 300)

def plot_mae_boxplot(mae_mf_all_subjects, mae_ww_all_subjects, subject_ids, outdir):
    # Prepare the data for the boxplot
    data = {
        'Subject': subject_ids * 2,
        'MAE': mae_mf_all_subjects + mae_ww_all_subjects,
        'Model': ['cerebellar MF'] * len(subject_ids) + ['WW model'] * len(subject_ids)
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(9, 6))
    sns.boxplot(x='Model', y='MAE', data=df, palette={'cerebellar MF': '#ede708', 'WW model': '#6fa8dc'})
    sns.swarmplot(x='Model', y='MAE', data=df, color='.25')

    # Set plot labels and title
    plt.xlabel('Model',fontsize = 16)
    plt.ylabel('MAE',fontsize = 16)
    plt.title('Mean Absolute Error distribution', fontsize = 16)
    plt.grid(True)
    plt.tight_layout()

    plt.ylim(0, 0.60)

    # Save the plot
    plt.savefig(outdir + 'boxplot_MAE.png')
    plt.savefig(outdir + 'boxplot_MAE.eps')
    plt.close()


def plot_mae_boxplot_4paper(mae_mf_all_subjects, mae_ww_all_subjects, subject_ids, outdir):
    # Prepare the data for the boxplot
    data = {
        'Subject': subject_ids * 2,
        'MAE': mae_mf_all_subjects + mae_ww_all_subjects,
        'Model': ['CRBL MF'] * len(subject_ids) + ['WW'] * len(subject_ids)
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(2.76, 3.50))
    sns.boxplot(x='Model', y='MAE', data=df, palette={'CRBL MF': '#ede708', 'WW': '#6fa8dc'})
    #sns.swarmplot(x='Model', y='MAE', data=df, color='.25', size = 3)

    # Set plot labels and title
    plt.xlabel('Model',fontsize = 16)
    plt.ylabel('MAE',fontsize = 16)
    plt.title('', fontsize = 16)

    # Remove the grid and the box border (RECOMMENDED FOR PAPERS)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.ylim(0, 0.60)

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(outdir + 'boxplot_MAE.png', dpi = 300)
    plt.savefig(outdir + 'boxplot_MAE.pdf', dpi = 300)
    plt.savefig(outdir + 'boxplot_MAE.eps', dpi = 300)
    plt.savefig(outdir + 'boxplot_MAE.svg', dpi = 300)
    
    plt.close()

  
def plot_mae_boxplot_4(mae_mf_all_subjects, mae_ww_all_subjects, mae_wc_all_subjects, mae_go_all_subjects, subject_ids, outdir):
    # Prepare the data for the boxplot
    data = {
        'Subject': subject_ids * 4,
        'MAE': mae_mf_all_subjects + mae_ww_all_subjects + mae_wc_all_subjects + mae_go_all_subjects,
        'Model': ['CRBL MF'] * len(subject_ids) + ['WW'] * len(subject_ids) + ['WC'] * len(subject_ids) + ['GO'] * len(subject_ids)
    }
    
    df = pd.DataFrame(data)

    plt.figure(figsize=(2.76, 3.50))
    sns.boxplot(x='Model', y='MAE', data=df, palette={'CRBL MF': '#ede708', 'WW': '#6fa8dc', 'WC': '#ff5733', 'GO': '#ff33e6'})
    #sns.swarmplot(x='Model', y='MAE', data=df, color='.25')

    # Set plot labels and title
    plt.xlabel('Model', fontsize = 16)
    plt.ylabel('MAE', fontsize = 16)
    plt.title('', fontsize = 16)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.ylim(0, 0.60)

    plt.tight_layout()

    # Save the plot
    plt.savefig(outdir + 'boxplot_MAE_MFWWWCGO.png', dpi = 300)
    plt.savefig(outdir + 'boxplot_MAE_MFWWWCGO.eps', dpi = 300)
    plt.savefig(outdir + 'boxplot_MAE_MFWWWCGO.pdf', dpi = 300)
    plt.savefig(outdir + 'boxplot_MAE_MFWWWCGO.svg', dpi = 300)
    plt.close()

def plot_mae_bars_across_regions_sem(mae_mf_regions_avg, mae_ww_regions_avg, mae_mf_regions_all_subjects, mae_ww_regions_all_subjects, regions_name, outdir = '/home/bcc/'):
    # Prepare data for plotting
    regions = range(len(mae_mf_regions_avg))
    mae_mf_means = mae_mf_regions_avg
    mae_ww_means = mae_ww_regions_avg

    # Calculate standard deviations for error bars
    mae_mf_stds = np.std(mae_mf_regions_all_subjects, axis=0)
    mae_ww_stds = np.std(mae_ww_regions_all_subjects, axis=0)

    # Calculate SEM (standard error of the mean)
    mae_mf_sem = mae_mf_stds / np.sqrt(len(mae_mf_regions_all_subjects))
    mae_ww_sem = mae_ww_stds / np.sqrt(len(mae_ww_regions_all_subjects))

    # Create the scatter plot with shaded error areas
    plt.figure(figsize=(6.51, 3.50))

    # Plot the shaded areas instead of error bars
    plt.fill_between(regions, mae_mf_means - mae_mf_sem, mae_mf_means + mae_mf_sem, color='#ede708', alpha=0.3)  # Cerebellar MF
    plt.fill_between(regions, mae_ww_means - mae_ww_sem, mae_ww_means + mae_ww_sem, color='#6fa8dc', alpha=0.3)  # WW model

    # Plot the means as points or lines
    plt.plot(regions, mae_mf_means, 'o-', color='#ede708', label='cerebellar MF')
    plt.plot(regions, mae_ww_means, 'o-', color='#6fa8dc', label='WW model')

    # Set plot labels and title
    plt.xlabel('Region')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error across regions')
    plt.xticks(regions, regions_name, rotation=90)
    #plt.legend()
    plt.grid(False)

    # Remove the box by hiding the spines (top, right, left, bottom)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Show only the x and y axes
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig(outdir + 'MAE_acrossregions_with_sem_no_box.png')
    plt.close()


def plot_mae_bars_across_regions(mae_mf_regions_avg, mae_ww_regions_avg, mae_mf_regions_all_subjects, mae_ww_regions_all_subjects, regions_name, outdir = '/home/bcc/'):
    # Prepare data for plotting
    regions = range(len(mae_mf_regions_avg))
    mae_mf_means = mae_mf_regions_avg
    mae_ww_means = mae_ww_regions_avg

    # Calculate standard deviations for error bars
    mae_mf_stds = np.std(mae_mf_regions_all_subjects, axis=0)
    mae_ww_stds = np.std(mae_ww_regions_all_subjects, axis=0)

    # Create the scatter plot with error bars
    plt.figure(figsize=(10, 6))

    plt.errorbar(regions, mae_mf_means, yerr=mae_mf_stds, fmt='o-', color = '#ede708', label='cerebellar MF', capsize=12)
    plt.errorbar(regions, mae_ww_means, yerr=mae_ww_stds, fmt='o-', color = '#6fa8dc', label='WW model', capsize=12)

    plt.xlabel('Region')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error across regions')
    plt.xticks(regions, regions_name, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir+'MAE_acrossregions.png')


def plot_mae_without_bars_across_regions(mae_mf_regions_avg, mae_ww_regions_avg, regions_name, outdir='/home/bcc/'):
    # Prepare data for plotting
    regions = range(len(mae_mf_regions_avg))
    mae_mf_means = mae_mf_regions_avg
    mae_ww_means = mae_ww_regions_avg

    # Create the scatter plot without error bars
    plt.figure(figsize=(9, 6))

    plt.plot(regions, mae_mf_means, 'o-', color='#ede708', label='cerebellar MF')
    plt.plot(regions, mae_ww_means, 'o-', color='#6fa8dc', label='WW model')

    plt.xlabel('Region', fontsize = 16)
    plt.ylabel('MAE',fontsize = 16)
    plt.title('Mean Absolute Error across regions',fontsize = 16)
    plt.xticks(regions, regions_name, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0, 0.60)

    plt.savefig(outdir + 'MAE_acrossregions_NOSD.png')
    plt.savefig(outdir + 'MAE_acrossregions_NOSD.eps')
    plt.close()


def plot_mae_bars_across_regions_4_sem(mae_mf_regions_avg, mae_ww_regions_avg, mae_wc_regions_avg, mae_go_regions_avg, mae_mf_regions_all_subjects, mae_ww_regions_all_subjects, mae_wc_regions_all_subjects, mae_go_regions_all_subjects, regions_name, outdir='/home/bcc/'):
    # Prepare data for plotting
    regions = range(len(mae_mf_regions_avg))
    mae_mf_means = mae_mf_regions_avg
    mae_ww_means = mae_ww_regions_avg
    mae_wc_means = mae_wc_regions_avg
    mae_go_means = mae_go_regions_avg

    # Calculate standard deviations for error bars
    mae_mf_stds = np.std(mae_mf_regions_all_subjects, axis=0)
    mae_ww_stds = np.std(mae_ww_regions_all_subjects, axis=0)
    mae_wc_stds = np.std(mae_wc_regions_all_subjects, axis=0)
    mae_go_stds = np.std(mae_go_regions_all_subjects, axis=0)

    # Calculate SEM (standard error of the mean)
    mae_mf_sem = mae_mf_stds / np.sqrt(len(mae_mf_regions_all_subjects))
    mae_ww_sem = mae_ww_stds / np.sqrt(len(mae_ww_regions_all_subjects))
    mae_wc_sem = mae_wc_stds / np.sqrt(len(mae_wc_regions_all_subjects))
    mae_go_sem = mae_go_stds / np.sqrt(len(mae_go_regions_all_subjects))

    # Create the plot
    plt.figure(figsize=(6.51, 3.0))

    # Plot the shaded areas instead of error bars
    plt.fill_between(regions, mae_mf_means - mae_mf_sem, mae_mf_means + mae_mf_sem, color='#ede708', alpha=0.2)  # Cerebellar MF
    plt.fill_between(regions, mae_ww_means - mae_ww_sem, mae_ww_means + mae_ww_sem, color='#6fa8dc', alpha=0.2)  # WW model
    plt.fill_between(regions, mae_wc_means - mae_wc_sem, mae_wc_means + mae_wc_sem, color='#ff5733', alpha=0.2)  # WC model
    plt.fill_between(regions, mae_go_means - mae_go_sem, mae_go_means + mae_go_sem, color='#ff33e6', alpha=0.2)  # GO model

    # Plot the means as points or lines
    plt.plot(regions, mae_mf_means, 'o-', color='#ede708', label='cerebellar MF')
    plt.plot(regions, mae_ww_means, 'o-', color='#6fa8dc', label='WW model')
    plt.plot(regions, mae_wc_means, 'o-', color='#ff5733', label='WC model')
    plt.plot(regions, mae_go_means, 'o-', color='#ff33e6', label='GO model')

    plt.xlabel('Region', fontsize = 12)
    plt.ylabel('MAE', fontsize = 12)
    plt.xticks(regions, regions_name, rotation=90)
    #plt.legend(ncol=2, fontsize=12, loc='best')
    plt.grid(False)
    
    plt.ylim(0, 1)

    # Remove the box by hiding the spines (top, right, left, bottom)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Show only the x and y axes
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig(outdir + 'MAE_acrossregions_MFWWWCGO.png', dpi = 300)
    plt.savefig(outdir + 'MAE_acrossregions_MFWWWCGO.pdf', dpi = 300)
    plt.close()



def plot_mae_bars_across_regions_4(mae_mf_regions_avg, mae_ww_regions_avg, mae_wc_regions_avg, mae_go_regions_avg, mae_mf_regions_all_subjects, mae_ww_regions_all_subjects, mae_wc_regions_all_subjects, mae_go_regions_all_subjects, regions_name, outdir='/home/bcc/'):
    # Prepare data for plotting
    regions = range(len(mae_mf_regions_avg))
    mae_mf_means = mae_mf_regions_avg
    mae_ww_means = mae_ww_regions_avg
    mae_wc_means = mae_wc_regions_avg
    mae_go_means = mae_go_regions_avg

    # Calculate standard deviations for error bars
    mae_mf_stds = np.std(mae_mf_regions_all_subjects, axis=0)
    mae_ww_stds = np.std(mae_ww_regions_all_subjects, axis=0)
    mae_wc_stds = np.std(mae_wc_regions_all_subjects, axis=0)
    mae_go_stds = np.std(mae_go_regions_all_subjects, axis=0)

    # Create the scatter plot with error bars
    plt.figure(figsize=(6.51, 3.0))

    plt.errorbar(regions, mae_mf_means, yerr=mae_mf_stds/np.sqrt(8), fmt='o-', color='#ede708', label='cerebellar MF', capsize=10)
    plt.errorbar(regions, mae_ww_means, yerr=mae_ww_stds/np.sqrt(8), fmt='o-', color='#6fa8dc', label='WW model', capsize=10)
    plt.errorbar(regions, mae_wc_means, yerr=mae_wc_stds/np.sqrt(8), fmt='o-', color='#ff5733', label='WC model', capsize=10)
    plt.errorbar(regions, mae_go_means, yerr=mae_go_stds/np.sqrt(8), fmt='o-', color='#ff33e6', label='GO model', capsize=10)

    plt.xlabel('Region', fontsize = 16)
    plt.ylabel('MAE', fontsize = 16)
    plt.xticks(regions, regions_name, rotation=45)
    #plt.legend(ncol=2, fontsize=12, loc='best')
    plt.grid(False)
 
    plt.ylim(0,0.60)
    plt.tight_layout()

    plt.savefig(outdir + 'MAE_acrossregions_MFWWWCGO.png',dpi = 300)
    #plt.savefig(outdir + 'MAE_acrossregions_MFWWWCGO.eps')
    plt.close()


def plot_mae_without_bars_across_regions_4(mae_mf_regions_avg, mae_ww_regions_avg, mae_wc_regions_avg, mae_go_regions_avg, regions_name, outdir='/home/bcc/'):
    # Prepare data for plotting
    regions = range(len(mae_mf_regions_avg))
    mae_mf_means = mae_mf_regions_avg
    mae_ww_means = mae_ww_regions_avg
    mae_wc_means = mae_wc_regions_avg
    mae_go_means = mae_go_regions_avg

    # Create the scatter plot without error bars
    plt.figure(figsize=(9, 6))

    plt.plot(regions, mae_mf_means, 'o-', color='#ede708', label='cerebellar MF')
    plt.plot(regions, mae_ww_means, 'o-', color='#6fa8dc', label='WW model')
    plt.plot(regions, mae_wc_means, 'o-', color='#ff5733', label='WC model')
    plt.plot(regions, mae_go_means, 'o-', color='#ff33e6', label='GO model')

    plt.xlabel('Region',fontsize = 16)
    plt.ylabel('MAE',fontsize = 16)
    plt.title('Mean Absolute Error across regions', fontsize = 16)
    plt.xticks(regions, regions_name, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir + 'MAE_acrossregions_MFWWWCGO_NOSD.png')
    plt.savefig(outdir + 'MAE_acrossregions_MFWWWCGO_NOSD.eps')
    plt.close()


def plot_mae_boxplot_across_regions(mae_mf_regions_all_subjects, mae_ww_regions_all_subjects, regions_name, outdir='/home/bcc/'):
    # Prepare the data for the boxplot
    num_subjects = mae_mf_regions_all_subjects.shape[0]
    num_regions = len(regions_name)
    
    # Flatten the data arrays
    mae_mf_flattened = mae_mf_regions_all_subjects.flatten()
    mae_ww_flattened = mae_ww_regions_all_subjects.flatten()

    # Create arrays for models and regions
    models = ['cerebellar MF'] * num_regions * num_subjects + ['WW model'] * num_regions * num_subjects
    regions = np.tile(regions_name, num_subjects * 2)

    # Create a DataFrame for seaborn
    df = pd.DataFrame({
        'Model': models,
        'Region': regions,
        'MAE': np.concatenate([mae_mf_flattened, mae_ww_flattened])
    })

    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Region', y='MAE', hue='Model', data=df, palette={'cerebellar MF': '#ede708', 'WW model': '#6fa8dc'})
    sns.swarmplot(x='Region', y='MAE', hue='Model', data=df, dodge=True, palette='dark:.25', alpha=0.5)

    # Set plot labels and title
    plt.xlabel('Region')
    plt.ylabel('Mean Absolute Error')
    plt.title('Boxplot of MAE across Regions for MF and WW Models')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(outdir + 'boxplot_MAE_acrossregions.png')
    plt.close()


def plot_coherence(coherence_results, regions_name):
    
    freq_coherence_list = coherence_results['freq_coherence']
    coherence_list = coherence_results['coherence']
    
    colormap = plt.cm.get_cmap('tab20')
    num_regions = len(regions_name)
    
    colors = [colormap(i / num_regions) for i in range(num_regions)]
    legend_handles = []
    
    plt.figure(figsize=(9, 4))
    
    for i, region_name in enumerate(regions_name):
        
        plt.semilogy(freq_coherence_list[i], coherence_list[i], color=colors[i], label=region_name)
        legend_handles.append(plt.Line2D([], [], color=colors[i], label=region_name))
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.title('Coherence Between Empirical and Simulated BOLD Signals')
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.show()


def plot_subspec_coherence_mat(avg_coherence_mat, f_coh, fband, sub_ids, regions_name, vmin, vmax, outdir, title_model):
    
    freq_band_start = fband[0]
    freq_band_end = fband[-1]

    for ind, subj in enumerate(sub_ids):
        plt.figure(figsize=(9, 9))
        plt.imshow(avg_coherence_mat[ind], cmap='hot', vmin = vmin, vmax = vmax, origin='lower', interpolation='nearest')
        plt.colorbar(label='Coherence')
        plt.title(f'Coherence Matrix for Subject {subj} ({f_coh[freq_band_start]:.2f}-{f_coh[freq_band_end]:.2f} Hz)', fontsize = 16)
        
        plt.xlabel('Regions', fontsize = 16)
        plt.ylabel('Regions', fontsize = 16)

        plt.xticks(np.arange(len(regions_name)), regions_name, rotation=90, fontsize = 16)
        plt.yticks(np.arange(len(regions_name)), regions_name, fontsize = 16)
        
        plt.tight_layout()
        plt.savefig(outdir + title_model + '_'+ subj+'_CohMat.png')
        plt.close()



def plot_subject_coherence_histograms(sim1_matrices, sim2_matrices, emp_matrices, regions_name_crbl, n_subj_ids, outdir):
    """
    Plots histograms of the average coherence values for each subject in Simulation 1, Simulation 2, and Empirical data.

    Parameters:
        sim1_matrices (numpy array): Coherence matrices for Simulation 1 (shape: nsubjects x nregions x nregions).
        sim2_matrices (numpy array): Coherence matrices for Simulation 2 (shape: nsubjects x nregions x nregions).
        emp_matrices (numpy array): Coherence matrices for Empirical data (shape: nsubjects x nregions x nregions).
        regions_name_crbl (list of str): Names of the regions to use as x-axis labels.
        n_subj_ids (list of str): List of subject IDs to use in plot titles and filenames.
        outdir (str): Directory to save the output figures.
    
    Returns:
        None. Saves figures with histograms for each subject.
    """
    nsubjects, nregions, _ = sim1_matrices.shape

    # Function to compute the average coherence values per region for a given subject
    def compute_avg_coherence(matrix):
        # Since the matrix is symmetric, use the upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(nregions, k=1)
        avg_coherence = np.mean(matrix[triu_indices[0], triu_indices[1]])
        #avg_coherence = np.sum(matrix[triu_indices[0], triu_indices[1]])
        return avg_coherence

    # Loop over each subject
    for subj, subj_id in enumerate(n_subj_ids):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Compute average coherence for the subject
        sim1_avg_coherence = compute_avg_coherence(sim1_matrices[subj])
        sim2_avg_coherence = compute_avg_coherence(sim2_matrices[subj])
        emp_avg_coherence = compute_avg_coherence(emp_matrices[subj])

        # Plot Simulation 1 coherence values for the subject
        axs[0].bar(range(1, nregions + 1), sim1_avg_coherence, color='blue', alpha=0.7)
        axs[0].set_title(f'Subject {subj_id}: Simulation 1 Average Coherence per Region')
        axs[0].set_ylabel('Average Coherence')

        # Plot Simulation 2 coherence values for the subject
        axs[1].bar(range(1, nregions + 1), sim2_avg_coherence, color='green', alpha=0.7)
        axs[1].set_title(f'Subject {subj_id}: Simulation 2 Average Coherence per Region')
        axs[1].set_ylabel('Average Coherence')

        # Plot Empirical coherence values for the subject
        axs[2].bar(range(1, nregions + 1), emp_avg_coherence, color='red', alpha=0.7)
        axs[2].set_title(f'Subject {subj_id}: Empirical Average Coherence per Region')
        axs[2].set_xlabel('Region')
        axs[2].set_ylabel('Average Coherence')

        # Set x-axis ticks with region names
        axs[2].set_xticks(range(nregions))
        axs[2].set_xticklabels(regions_name_crbl, rotation=90)

        # Improve layout and save the plot
        plt.tight_layout()
        plt.savefig(outdir + f'/{subj_id}_coherence.png')
        plt.savefig(outdir + f'/{subj_id}_coherence.eps')
        plt.close()




def plot_psd(psd_matrices, f_psd, outdir, subj_ids, regions_name, title_model):
    """
    Plot PSD for each subject across all regions.

    Parameters:
    - psd_matrices (numpy.ndarray): Array of shape (nsubj, nregion, nfreq) containing PSD values.
    - f_psd (numpy.ndarray): Array of shape (nfreq,) containing frequency values corresponding to PSD.
    - regions_name (list or None): List of region names corresponding to each region index (optional).

    """
    #nsubj, nregion, nfreq = psd_matrices.shape

    for ind, subj in enumerate(subj_ids):
        plt.figure(figsize=(12, 8))
        plt.suptitle('PSD '+ subj)

        for indreg, region_name in enumerate(regions_name):
            # Retrieve PSD for current subject and region
            psd_values = psd_matrices[ind, indreg]

            # Plot PSD
            plt.subplot(5, 6, indreg + 1)  # Adjust subplot layout as needed
            plt.semilogy(f_psd, psd_values, label=region_name)
            plt.title(region_name)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (dB/Hz)')
            plt.ylim([1e-7, 1e-1])  # Adjust y-axis limits if needed
            plt.grid(True)

            if region_name:
                plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
                plt.gca().set_xticks(f_psd[::20])  # Show only every 10th frequency for clarity
                plt.gca().set_xticklabels(f_psd[::20].round(2))  # Set frequency labels
            else:
                plt.xlabel('Frequency (Hz)')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(outdir + title_model+'_'+subj+'_PSD.png')
        plt.close()



def plot_avg_coherence_mat(avg_coherence_mat, f_coh, fband, regions_name, vmax, outdir, title_model):
    """
    Plot the average coherence matrix for a specified frequency band.

    Parameters:
    - coherence_matrices (numpy.ndarray): Array of shape (nsubj, nreg, nreg, nfreq) containing coherence values.
    - f_coh (numpy.ndarray): Array of shape (nfreq,) containing frequency values corresponding to coherence.
    - fband (tuple): Tuple (start_freq, end_freq) specifying the frequency band index.
    - regions_name (list): List of region names corresponding to each region index.
    - vmax (float): Maximum value for color scale.
    - outdir (str): Output directory to save the plot.

    Returns:
    - None
    """
    freq_band_start = fband[0]
    freq_band_end = fband[-1]
    

    avg_coherence_band = avg_coherence_mat.mean(axis=0) #Extra avg on subjecgs
    
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_coherence_band, cmap='hot', vmin=0, vmax=vmax, origin='lower', interpolation='nearest')
    plt.colorbar(label='Coherence')
    plt.title(f'Average Coherence Matrix ({f_coh[freq_band_start]:.2f}-{f_coh[freq_band_end]:.2f} Hz)')
    plt.xlabel('Regions')
    plt.ylabel('Regions')

    plt.xticks(np.arange(len(regions_name)), regions_name, rotation=90)
    plt.yticks(np.arange(len(regions_name)), regions_name)
    
    plt.tight_layout()
    plt.savefig(outdir + title_model +'_AVG_COHERENCE.png')
    plt.close()



def plot_coherence_frequencies(bold_data, subj_ids, region1, region2, fs, regions_name, outdir):
    """
    Plot and save the coherence between two regions for all subjects in one figure with subplots.

    Parameters:
    - bold_data (numpy.ndarray): Array of shape (nsubj, timepoint, nregion) containing BOLD signals.
    - subj_ids (list): List of subject IDs.
    - region1 (int): Index of the first region.
    - region2 (int): Index of the second region.
    - fs (float): Sampling frequency.
    - regions_name (list): List of region names corresponding to each region index.
    - outdir (str): Output directory to save the plot.

    Returns:
    - None
    """

    #bold_data = np.array(bold_data, dtype=np.float64)  # Ensure bold_data is numpy array of floats
    #fs = float(fs)
    nsubj, timepoint, nregion = bold_data.shape
    
    # Determine the number of rows and columns for subplots
    ncols = 3  # You can adjust this based on your preference
    nrows = int(np.ceil(nsubj / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axs = axs.flatten()  # Flatten the array for easier indexing

    for ind, subj_id in enumerate(subj_ids):
        f, Cxy = coherence(bold_data[ind, :, region1], bold_data[ind, :, region2], fs=fs, nperseg=min(256, timepoint))
    
        axs[ind].plot(f, Cxy)
        axs[ind].set_xlabel('Frequency (Hz)')
        axs[ind].set_ylabel('Coherence')
        axs[ind].set_title(f'Subject {subj_id}\n{regions_name[region1]} - {regions_name[region2]}')
    
    # Hide any unused subplots
    for i in range(ind + 1, len(axs)):
        fig.delaxes(axs[i])
    
    plt.suptitle(f'Coherence between {regions_name[region1]} and {regions_name[region2]} for All Subjects', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Coherence_{regions_name[region1]}_{regions_name[region2]}_All_Subjects.png")
    plt.close()



def plot_brain_network(coherence_matrices, frequency_band, freq_values, threshold, subj_ids, regions_name, outdir):
    """
    Plot and save the brain network coherence graph for each subject in one figure with subplots.

    Parameters:
    - coherence_matrices (numpy.ndarray): Array of shape (nsubj, nreg, nreg, nfreq) containing coherence values.
    - frequency_band (float): The frequency band of interest.
    - freq_values (numpy.ndarray): Array of shape (nfreq,) containing frequency values.
    - threshold (float): Threshold for including edges in the graph.
    - subj_ids (list): List of subject IDs.
    - regions_name (list): List of region names corresponding to each region index.
    - outdir (str): Output directory to save the plots.

    Returns:
    - None
    """
    freq_idx = (np.abs(freq_values - frequency_band)).argmin()
    
    nsubj = len(subj_ids)
    ncols = 3  # Number of columns for subplots
    nrows = int(np.ceil(nsubj / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axs = axs.flatten()

    # Initialize the graph and compute positions once
    G = nx.Graph()
    for i in range(len(regions_name)):
        G.add_node(i)
    pos = nx.spring_layout(G)
    
    for ind, subj_id in enumerate(subj_ids):
        fc_matrix = coherence_matrices[ind, :, :, freq_idx]
        
        G.remove_edges_from(list(G.edges()))  # Clear previous edges
        
        # Add edges with coherence above threshold
        for i in range(fc_matrix.shape[0]):
            for j in range(i + 1, fc_matrix.shape[0]):
                if fc_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=fc_matrix[i, j])
        
        nx.draw_networkx_nodes(G, pos, ax=axs[ind], node_size=500, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, ax=axs[ind], width=[d['weight'] * 5 for (u, v, d) in G.edges(data=True)])
        nx.draw_networkx_labels(G, pos, ax=axs[ind], font_size=8, labels={i: regions_name[i] for i in range(len(regions_name))})
        
        axs[ind].set_title(f'Subject {subj_id}')
    
    # Hide any unused subplots
    for i in range(ind + 1, len(axs)):
        fig.delaxes(axs[i])
    
    # Add legend to the figure
    labels = {i: regions_name[i] for i in range(len(regions_name))}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label=f'{i}: {name}') for i, name in labels.items()]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Region Nodes")

    plt.suptitle(f'Brain Network at {frequency_band} Hz for All Subjects', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
    plt.savefig(f"{outdir}/Brain_Network_All_Subjects.png")
    plt.close()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_coherence_matrices_band(empirical, homogeneous, hybrid, regions_name, band_name, out_dir, subject_idx=None):
    """
    Plots the coherence matrices for empirical, homogeneous, and hybrid models.
    If subject_idx is provided, it plots for that specific subject.
    If subject_idx is None, it plots the average coherence across subjects.
    """
    if subject_idx is None:
        # Plot averaged coherence across subjects
        title_suffix = "Avg Across Subjects"
    else:
        # Plot coherence for a specific subject
        title_suffix = f"Subject {subject_idx}"
    
    # Create the figure with 3 subplots (empirical, homogeneous, hybrid)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})
    
    # Plot Empirical Coherence
    sns.heatmap(empirical, ax=axes[0], cmap="hot", vmax=0.7,
                xticklabels=regions_name, yticklabels=regions_name, cbar=True)
    axes[0].set_title(f'Empirical Coherence - {title_suffix} ({band_name})')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_aspect('equal')  # Ensure square aspect ratio

    # Plot Homogeneous Coherence
    sns.heatmap(homogeneous, ax=axes[1], cmap="hot", vmax=0.7,
                xticklabels=regions_name, yticklabels=regions_name, cbar=True)
    axes[1].set_title(f'Homogeneous Coherence - {title_suffix} ({band_name})')
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_aspect('equal')  # Ensure square aspect ratio

    # Plot Hybrid Coherence with color bar
    sns.heatmap(hybrid, ax=axes[2], cmap="hot", vmax=0.7,
                xticklabels=regions_name, yticklabels=regions_name, cbar=True)
    axes[2].set_title(f'Hybrid Coherence - {title_suffix} ({band_name})')
    axes[2].tick_params(axis='x', rotation=90)
    axes[2].set_aspect('equal')  # Ensure square aspect ratio

    # Adjust layout for better visibility
    plt.tight_layout()
    plt.savefig(out_dir + f"/band{band_name}_subj{subject_idx}.png")
    plt.savefig(out_dir + f"/band{band_name}_subj{subject_idx}.eps")
    plt.close(fig) 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def plot_sc_circle(connection, color='tab:green', show_labels=False,
                   label_fontsize=5, figsize=(7, 7)):
    W = np.maximum(connection.weights, connection.weights.T)
    n = connection.number_of_regions
    strength = W.sum(1)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.c_[np.cos(theta), np.sin(theta)]

    i, j = np.triu_indices(n, k=1)
    keep = W[i, j] > 0
    i, j, w = i[keep], j[keep], W[i, j][keep]
    wn = w / w.max()

    fig, ax = plt.subplots(figsize=figsize)
    for k in np.argsort(wn):                      # archi forti sopra
        verts = [pos[i[k]], (0, 0), pos[j[k]]]
        path = Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        ax.add_patch(PathPatch(path, fc='none', ec=color,
                               lw=0.2 + 2 * wn[k], alpha=0.2 + 0.8 * wn[k]))

    ax.scatter(pos[:, 0], pos[:, 1], s=5 + 60 * strength / strength.max(),
               c='0.25', zorder=3)

    if show_labels:
        for k, lbl in enumerate(connection.region_labels):
            ang = np.degrees(theta[k])
            flip = 90 < ang < 270
            ax.text(*(1.05 * pos[k]), lbl, fontsize=label_fontsize,
                    rotation=ang + 180 if flip else ang,
                    ha='right' if flip else 'left', va='center',
                    rotation_mode='anchor')

    ax.set_aspect('equal'); ax.axis('off')
    lim = 1.4 if show_labels else 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    plt.tight_layout(); plt.show()
    return fig, ax

