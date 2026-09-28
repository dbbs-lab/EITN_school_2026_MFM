import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import scipy.special as sp_spec
import matplotlib as mpl
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd


def plot_heatmap_vars(data=None,
                       row_values=None,
                       col_values=None,
                       figsize=None,
                       cmap=None,
                       cbarlabel=None,
                       plt_title=None,
                       x_lab=None,
                       y_lab=None,
                       annotate=False):
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10, 7))

    if cmap is None:
        cmap = 'viridis'
    
    ax = sns.heatmap(data,
                     cmap=cmap,
                     #linewidths=0.05,
                     #linecolor='dimgrey',
                     annot=annotate,
                     #fmt=".1f",
                     cbar_kws={'label': cbarlabel})

    # Add a visible frame around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('k')       # black frame
        spine.set_linewidth(1)     # thickness of the frame

    # Add frame around the colorbar 
    cbar = ax.collections[0].colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_color('k')
        spine.set_linewidth(1)    
    
    # Compute centered tick positions
    xticks = np.arange(len(col_values))[::3] + 0.5
    yticks = np.arange(len(row_values))[::3] + 0.5

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.array(col_values)[::3])
    ax.set_yticklabels(np.array(row_values)[::3])

    # Titles and labels
    ax.set_title(plt_title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    # Flip y-axis so origin is bottom-left
    ax.invert_yaxis()
    return fig


## ===========================================
## ===== Membrane fluctuation properties =====
## ===========================================
# based on shotnoise theory, computation of the three statistical moment_
# muV, sigV, Tv (Ref.: Zerlaut et al. 2018)
def get_membrane_fluct_eglif(fe_grid = None,
                             fi_grid = None,
                             adapt = None,
                             params_SI = None):

    Cm = params_SI['C_m'] # membrane capacitance
    Gl = params_SI['g_L'] # leak condunctance
    El = params_SI['E_L'] # leak equilibrium potential

    # excitatory synapses
    Qe = params_SI['Q_e']
    Te = params_SI['T_e']
    Ee = params_SI['E_e']
    Ke = params_SI['K_e']
    # inhibitory synapses    
    Qi = params_SI['Q_i']
    Ti = params_SI['T_i']
    Ei = params_SI['E_i']
    Ki = params_SI['K_i']
    
    fi = fi_grid
    fe = fe_grid

    # ----- Mean Exc and Inh conductances that the neuron "receives" ----- #
    muGe, muGi = Qe * Ke * Te * fe, Qi * Ki * Ti * fi  
    # ----- Input conductance of the given neuron ----- #
    muG = Gl + muGe + muGi  

   
    # ----- Membrane Fluctuation Properties ----- # 

    # Mean Membrane Potential 
    muV = (np.e * (muGe * Ee + muGi * Ei + Gl * El) - adapt) / muG  
    # Membrane time constant
    Tm = Cm / muG
    muGn = muG / Gl  # normalized

    Ue, Ui = Qe / muG * (Ee - muV), Qi / muG * (Ei - muV) 
    
    # Standard deviation of the fluctuation
    sigVe = (2 * Tm + Te) * ((np.e * Ue * Te) / (2 * (Te + Tm))) ** 2 * Ke * fe
    sigVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2 * (Ti + Tm))) ** 2 * Ki * fi

    sigV = np.sqrt(sigVe + sigVi)

    #'''
    # Handle zero elements in sigV by replacing them with the second smallest value
    sigV_flat = sigV.flatten()
    # Adding this to avoid instability due to elements of sigV = 0.0
    unique_sorted = np.unique(sigV_flat)
    if unique_sorted.size > 1 and unique_sorted[0] == 0.0:
        second_smallest = unique_sorted[1]
        n_zeros = np.sum(sigV == 0.0)
        print(f"Warning: {n_zeros} element(s) in sigV were 0.0 and have been replaced "
              f"with the second smallest value ({second_smallest:.3e}).")
        sigV = np.where(sigV == 0.0, second_smallest, sigV)
    elif unique_sorted[0] == 0.0:
        print("Warning: All elements of sigV are 0.0 — cannot replace with a second smallest value.")
    #'''
    
    fe, fi = fe + 1e-9, fi + 1e-9  # just to insure a non zero division

    # Autocorrelation
    Tv_num = Ke * fe * Ue ** 2 * Te ** 2 * np.e ** 2 + Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
    Tv = 0.5 * Tv_num / ((sigV + 1e-20) ** 2)

    '''
    # Detect extreme outliers: elements much smaller than typical scale
    # Find indices of elements that are >= 2 orders smaller than median

    Tv_flat = Tv.flatten()
    median_val = np.median(Tv_flat)
    bad_idx = np.where(Tv_flat < median_val / 100)[0]

    if bad_idx.size > 0:
        for idx in bad_idx:
            if idx < len(Tv_flat) - 1:
                replacement_val = Tv_flat[idx + Tv.shape[0]]
                print(f"Warning: Tv_flat[{idx}] = {Tv_flat[idx]:.3e} "
                      f"is ≥2 orders smaller than median ({median_val:.3e}); "
                      f"replaced with Tv_flat[{idx + Tv.shape[0]}] = {replacement_val:.3e}.")
                Tv_flat[idx] = replacement_val
            else:
                print(f"Warning: Tv_flat[{idx}] is last element and small "
                      f"({Tv_flat[idx]:.3e}); cannot replace with successor.")

    # reshape back to original shape
    Tv = Tv_flat.reshape(Tv.shape)
    #'''
    
    TvN = Tv * Gl / Cm  # normalized

    return muGe, muGi, muG, muV, sigV + 1e-20, muGn, TvN



import seaborn as sns
def plot_MF_statistics(row_values = None,
                       col_values = None,
                       muG = None,
                       muV = None,
                       sigV = None,
                       Tv = None,
                       mask = None,
                       figs = None):
    if figs == None:
        figs = []
    
    fig = plt.figure(figsize=(12,5))
    ax = sns.heatmap(muG, 
                     cmap='viridis',
                     xticklabels=col_values,
                     yticklabels=row_values,
                     linewidths = 0.2,
                     cbar_kws={'label': '[S]'})
    ax.set_title('Target neuron conductance')
    ax.set_xlabel('Fe [Hz]')
    ax.set_ylabel('Fi [Hz]')   
    # Flip the y-axis so the origin is bottom-left
    ax.invert_yaxis()
    figs.append(fig)
    
    fig = plt.figure(figsize=(12,5))
    ax = sns.heatmap(muV, 
                     cmap='viridis',
                     xticklabels=col_values,
                     yticklabels=row_values,
                     linewidths = 0.2,
                     cbar_kws={'label': '[V]'})
    ax.set_title('Average Membrane Potential')
    ax.set_xlabel('Fe [Hz]')
    ax.set_ylabel('Fi [Hz]')   
    # Flip the y-axis so the origin is bottom-left
    ax.invert_yaxis()
    figs.append(fig)

    fig = plt.figure(figsize=(12,5))
    ax = sns.heatmap(sigV, 
                     cmap='viridis',
                     xticklabels=col_values,
                     yticklabels=row_values,
                     linewidths = 0.2,
                     cbar_kws={'label': '[V]'})
    ax.set_title('Membrane Potential Standard Deviation')
    ax.set_xlabel('Fe [Hz]')
    ax.set_ylabel('Fi [Hz]')   
    # Flip the y-axis so the origin is bottom-left
    ax.invert_yaxis()
    figs.append(fig)

    fig = plt.figure(figsize=(12,5))
    ax = sns.heatmap(Tv, 
                     mask = mask,
                     cmap='viridis',
                     xticklabels=col_values,
                     yticklabels=row_values,
                     linewidths = 0.2,
                     cbar_kws={'label': '[s]'})
    ax.set_title('Autocorrelation Time')
    ax.set_xlabel('Fe [Hz]')
    ax.set_ylabel('Fi [Hz]')   
    # Flip the y-axis so the origin is bottom-left
    ax.invert_yaxis()
    figs.append(fig)
    
    return figs

def fitting_Vthre_then_Freq_data_eglif(muGe = None,
                                       muGi = None,
                                       muG = None,
                                       muV = None,
                                       sigV = None,
                                       muGn = None,
                                       TvN = None, 
                                       Freq_data = None,
                                       fe_grid = None, 
                                       fi_grid  = None,
                                       adapt = None,
                                       params_SI = None, 
                                       alpha = None,
                                       maxiter=50000, xtol=1e-5):
    
    Gl, Cm, El = params_SI['g_L'], params_SI['C_m'], params_SI['E_L']

    freq_data_lim= Freq_data.max()

    i_non_zeros = np.where((Freq_data > 0.) & (Freq_data < freq_data_lim))
    print("Freq_data Limit: ", freq_data_lim)
   
    Vthre_eff = effective_Vthre(Y = Freq_data[i_non_zeros],
                                muV = muV[i_non_zeros], 
                                sigV = sigV[i_non_zeros],
                                TvN = TvN[i_non_zeros],
                                Gl = params_SI['g_L'],
                                Cm = params_SI['C_m'],
                                alpha = alpha)

    P = [-50e-3, 0, 0, 0, 0]
    
    print("========== OPTIMIZATION STEP 1 ========== \nVthre Optimization")

    def Res(poly_coeff):
        Vthre = threshold_func(*poly_coeff,
                               muV = muV[i_non_zeros],
                               sigV = sigV[i_non_zeros],
                               TvN = TvN[i_non_zeros],
                               muGn = muGn[i_non_zeros],
                               )
        
        return np.mean((Vthre_eff - Vthre) ** 2)

    plsq = minimize(Res, P, options={'disp': True})
    P = plsq.x

    print('========== P output from Vthreshold opt: ', P)

    print("========== OPTIMIZATION STEP 2 ========== \nFreq_data Optimization")

    def Res(poly_coeff):
        return np.mean((Freq_data -
                        TF_template_eglif(*poly_coeff,
                                           fe = fe_grid,
                                           fi = fi_grid,
                                           adapt = adapt, 
                                           alpha = alpha,
                                           params_SI = params_SI)) ** 2)

    plsq = minimize(Res, P, method='nelder-mead',
                    options={'xatol': xtol, 'disp': True, 'maxiter': maxiter})

    P = plsq.x
    print('========== P output from Freq_data opt: ', P)
    print("========== END OF OPTIMIZATION PROCESS ==========")

    params_SI['P'] = P

    return P


def effective_Vthre(Y = None,
                    muV = None,
                    sigV = None,
                    TvN = None,
                    Gl = None,
                    Cm = None,
                    alpha= None):

    arg = (1 / alpha) * (Y * 2 * TvN * Cm / Gl)
    print(f'min: {np.min(arg)}, max:{np.max(arg)}')
    print(np.sum((arg <= 0) | (arg >= 2)), "invalid argument values")

    Vthre_eff = muV + np.sqrt(2) * sigV * sp_spec.erfcinv((1 / alpha) * (Y * 2 * TvN * Cm / Gl))  # effective threshold

    return Vthre_eff

# Polynomial expression of V eff thre - initial condition
def threshold_func(P0, P1, P2, P3, P4,
                   muV = None,
                   sigV = None,
                   TvN = None,
                   muGn = None):

    # normalization factors as in Zerlaut et al., (2018)
    muV0, DmuV0 = -60e-3, 10e-3
    sigV0, DsigV0 = 4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.

    Vthre =  P0 + \
             P1 * (muV - muV0) / DmuV0 + \
             P2 * (sigV - sigV0) / DsigV0 + \
             P3 * (TvN - TvN0) / DTvN0 + \
             P4 * np.log(muGn)
    return Vthre

def TF_template_eglif(P0, P1, P2, P3, P4,
                      fe = None,
                      fi = None,
                      adapt = None, 
                      alpha = None,
                      params_SI = None):
    
    # here TOTAL (sum over synapses) excitatory and inhibitory input

    if (hasattr(fe, "__len__")):
        fe[fe < 1e-8] = 1e-8
    else:
        if (fe < 1e-8):
            fe = 1e-8
    if (hasattr(fi, "__len__")):
        fi[fi < 1e-8] = 1e-8
    else:
        if (fi < 1e-8):
            fi = 1e-8

    muGe, muGi, muG, muV, sigV, muGn, TvN = get_membrane_fluct_eglif(fe_grid = fe,
                                                                     fi_grid=fi,
                                                                     adapt = adapt,
                                                                     params_SI = params_SI)

    Vthre = threshold_func(muV = muV,
                           sigV = sigV,
                           TvN = TvN,
                           muGn = muGn,
                           P0 = P0,
                           P1 = P1,
                           P2 = P2,
                           P3 = P3,
                           P4 = P4)

    if (hasattr(muV, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        sigV[sigV < 1e-4] = 1e-4
    else:
        if (sigV < 1e-4):
            sigV = 1e-4

    Fout_th = erfc_func(muV = muV,
                        sigV = sigV,
                        TvN = TvN,
                        Vthre = Vthre,
                        Gl = params_SI['g_L'],
                        Cm = params_SI['C_m'],
                        alpha = alpha)

    if (hasattr(Fout_th, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        Fout_th[Fout_th < 1e-8] = 1e-8
    else:
        if (Fout_th < 1e-8):
            Fout_th = 1e-8

    return Fout_th

# EQUATION OF ERFC = vout = Fout = TF expression (Zerlaut et al. 2018)
def erfc_func(muV = None,
              sigV = None,
              TvN = None,
              Vthre = None,
              Gl = None,
              Cm= None, 
              alpha = None):
    
    return .5 / TvN * Gl / Cm * (sp_spec.erfc((Vthre - muV) / np.sqrt(2) / sigV)) * alpha





def build_bar_legend(X, n_tick, ax, mymap, label='$\\nu$ (Hz)',
                     bounds=None, ticks_labels=None,
                     orientation='vertical', scale='linear',
                     color_discretization=None):
   
    if color_discretization is None:
        color_discretization = len(X)
        
    # Determine scale and bounds
    if scale == 'linear':
        if bounds is None:
            bounds = [X[0], X[-1]]
        norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1])
        bounds = np.linspace(bounds[0], bounds[1], color_discretization)
        
    elif scale == 'log10':
        if bounds is None:
            bounds = [np.log10(X[0]), np.log10(X[-1])]
        norm = mpl.colors.LogNorm(vmin=10**bounds[0], vmax=10**bounds[1])
        bounds = np.logspace(bounds[0], bounds[1], color_discretization)
        
    elif scale == 'custom':
        bounds = np.linspace(X[0], X[-1], color_discretization)
        norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[-1])

    # Add an extra boundary edge so the last color appears
    boundaries = np.append(bounds, bounds[-1] + (bounds[-1] - bounds[-2]))

    # Build discrete colorbar
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=mymap, norm=norm,
        boundaries=boundaries,
        orientation=orientation,
        spacing='proportional'
    )

    # Set ticks
    if ticks_labels is None:
        ticks = np.linspace(X[0], X[-1], n_tick)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{tick:.1f}" for tick in ticks])
    else:
        cb.set_ticks(X)
        cb.set_ticklabels(ticks_labels)

    cb.set_label(label)
    
def plot_TF_numerical_vs_analytical_2D(numTF, SDfreq, fiSim, Fe_eff_grc, w,
                                                     P, alpha, params, xname, barname):

        fiSim = fiSim[:, 0]
        fiSim = np.meshgrid(np.zeros(Fe_eff_grc.shape[1]), fiSim)[1]
        levels = np.unique(fiSim)  # to store for colors

        if P is not None:
            params['P'] = P

        # # #### FIGURE AND COLOR GRADIENT STUFF

        fig1 = plt.figure(figsize=(6, 4))
        plt.subplots_adjust(bottom=.2, left=.15, right=.85, wspace=.2)
        ax = plt.subplot2grid((1, 8), (0, 0), colspan=7)
        ax_cb = plt.subplot2grid((1, 8), (0, 7))

        # -- Setting up a colormap that's a simple transtion
        mymap = plt.cm.viridis
        #mymap = get_linear_colormap()
        build_bar_legend(np.round(levels, 1), levels.size, ax_cb, mymap,
                         label=barname + ' ($\\nu_i$ [Hz])')  # BUILD THE COLOR SCALE BAR
        #import pdb
        #pdb.set_trace()
        for i in range(levels.size):
            SIMvector = numTF[i][:]
            SDvector = SDfreq[i][:]
            feSim = Fe_eff_grc[i][:]
            feth = np.linspace(feSim.min(), feSim.max(), int(1e2))
            fi = fiSim[i][0]
            wee = w[i][0]

            _, _, _, muV, sigV, muGn, TvN = get_membrane_fluct_eglif(fi_grid = fi,
                                                                   fe_grid = fe,
                                                                    adapt = adapt,
                                                                    params_SI = params_SI)

            
            Fout_th = erfc_func(muV, sigV, TvN, threshold_func(muV, sigV, TvN, muGn, *P), params['g_L'], params['C_m'], alpha)


            cond = Fout_th <= Fout_th.max()

            r = (float(levels[i]) - levels.min()) / (levels.max() - levels.min())
            #import pdb
            #pdb.set_trace()
            ax.errorbar(feSim[cond], SIMvector[cond], yerr=SDvector[cond], \
                        color=mymap(r, 1), marker='D', ms=5, capsize=3, elinewidth=1, lw=0)


            ax.plot(feSim[cond], Fout_th[cond], color=mymap(r, 1))


        set_plot(ax, ['bottom', 'left'], xlabel=xname+' ($\\nu_{e}$ [Hz])', \
                 ylabel='$\\nu_{out}$ [Hz]')

        plt.show()

  
        diff = np.absolute(np.array([SIMvector - Fout_th]))
        print('QUANTITATIVE SCORE!!!!!\n diff (Numerical TF - Analytic TF)', diff)


        print('----------------------------------------')
        diff_mean = diff.mean()
        diff_max = diff.max()
        diff_sd = diff.std()
        diff_min = diff.min()
        print('Mean diff +- stdev (numTF-SemiAnalyticTF)', diff_mean, '+-', diff_sd)
        print('Min diff', diff_min)

def pseq_params_eglif(params):
    Qe = params['Q_e']
    Te, Ee = params['T_e'], params['E_e']
    Qi, Ti, Ei = params['Q_i'], params['T_i'], params['E_i']
    Gl, Cm, El = params['g_L'], params['C_m'], params['E_L']

    Ke = params['K_e']
    Ki = params['K_i']

    return Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke, Ki
    
def set_plot(ax, spines=['left', 'bottom'],\
                num_xticks=5, num_yticks=5,\
                xlabel='', ylabel='', tck_outward=5,\
                xticks=None, yticks=None,\
                xticks_labels=None, yticks_labels=None,\
                xticks_rotation=0, yticks_rotation=0,\
                xlim_enhancment=2, ylim_enhancment=2,\
                xlim=None, ylim=None):
    
    # drawing spines
    adjust_spines(ax, spines, tck_outward=tck_outward)
    
    # Boundaries
    if xlim is None:
        xmin, xmax = ax.get_xaxis().get_view_interval()
        dx = xmax-xmin
        ax.set_xlim([xmin-xlim_enhancment*dx/100.,xmax+xlim_enhancment*dx/100.])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ymin, ymax = ax.get_yaxis().get_view_interval()
        dy = ymax-ymin
        ax.set_ylim([ymin-ylim_enhancment*dy/100.,ymax+ylim_enhancment*dy/100.])
    else:
        ax.set_ylim(ylim)

    if (xticks is None) and ('bottom' or 'top' in spines):
        ax.xaxis.set_major_locator( MaxNLocator(nbins = num_xticks) )
    else:
        ax.set_xticks(xticks)
        
    if xticks_labels is not None:
        ax.set_xticklabels(xticks_labels, rotation=xticks_rotation)

    if (yticks is None) and ('left' or 'right' in spines):
        ax.yaxis.set_major_locator( MaxNLocator(nbins = num_yticks) )
    else:
        ax.set_yticks(yticks)
        
    if yticks_labels is not None:
        ax.set_yticklabels(yticks_labels, rotation=yticks_rotation)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def adjust_spines(ax, spines, tck_outward=3):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', tck_outward)) # outward by 10 points by default
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

#---------------------- DCNp special ----------------------#
### We currently don't need this function because
###  we are not considering input from IO
def fitting_Vthre_then_Freq_data_eglif_dcnp(muGe_i = None,
                                            muGe_m = None,     
                                            muGi = None,
                                            muG = None,
                                            muV = None,
                                            sigV = None,
                                            muGn = None,
                                            TvN = None, 
                                            Freq_data = None,
                                            fe_m_grid = None, 
                                            fe_i_grid = None,                                             
                                            fi_grid  = None,
                                            adapt = None,
                                            params_SI = None, 
                                            alpha = None,
                                            maxiter=50000, xtol=1e-5, with_square_terms=True):
    
    Gl, Cm, El = params_SI['g_L'], params_SI['C_m'], params_SI['E_L']

    muGe_io, muGe_m, muGi, muG, muV, sigV, muGn, TvN, Tv = get_membrane_fluct_eglif_dcnp(fi_grid = fi_grid,
                                                                                       fe_m_grid = fe_m_grid,
                                                                                       fe_i_grid = fe_i_grid,
                                                                                       adapt = adapt,
                                                                                       params_SI = params_SI)

    freq_data_lim= Freq_data.max()

    i_non_zeros = np.where((Freq_data > 0.) & (Freq_data < freq_data_lim))
    print("Freq_data Limit: ", freq_data_lim)

    Vthre_eff = effective_Vthre(Y = Freq_data[i_non_zeros],
                                muV = muV[i_non_zeros], 
                                sigV = sigV[i_non_zeros],
                                TvN = TvN[i_non_zeros],
                                Gl = params_SI['g_L'],
                                Cm = params_SI['C_m'],
                                alpha = alpha)

    P = [-50e-3, 0, 0, 0, 0]

    print("========== OPTIMIZATION STEP 1 ========== \nVthre Optimization")
    
    def Res(poly_coeff):
        Vthre = threshold_func(*poly_coeff,
                               muV = muV[i_non_zeros],
                               sigV = sigV[i_non_zeros],
                               TvN = TvN[i_non_zeros],
                               muGn = muGn[i_non_zeros],
                               )
        
        return np.mean((Vthre_eff - Vthre) ** 2)

    plsq = minimize(Res, P, options={'disp': True})
    P = plsq.x

    print('========== P output from Vthreshold opt: ', P)

    print("========== OPTIMIZATION STEP 2 ========== \nFreq_data Optimization")

    def Res(poly_coeff):
        return np.mean((Freq_data -
                        TF_template_eglif_dcnp(*poly_coeff,
                                          fe_m = fe_m_grid,
                                          fe_i = fe_i_grid,
                                          fi = fi_grid,
                                          adapt = adapt, 
                                          alpha = alpha,
                                          params_SI = params_SI)) ** 2)


    plsq = minimize(Res, P, method='nelder-mead',
                    options={'xatol': xtol, 'disp': True, 'maxiter': maxiter})

    P = plsq.x
    print('========== P output from Freq_data opt: ', P)

    print("========== END OF OPTIMIZATION PROCESS ==========")

    params_SI['P'] = P

    return P

### not currently used because we are not considering the input from IO ###    
def get_membrane_fluct_eglif_dcnp(fi_grid = None, 
                                  fe_m_grid = None,
                                  fe_i_grid = None,                                  
                                  adapt = None,
                                  params_SI = None):

    Cm = params_SI['C_m'] # membrane capacitance
    Gl = params_SI['g_L'] # leak condunctance
    El = params_SI['E_L'] # leak equilibrium potential

    # excitatory synapses
    Qe_m = params_SI['Q_e_m']
    Te_m = params_SI['T_e_m']
    Ee = params_SI['E_e']
    Ke_m = params_SI['K_e_m']

    Qe_i = params_SI['Q_e_i']
    Te_i = params_SI['T_e_i']
    #Ee = params_SI['E_e'] #one Ee for both exc syn
    Ke_i = params_SI['K_e_i']    
    
    # inhibitory synapses    
    Qi = params_SI['Q_i']
    Ti = params_SI['T_i']
    Ei = params_SI['E_i']
    Ki = params_SI['K_i']
    
    fi = fi_grid
    fe_m = fe_m_grid
    fe_i = fe_i_grid


    # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
    muGe_i, muGe_m, muGi = Qe_i * Ke_i * Te_i * fe_i, Qe_m * Ke_m * Te_m * fe_m, Qi * Ki * Ti * fi #EQUAL TO EXP
    # ---------------------------- Input cond:  mu PC -----------------------------------------------
    muG = Gl + muGe_i + muGe_m + muGi #EQUAL TO EXP
    # ---------------------------- Membrane Fluctuation Properties ----------------------------------
    muV = (np.e * (muGe_i * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - adapt) / muG  # XX = adaptation

    muGn, Tm = muG / Gl, Cm / muG  # normalization

    Ue_i, Ue_m, Ui = Qe_i / muG * (Ee - muV), Qe_m / muG * (Ee - muV), Qi / muG * (Ei - muV) #EQUAL TO EXP



    sigVe_i = (2 * Tm + Te_i) * ((np.e * Ue_i * Te_i)/ (2 * (Te_i + Tm))) ** 2 * Ke_i * fe_i
    sigVe_m = (2 * Tm + Te_m) * ((np.e * Ue_m * Te_m) / (2 * (Te_m + Tm))) ** 2 * Ke_m * fe_m
    sigVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2* (Ti + Tm))) ** 2 * Ki * fi

    sigV = np.sqrt(sigVe_i + sigVe_m + sigVi)

    fe_m, fe_i, fi = fe_m + 1e-15, fe_i + 1e-15, fi + 1e-15  # just to insure a non zero division

    Tv_num= Ke_i * fe_i * Ue_i ** 2 * Te_i ** 2 * np.e ** 2 + \
            Ke_m * fe_m * Ue_m ** 2 * Te_m ** 2 * np.e ** 2 + \
            Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
    Tv = 0.5 * Tv_num / ((sigV+1e-20) ** 2)


    TvN = Tv * Gl / Cm  # normalization

    return muGe_i, muGe_m, muGi, muG, muV, sigV+1e-20, muGn, TvN, Tv

### not currently used because we are not considering the input from IO ###    
def pseq_params_eglif_dcnp(params):

    Qe_i, Qe_m = params['Q_e_i'], params['Q_e_m']
    Te_i, Te_m, Ee = params['T_e_i'], params['T_e_m'], params['E_e']
    Qi, Ti, Ei = params['Q_i'], params['T_i'], params['E_i']
    Gl, Cm , El = params['g_L'], params['C_m'] , params['E_L']

    Ke_i = params['K_e_i']
    Ke_m = params['K_e_m']
    Ki = params['K_i']

    return Qe_i, Qe_m, Te_i, Te_m, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_i, Ke_m, Ki

### not currently used because we are not considering the input from IO ###    
def TF_template_eglif_dcnp(P0, P1, P2, P3, P4,
                      fe_i = None,
                      fe_m = None,
                      fi = None,
                      adapt = None, 
                      alpha = None,
                      params_SI = None):
    
    # here TOTAL (sum over synapses) excitatory and inhibitory input

    if (hasattr(fe_m, "__len__")):
        fe_m[fe_m < 1e-8] = 1e-8
    else:
        if (fe_m < 1e-8):
            fe_m = 1e-8
    if (hasattr(fe_i, "__len__")):
        fe_i[fe_i < 1e-8] = 1e-8
    else:
        if (fe_i < 1e-8):
            fe_i = 1e-8
    if (hasattr(fi, "__len__")):
        fi[fi < 1e-8] = 1e-8
    else:
        if (fi < 1e-8):
            fi = 1e-8

    muGe_io, muGe_m, muGi, muG, muV, sigV, muGn, TvN, Tv = get_membrane_fluct_eglif_dcnp(fi_grid = fi,
                                                                     fe_m_grid = fe_m,
                                                                     fe_i_grid = fe_i,
                                                                     adapt = adapt,
                                                                     params_SI = params_SI)


    Vthre = threshold_func(muV = muV,
                           sigV = sigV,
                           TvN = TvN,
                           muGn = muGn,
                           P0 = P0,
                           P1 = P1,
                           P2 = P2,
                           P3 = P3,
                           P4 = P4)


    if (hasattr(muV, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        sigV[sigV < 1e-4] = 1e-4
    else:
        if (sigV < 1e-4):
            sigV = 1e-4

    Fout_th = erfc_func(muV = muV,
                        sigV = sigV,
                        TvN = TvN,
                        Vthre = Vthre,
                        Gl = params_SI['g_L'],
                        Cm = params_SI['C_m'],
                        alpha = alpha)

    if (hasattr(Fout_th, "__len__")):
        # print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        Fout_th[Fout_th < 1e-8] = 1e-8
    else:
        if (Fout_th < 1e-8):
            Fout_th = 1e-8

    return Fout_th

### not currently used because we are not considering the input from IO ###    
def plot_TF_numerical_vs_analytical_3D_sliced(numTF, SDfreq, fiSim, fe_i,fe_m, w,
                                                     P, alpha, params, xname, barname,
                                                     title = None):

        fiSim = fiSim[:, 0]

        fiSim = np.meshgrid(np.zeros(fe_i.shape[1]), fiSim)[1] 
        levels = np.unique(fiSim)  # to store for colors

        if P is not None:
            params['P'] = P

        # # #### FIGURE AND COLOR GRADIENT STUFF

        fig1 = plt.figure(figsize=(6, 4))
        plt.subplots_adjust(bottom=.2, left=.15, right=.85, wspace=.2)
        ax = plt.subplot2grid((1, 8), (0, 0), colspan=7)
        ax_cb = plt.subplot2grid((1, 8), (0, 7))

        # -- Setting up a colormap that's a simple transtion
        mymap = plt.cm.viridis
        #mymap = get_linear_colormap()
        build_bar_legend(np.round(levels, 1), levels.size, ax_cb, mymap,
                         label=barname + ' ($\\nu_i$ [Hz])')  # BUILD THE COLOR SCALE BAR
        #import pdb
        #pdb.set_trace()
        for i in range(levels.size):
            SIMvector = numTF[i][:]
            SDvector = SDfreq[i][:]
            feIO = fe_i[i][:]
            feM = fe_m[i][:]
            feth = np.linspace(feIO.min(), feIO.max(), int(1e2))
            fi = fiSim[i][0]
            wee = w[i][0]

            _, _, _, _, muV, sigV, muGn, TvN, Tv = get_membrane_fluct_eglif_dcnp(fi_grid = fi,
                                                                     fe_m_grid = fe_m,
                                                                     fe_i_grid = fe_i,
                                                                     adapt = w,
                                                                     params_SI = params)
            
            Fout_th = erfc_func(muV = muV,
                        sigV = sigV,
                        TvN = TvN,
                        Vthre = threshold_func(muV = muV,
                           sigV = sigV,
                           TvN = TvN,
                           muGn = muGn,
                           P0 = P[0],
                           P1 = P[1],
                           P2 = P[2],
                           P3 = P[3],
                           P4 = P[4]),
                        Gl = params['g_L'],
                        Cm = params['C_m'],
                        alpha = alpha)


            #cond = Fout_th <= Fout_th.max()

            r = (float(levels[i]) - levels.min()) / (levels.max() - levels.min())
            #import pdb
            #pdb.set_trace()
            #ax.errorbar(feM[cond], SIMvector[cond], yerr=SDvector[cond], \
            #            color=mymap(r, 1), marker='D', ms=5, capsize=3, elinewidth=1, lw=0)
            ax.errorbar(feM, SIMvector, yerr=SDvector, \
                        color=mymap(r, 1), marker='D', ms=5, capsize=3, elinewidth=1, lw=0)


            #ax.plot(feM[cond], Fout_th[cond], color=mymap(r, 1))
            ax.plot(feM, Fout_th[i][:], color=mymap(r, 1))

        set_plot(ax, ['bottom', 'left'], xlabel=xname+' ($\\nu_{e}$ [Hz])', \
                 ylabel='$\\nu_{out}$ [Hz]')
        if title != None:
            ax.set_title(title)
        plt.show()

        '''
        diff = np.absolute(np.array([SIMvector - Fout_th]))
        print('QUANTITATIVE SCORE!!!!!\n diff (Numerical TF - Analytic TF)', diff)


        print('----------------------------------------')
        diff_mean = diff.mean()
        diff_max = diff.max()
        diff_sd = diff.std()
        diff_min = diff.min()
        print('Mean diff +- stdev (numTF-SemiAnalyticTF)', diff_mean, '+-', diff_sd)
        print('Min diff', diff_min)
        '''
    
### not currently used because we are not considering the input from IO ###    
def plot_TF_surfaces_by_IO(z_s, fe_m_grid, fe_i_grid, fi_grid, df_DCNp_array,
                            cmap_col='Purples', zlabel='Output rate (Hz)',
                            xlabel='Excitatory input (via mossy RS)',
                            ylabel='Inhibitory input (via PC)',
                            title='3D Surfaces of numerica TF colored by IO'):
    """
    Plot 3D surfaces of numerical transfer functions for each IO frequency.
    Each IO frequency corresponds to one surface, color-coded by IO value.
    """

    #fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot(111, projection='3d')

    n_IO = len(z_s)
    cmap = cm.get_cmap(cmap_col, n_IO)
    norm = colors.BoundaryNorm(boundaries=np.arange(n_IO+1)-0.5, ncolors=n_IO)
    figs=[]
    for idx_io, io_val in enumerate(z_s):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')        
        # Extract the data slice for this IO frequency
        Z = df_DCNp_array[:, :, idx_io]
        X = fe_m_grid[:, :, idx_io]
        Y = fi_grid[:, :, idx_io]

        # Discrete color for this IO
        color = cmap(idx_io)
        facecolors = np.tile(color, (Z.shape[0], Z.shape[1], 1))

        # Create surface
        ax.plot_surface(X, Y, Z, facecolors=facecolors, edgecolor='none', alpha=0.85)

        # --- Labels and style
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
    
        # --- Colorbar for IO
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, ticks=np.arange(n_IO))
        cbar.set_ticklabels([str(v) for v in z_s])
        cbar.set_label('Input via IO (Hz)')
    
        plt.tight_layout()
        figs.append(fig)
    return figs

### not currently used because we are not considering the input from IO ###    
def plot_TF_surfaces_by_IO_with_analytic(z_s, fe_m_grid, fe_i_grid, fi_grid,
                                         df_DCNp_array, w,
                                         params,
                                         alpha,
                                         TF_params,
                                         cmap_col='Purples',
                                         zlabel='Output rate (Hz)',
                                         xlabel='Excitatory input (via mossy)',
                                         ylabel='Inhibitory input (via PC)',
                                         title='Numerical vs Analytical TF Surfaces'):
    """
    For each IO frequency:
        - plot the numerical transfer function surface
        - overlay analytical transfer function surface computed from TF_func
    """

    #import pdb
    #pdb.set_trace()
    n_IO = len(z_s)
    cmap = cm.get_cmap(cmap_col, n_IO)

    #cmap_ana = cm.get_cmap('Reds', n_IO).reversed()

    norm = colors.BoundaryNorm(boundaries=np.arange(n_IO+1)-0.5, ncolors=n_IO)
    figs = []

    for idx_io, io_val in enumerate(z_s):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Get grid and numerical TF
        X = fe_m_grid[:, :, idx_io]
        Y = fi_grid[:, :, idx_io]
        Z_num = df_DCNp_array[:, :, idx_io]

        # --- Compute analytical TF
        Z_ana = TF_template_eglif_dcnp(TF_params[0],TF_params[1],TF_params[2],TF_params[3], TF_params[4],
            fe_m = fe_m_grid[:, :, idx_io],
            fe_i = fe_i_grid[:, :, idx_io],
            fi = fi_grid[:, :, idx_io],
            adapt = w[:, :, idx_io],
            alpha = alpha, 
            params_SI = params)

        # --- Numerical surface
        color_P = cmap(idx_io)
        facecolors = np.tile(color_P, (Z_num.shape[0], Z_num.shape[1], 1))
        ax.plot_surface(X, Y, Z_num, facecolors=facecolors, edgecolor='none', alpha=0.85)

        # --- Analytical surface (wireframe)
        #ax.plot_wireframe(X, Y, Z_ana, color=cmap_ana(idx_io), linewidth=1, alpha=1)
        ax.plot_wireframe(X, Y, Z_ana, color='r', linewidth=1, alpha=1)

        # --- Labels and colorbar
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(f'{title}\nIO input: {io_val:.1f} Hz')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, ticks=np.arange(n_IO))
        cbar.set_ticklabels([str(v) for v in z_s])
        cbar.set_label('Input via IO (Hz)')

        plt.tight_layout()
        figs.append(fig)

    return figs

###########################################################################################################################################

