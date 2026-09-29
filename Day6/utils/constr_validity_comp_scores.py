
import numpy as np

import os
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from scipy import signal
from scipy.stats import ttest_ind
from scipy import stats
from scipy.stats import spearmanr, pearsonr, entropy, ttest_rel, wilcoxon
from scipy.signal import welch, coherence, butter, filtfilt

import networkx as nx

from numpy.linalg import norm


def get_indeces(n_regions, ROB_idx, DCN_idx, CBL_idx):
    
    _inds = {}
    
    _inds["crbl"]  = CBL_idx
    _inds["cortical"] = ROB_idx
    _inds["dcn"] = DCN_idx
    _region_label_mock = np.arange(1,n_regions+1,1)

    _inds["crbl"] = np.arange(len(_region_label_mock)).astype('i') #metto al posto di inds crbl un vettore di interi lungo quando argomento di arange()
    #print("inds after astype\n",inds)
    _inds["crbl"] = np.delete(_inds["crbl"], _inds["cortical"]) #delete inds cortical(second input) from inds crbl(first input)
    #print("inds afeter delete\n", inds)
    _is_cortical = np.array([False] * _region_label_mock.shape[0]).astype("bool") #inizializzo a false
    #print('shape 0 di region_labels:',region_label_mock.shape[0])
    #print('Iscortical:\n', is_cortical)
    _is_cortical[_inds["cortical"]] = True
    _is_cortical[_inds["dcn"]] = True
    #print('Final Iscortical:\n', _is_cortical)
    _is_crbl = np.logical_not(_is_cortical)
    #print('Final crbl:\n', _is_crbl)
    
    return _is_crbl, _is_cortical

    
def get_cerebellar_cortex(myBOLD, start_crbl, DCN_idx):
    
    # # Remove Deep Cerebellar Nuclei
    mask = np.ones(np.shape(myBOLD)[1], dtype=bool)
    mask[DCN_idx] = False
    myBOLD = myBOLD[:,mask]
    myBOLD = myBOLD[:, start_crbl:]
    return myBOLD

def get_dcn(myBOLD, DCN_idx):
    
    return myBOLD[:, DCN_idx]

def get_cerebellum_wDCN(myBOLD, start_crbl):
    return myBOLD[:, start_crbl:]

def get_bst(myBOLD, BST_idx, DCN_idx):
    return myBOLD[:,BST_idx]
    #return myBOLD[:, BST_idx + DCN_idx]

def get_cerebrum(myBOLD, start_crbl):
    return myBOLD[:,:start_crbl]


def compute_mae(subject_ids, bold_MF, bold_WW, bold_EMP):
    # Initialize lists to store MAE values
    
    mae_mf_all_subjects = []
    mae_ww_all_subjects = []

    mae_SD_mf_all_subjects = []
    mae_SD_ww_all_subjects = []


    mae_mf_regions_all_subjects = []
    mae_ww_regions_all_subjects = []

    # Compute MAE for each subject and each region
    for i in range(len(subject_ids)):
        mae_mf_subject = mean_absolute_error(bold_EMP[i], bold_MF[i], multioutput='raw_values')
        mae_ww_subject = mean_absolute_error(bold_EMP[i], bold_WW[i], multioutput='raw_values')

        # Append the overall average MAE for each subject
        mae_mf_all_subjects.append(np.mean(mae_mf_subject))
        mae_ww_all_subjects.append(np.mean(mae_ww_subject))

        mae_SD_mf_all_subjects.append(np.std(mae_mf_subject))
        mae_SD_ww_all_subjects.append(np.std(mae_ww_subject))

        # Append the region-specific MAE for each subject
        mae_mf_regions_all_subjects.append(mae_mf_subject)
        mae_ww_regions_all_subjects.append(mae_ww_subject)

    # Convert to numpy arrays for easier manipulation
    mae_mf_regions_all_subjects = np.array(mae_mf_regions_all_subjects)
    mae_ww_regions_all_subjects = np.array(mae_ww_regions_all_subjects)

    # Compute average MAE for each region across all subjects
    mae_mf_regions_avg = np.mean(mae_mf_regions_all_subjects, axis=0)
    mae_ww_regions_avg = np.mean(mae_ww_regions_all_subjects, axis=0)

    mae_mf_regions_SD = np.std(mae_mf_regions_all_subjects, axis=0)
    mae_ww_regions_SD = np.std(mae_ww_regions_all_subjects, axis=0)

    return mae_mf_all_subjects, mae_ww_all_subjects, mae_SD_mf_all_subjects, mae_SD_ww_all_subjects, mae_mf_regions_all_subjects, mae_ww_regions_all_subjects, mae_mf_regions_avg, mae_ww_regions_avg, mae_mf_regions_SD, mae_ww_regions_SD


def compute_mae_4(subject_ids, bold_MF, bold_WW, bold_WC, bold_GO, bold_EMP):
    # Initialize lists to store MAE values
    
    mae_mf_all_subjects = []
    mae_ww_all_subjects = []
    mae_wc_all_subjects = []
    mae_go_all_subjects = []

    mae_SD_mf_all_subjects = []
    mae_SD_ww_all_subjects = []
    mae_SD_wc_all_subjects = []
    mae_SD_go_all_subjects = []


    mae_mf_regions_all_subjects = []
    mae_ww_regions_all_subjects = []
    mae_wc_regions_all_subjects = []
    mae_go_regions_all_subjects = []


    # Compute MAE for each subject and each region
    for i in range(len(subject_ids)):
        mae_mf_subject = mean_absolute_error(bold_EMP[i], bold_MF[i], multioutput='raw_values')
        mae_ww_subject = mean_absolute_error(bold_EMP[i], bold_WW[i], multioutput='raw_values')
        mae_wc_subject = mean_absolute_error(bold_EMP[i], bold_WC[i], multioutput='raw_values')
        mae_go_subject = mean_absolute_error(bold_EMP[i], bold_GO[i], multioutput='raw_values')

        # Append the overall average MAE for each subject
        mae_mf_all_subjects.append(np.mean(mae_mf_subject))
        mae_ww_all_subjects.append(np.mean(mae_ww_subject))
        mae_wc_all_subjects.append(np.mean(mae_wc_subject))
        mae_go_all_subjects.append(np.mean(mae_go_subject))

        mae_SD_mf_all_subjects.append(np.std(mae_mf_subject))
        mae_SD_ww_all_subjects.append(np.std(mae_ww_subject))
        mae_SD_wc_all_subjects.append(np.std(mae_wc_subject))
        mae_SD_go_all_subjects.append(np.std(mae_go_subject))

        # Append the region-specific MAE for each subject
        mae_mf_regions_all_subjects.append(mae_mf_subject)
        mae_ww_regions_all_subjects.append(mae_ww_subject)
        mae_wc_regions_all_subjects.append(mae_wc_subject)
        mae_go_regions_all_subjects.append(mae_go_subject)


    # Convert to numpy arrays for easier manipulation
    mae_mf_regions_all_subjects = np.array(mae_mf_regions_all_subjects)
    mae_ww_regions_all_subjects = np.array(mae_ww_regions_all_subjects)
    mae_wc_regions_all_subjects = np.array(mae_wc_regions_all_subjects)
    mae_go_regions_all_subjects = np.array(mae_go_regions_all_subjects)

    # Compute average MAE for each region across all subjects
    mae_mf_regions_avg = np.mean(mae_mf_regions_all_subjects, axis=0)
    mae_ww_regions_avg = np.mean(mae_ww_regions_all_subjects, axis=0)
    mae_wc_regions_avg = np.mean(mae_wc_regions_all_subjects, axis=0)
    mae_go_regions_avg = np.mean(mae_go_regions_all_subjects, axis=0)

    mae_mf_regions_SD = np.std(mae_mf_regions_all_subjects, axis=0)
    mae_ww_regions_SD = np.std(mae_ww_regions_all_subjects, axis=0)
    mae_wc_regions_SD = np.std(mae_wc_regions_all_subjects, axis=0)
    mae_go_regions_SD = np.std(mae_go_regions_all_subjects, axis=0)

    return mae_mf_all_subjects, mae_ww_all_subjects, mae_wc_all_subjects, mae_go_all_subjects,\
          mae_SD_mf_all_subjects, mae_SD_ww_all_subjects,mae_SD_wc_all_subjects, mae_SD_go_all_subjects, \
            mae_mf_regions_all_subjects, mae_ww_regions_all_subjects, mae_wc_regions_all_subjects, mae_go_regions_all_subjects, \
                mae_mf_regions_avg, mae_ww_regions_avg, mae_wc_regions_avg, mae_go_regions_avg,\
                mae_mf_regions_SD, mae_ww_regions_SD, mae_wc_regions_SD, mae_go_regions_SD


def check_normality(models, maes):
    # Shapiro-Wilk test for normality
    #models = ['cerebellar MF', 'WW model', 'WC model', 'GO model']
    #maes = [mae_mf_all_subjects, mae_ww_all_subjects, mae_wc_all_subjects, mae_go_all_subjects]

    shapiro_results = {}
    for model, mae in zip(models, maes):
        stat, p_value = stats.shapiro(mae)
        shapiro_results[model] = (stat, p_value)
    
    # Print results
    for model, (stat, p_value) in shapiro_results.items():
        print(f"Shapiro-Wilk test for {model}: statistic = {stat:.4f}, p-value = {p_value:.4e}")
    
    return shapiro_results


def check_homogeneity_of_variance(maes):
    # Levene's test for homogeneity of variance
    stat, p_value = stats.levene(*maes)
    
    # Print result
    print(f"Levene's test for homogeneity of variance: statistic = {stat:.4f}, p-value = {p_value:.4e}")
    
    return stat, p_value


def perform_stat_tests(models, maes, test_name, outdir):
    
    #models = ['cerebellar MF', 'WW model', 'WC model', 'GO model']
    #maes = [mae_mf_all_subjects, mae_ww_all_subjects, mae_wc_all_subjects, mae_go_all_subjects]
    
    if test_name == 'ttest':
        t_test_results = {}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                t_stat, p_value = ttest_ind(maes[i], maes[j])
                t_test_results[f"{models[i]} vs {models[j]}"] = (t_stat, p_value)

        # Print t-test results
        for comparison, (t_stat, p_value) in t_test_results.items():
            print(f"{comparison}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")

        # Save t-test results to a file
        with open(outdir + 't_test_results_4models.txt', 'w') as f:
            for comparison, (t_stat, p_value) in t_test_results.items():
                f.write(f"{comparison}: t-statistic = {t_stat:.6f}, p-value = {p_value:.6e}\n")
    
    elif test_name == 'MannW':
        mann_whitney_results = {}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                u_stat, p_value = stats.mannwhitneyu(maes[i], maes[j], alternative='two-sided')
                mann_whitney_results[f"{models[i]} vs {models[j]}"] = (u_stat, p_value)

        # Print Mann-Whitney U test results
        for comparison, (u_stat, p_value) in mann_whitney_results.items():
            print(f"{comparison}: U-statistic = {u_stat:.6f}, p-value = {p_value:.6e}")

        # Save Mann-Whitney U test results to a file
        with open(outdir + 'mann_whitney_results.txt', 'w') as f:
            for comparison, (u_stat, p_value) in mann_whitney_results.items():
                f.write(f"{comparison}: U-statistic = {u_stat:.6f}, p-value = {p_value:.6e}\n")

    else: print('Unknown test: choose ttest or Mannw')


def compute_coherence(empBOLD_m, simBOLD_m, TR):
    
    num_regions = empBOLD_m.shape[1]
    freq_coherence_list = []
    
    coherence_list = []
    for i in range(num_regions):
        freq_coherence, Cxy = signal.coherence(empBOLD_m[:, i] - np.mean(empBOLD_m[:, i]),
                                               simBOLD_m[:, i] - np.mean(simBOLD_m[:, i]), fs=1/TR)
        freq_coherence_list.append(freq_coherence)
        coherence_list.append(Cxy)
    coherence_results = {
        'freq_coherence': freq_coherence_list,
        'coherence': coherence_list
    }
    return coherence_results


def compute_coherence_loop(empBOLD_m, simBOLD_m, TR, subject_scores, sub_id, regions_name):
    
    coherence_results = compute_coherence(empBOLD_m, simBOLD_m, TR)
    # Calculate average, standard deviation, and maximum coherence per region
    
    num_regions = empBOLD_m.shape[1]
    
    for region_idx in range(num_regions):
        coherence_values = coherence_results['coherence'][region_idx]
        avg_coherence = np.mean(coherence_values)
        sd_coherence = np.std(coherence_values)
        max_coherence = np.max(coherence_values)
        
        subject_scores.append({
            'subject_id': sub_id,
            'region': regions_name[region_idx],
            'avg_coherence': avg_coherence,
            'sd_coherence': sd_coherence,
            'max_coherence': max_coherence
            })
        
    return subject_scores


def comp_FC(myBOLD):
    """
    input: bold data
    return: simulated FC (PCC matrix)
    """

    FC = np.corrcoef(myBOLD.T) #BOLD shoul be #region x time
    np.fill_diagonal(FC,0)

    return FC

def compute_CCs(*FCs, corr_type='PCC'):
    """
    Compute the Spearman correlation coefficient between multiple FC matrices for each subject.

    Parameters:
        *args: Variable number of FC matrices as input, each of shape (nsubj, nregion, nregion).

    Returns:
        spearman_correlations: List of lists with Spearman correlation coefficients for each subject.
    """
    num_matrices = len(FCs)
    nsubj = FCs[0].shape[0]
    spearman_correlations = []

    # Ensure all input matrices are NumPy arrays
    fc_matrices = [np.array(fc) for fc in FCs]

    if corr_type == 'PCC':
        funz = pearsonr
    elif corr_type == 'SCC':
        funz = spearmanr
    else:
        print('Wrong corr_type option, choose between PCC and SCC')

    # Compute Spearman correlation for each pair of matrices for each subject
    for subj in range(nsubj):
        subj_correlations = []
        for i in range(num_matrices):
            for j in range(i + 1, num_matrices):
                matrix1 = fc_matrices[i][subj].flatten()
                matrix2 = fc_matrices[j][subj].flatten()
                
                corr, _ = funz(matrix1, matrix2)


                subj_correlations.append(((i, j), corr))
                print(f"Subject {subj + 1}: {corr_type} between matrix {i} and matrix {j}: {corr}")
        spearman_correlations.append(subj_correlations)
    
    return spearman_correlations


def compute_psd(bold_data, fs):
    """
    Compute the Power Spectral Density (PSD) for each region and each subject.

    Parameters:
        bold_data: 3D NumPy array with shape (nsubj, timepoint, nregion)
        fs: Sampling frequency

    Returns:
        f: Array of sample frequencies
        psd: 3D NumPy array with shape (nsubj, nregion, nfreq)
    """
    nsubj, timepoint, nregion = bold_data.shape
    
    # Compute PSD for the first subject and region to determine the shape
    f, Pxx = welch(bold_data[0, :, 0], fs=fs, nperseg=min(256, timepoint))
    nfreq = len(f)
    
    # Initialize PSD array with the correct shape
    psd = np.zeros((nsubj, nregion, nfreq))
    
    # Compute PSD for each subject and region
    for subj in range(nsubj):
        for region in range(nregion):
            _, Pxx = welch(bold_data[subj, :, region], fs=fs, nperseg=min(256, timepoint))
            psd[subj, region, :] = Pxx
    
    return f, psd


def compute_spectral_coherence(bold_data, fs, nperseg):
    """
    Compute the spectral coherence between all pairs of regions for each subject.

    Parameters:
        bold_data: 3D NumPy array with shape (nsubj, timepoint, nregion)
        fs: Sampling frequency (1/TR for bold signal!)

    Returns:
        f: Array of sample frequencies
        coherence_matrices: 4D NumPy array with shape (nsubj, nregion, nregion, nfreq)

    Additional notes:
    - Coherence  = measure of the linear relationship between the frequency components of two signals.
    - nperseg = size window for each FFT calculation. Defualt value = 256. 
                Rule of Thumb: common choices for about 100 timepoint = 128, 256, 512
                Too small = loss of freq details
                Too large = less number of "segments" to averaged = variance increases 
    - Usage in the hybrid TVB: evaluating emergent functional network 
        (cannot do ICA because I have one singal per region); Limitation: only linear relations

    """
    nsubj, timepoint, nregion = bold_data.shape
    #nperseg = min(256, timepoint)
    
    # Compute coherence for the first subject and region pair to determine the shape of the coherence matrix.
    f, Cxy = coherence(bold_data[1, :, 0], bold_data[1, :, 1], fs=fs, nperseg=nperseg)
    
    #print('Coherence shape for one subject and one region', np.shape(Cxy),'\nfreqs: ', f)
    
    nfreq = len(f)
    
    # Initialize coherence matrix with the correct shape
    coherence_matrices = np.zeros((nsubj, nregion, nregion, nfreq))
    
    # Compute coherence for each subject and pair of regions
    for subj in range(nsubj):
        for region1 in range(nregion):
            for region2 in range(nregion):
                _, Cxy = coherence(bold_data[subj, :, region1], bold_data[subj, :, region2], fs=fs, nperseg=nperseg)
                
                coherence_matrices[subj, region1, region2, :] = Cxy

    
    return f, coherence_matrices


def compute_coherence_matrix_scores(output_csv, *coherence_matrices):
    """
    Compute scores to compare coherence matrices and saved them in a csv file.

    Parameters:
        output_csv (str): Path to save the output CSV file containing scores.
        *coherence_matrices: coherence matrices (* = variables number of matices).
                             Expected order: (simulated_1, simulated_2, ..., empirical)
                             All matrices should have the shape (nsubjects, nregions, nregions).
    
    Returns:
        A CSV file with subjects on cols and scores on rows.
    """
    # Unpack the coherence matrices. PASS AS INPUT BEFORE THE SIMULATION AND THEN THE EMPIRICAL
    *simulations, empirical_data = coherence_matrices

    nsubjects = empirical_data.shape[0]


    scores_list = []

    
    for subj in range(nsubjects):
        subject_scores = {'Subject': subj + 1}


        emp_mat = empirical_data[subj]

        for i, sim_mat in enumerate(simulations):
       
            sim_mat = sim_mat[subj]

            # # Mean Squared Error (MSE)
            # # Overall error metric - goodness-of-fit between the simulated and empirical data
            mse = mean_absolute_error(sim_mat.flatten(), emp_mat.flatten())
            subject_scores[f'Simulation_{i+1}_MSE'] = mse

            # Frobenius Norm
            # # Measure of difference between 2 mat by computing
            #  the sum of the squares of their element-wise differences
            # # ---- overall deviation between the two matrices
            frobenius_norm = np.linalg.norm(sim_mat - emp_mat, ord='fro')
            subject_scores[f'Simulation_{i+1}_Frobenius'] = frobenius_norm

            # Compute Cosine Similarity
            # # IT DOESN'T COMPARE MAGNITUDE OF THE VALUES!!!!
            # # Similarity in the direction of 2 vectors (i.e., flattened matrices)
            cosine_sim = cosine_similarity(sim_mat.flatten().reshape(1, -1), emp_mat.flatten().reshape(1, -1))[0][0]
            subject_scores[f'Simulation_{i+1}_Cosine_Similarity'] = cosine_sim

            # Compute Pearson Correlation
            pearson_corr, _ = pearsonr(sim_mat.flatten(), emp_mat.flatten())
            subject_scores[f'Simulation_{i+1}_Pearson'] = pearson_corr

            # Compute Spearman Correlation
            spearman_corr, _ = spearmanr(sim_mat.flatten(), emp_mat.flatten())
            subject_scores[f'Simulation_{i+1}_Spearman'] = spearman_corr

            # Compute KL Divergence
            # Normalize matrices to ensure they sum to 1 (for KL divergence to work as a probability distribution)
            # # It measures the difference between the probability distributions of the two matrices
            sim_flat_norm = sim_mat.flatten() / np.sum(sim_mat.flatten())
            emp_flat_norm = emp_mat.flatten() / np.sum(emp_mat.flatten())
            kl_divergence = entropy(sim_flat_norm, emp_flat_norm)
            subject_scores[f'Simulation_{i+1}_KL_Divergence'] = kl_divergence

        scores_list.append(subject_scores)

    # Create a DataFrame from the results and save it to CSV
    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(output_csv, index=False)

    return scores_df


def compute_coherence_matrix_scores_stats(output_csv, *coherence_matrices):
    """
    Same as compute_coherence_matrix_scores (see above)
    Returns:
        A CSV file with the scores and statistical comparison results.
    """
    # Unpack the coherence matrices
    *simulations, empirical_data = coherence_matrices

    nsubjects = simulations[0].shape[0]
    
    # Prepare dictionaries to hold the scores
    scores_dict_sim1 = {}
    scores_dict_sim2 = {}
    
    # List of score types
    score_types = ['Pearson', 'Spearman', 'Frobenius_Norm', 'Cosine_Similarity', 'MAE', 'KL']
    
    for score in score_types:
        scores_dict_sim1[score] = []
        scores_dict_sim2[score] = []

    # Compute scores for each subject and store them
    for subj in range(nsubjects):
        avg_empirical = empirical_data[subj]
        for i, sim_mat in enumerate(simulations):
            avg_sim = sim_mat[subj]
            
            avg_empirical_t = np.triu(avg_empirical) 
            avg_sim_t = np.triu(avg_sim)

            # Flatten matrices for correlation and similarity calculations
            avg_emp_flat = avg_empirical_t.flatten()
            avg_sim_flat = avg_sim_t.flatten()

            # Calculate scores for each subject and store them
            if i == 0:
                # Simulation 1
                pearson_corr, _ = pearsonr(avg_emp_flat, avg_sim_flat)
                spearman_corr, _ = spearmanr(avg_emp_flat, avg_sim_flat)
                frobenius_norm = np.linalg.norm(avg_empirical - avg_sim, 'fro')
                cos_sim = cosine_similarity(avg_emp_flat.reshape(1, -1), avg_sim_flat.reshape(1, -1))[0][0]
                mae = mean_absolute_error(avg_emp_flat, avg_sim_flat)
                sim_flat_norm = avg_sim_t.flatten() / np.sum(avg_sim_t.flatten())
                emp_flat_norm = avg_empirical_t.flatten() / np.sum(avg_empirical_t.flatten())
                kl_divergence = entropy(sim_flat_norm, emp_flat_norm)

                scores_dict_sim1['Pearson'].append(pearson_corr)
                scores_dict_sim1['Spearman'].append(spearman_corr)
                scores_dict_sim1['Frobenius_Norm'].append(frobenius_norm)
                scores_dict_sim1['Cosine_Similarity'].append(cos_sim)
                scores_dict_sim1['MAE'].append(mae)
                scores_dict_sim1['KL'].append(kl_divergence)

            else:
                # Simulation 2
                pearson_corr, _ = pearsonr(avg_emp_flat, avg_sim_flat)
                spearman_corr, _ = spearmanr(avg_emp_flat, avg_sim_flat)
                frobenius_norm = np.linalg.norm(avg_empirical - avg_sim, 'fro')
                cos_sim = cosine_similarity(avg_emp_flat.reshape(1, -1), avg_sim_flat.reshape(1, -1))[0][0]
                mae = mean_absolute_error(avg_emp_flat, avg_sim_flat)
                sim_flat_norm = avg_sim_t.flatten() / np.sum(avg_sim_t.flatten())
                emp_flat_norm = avg_empirical_t.flatten() / np.sum(avg_empirical_t.flatten())
                kl_divergence = entropy(sim_flat_norm, emp_flat_norm)
      

                scores_dict_sim2['Pearson'].append(pearson_corr)
                scores_dict_sim2['Spearman'].append(spearman_corr)
                scores_dict_sim2['Frobenius_Norm'].append(frobenius_norm)
                scores_dict_sim2['Cosine_Similarity'].append(cos_sim)
                scores_dict_sim2['MAE'].append(mae)
                scores_dict_sim2['KL'].append(kl_divergence)

    # Statistical comparison (paired t-test or Wilcoxon test) between the two simulations for each score type
    stats_results = {}
    for score in score_types:
        sim1_scores = scores_dict_sim1[score]
        sim2_scores = scores_dict_sim2[score]

        t_stat, p_value = ttest_rel(sim1_scores, sim2_scores)
        #t_stat, p_value = stats.mannwhitneyu(sim1_scores, sim2_scores, alternative='two-sided')


        stats_results[f'{score}_t_stat'] = t_stat
        stats_results[f'{score}_p_value'] = p_value

    # Combine the score dictionaries
    scores_combined = {f'Sim1_{score}': scores_dict_sim1[score] for score in score_types}
    scores_combined.update({f'Sim2_{score}': scores_dict_sim2[score] for score in score_types})

    # Create DataFrames for scores and stats
    scores_df = pd.DataFrame(scores_combined)
    stats_df = pd.DataFrame([stats_results])

    # Save scores and stats to CSV
    scores_df.to_csv(output_csv, index=False)
    stats_df.to_csv(output_csv.replace('.csv', '_stats.csv'), index=False)

    return scores_df, stats_df



def compute_dynamic_coherence_matrix_scores(output_csv, *coherence_matrices):
    """
    DYNAMIC COMPARISON
    Compute scores for coherence matrices across different frequencies and save them in a CSV file.
    
    Parameters:
        output_csv (str): Path to save the output CSV file containing scores.
        *coherence_matrices: 4D coherence matrices of shape (nsubjects, nregions, nregions, nfreq).
                             Expected order: (simulated_1, simulated_2, ..., empirical)
    
    Returns:
        A CSV file with subjects and scores for each frequency.
    """
    # Unpack the coherence matrices (simulations, then empirical data)
    *simulations, empirical_data = coherence_matrices

    # Get number of subjects and frequencies
    nsubjects = empirical_data.shape[0]
    nfreq = empirical_data.shape[3]  # Assumes 4D input: nsubjects x nregions x nregions x nfreq

    scores_list = []

    # Iterate over subjects and frequency bands
    for subj in range(nsubjects):
        for freq in range(nfreq):
            subject_scores = {'Subject': subj + 1, 'Frequency': freq + 1}

            # Get empirical matrix for the current subject and frequency
            emp_mat = empirical_data[subj, :, :, freq]

            # Iterate over the simulations (Sim1, Sim2)
            for i, sim_mat in enumerate(simulations):
                # Get simulation matrix for the current subject and frequency
                sim_mat = sim_mat[subj, :, :, freq]

                # Compute Mean Absolute Error (MAE)
                mae = mean_absolute_error(sim_mat.flatten(), emp_mat.flatten())
                subject_scores[f'Simulation_{i+1}_MAE'] = mae

                # Compute Frobenius Norm
                frobenius_norm = np.linalg.norm(sim_mat - emp_mat, ord='fro')
                subject_scores[f'Simulation_{i+1}_Frobenius'] = frobenius_norm

                # Compute Cosine Similarity
                cosine_sim = cosine_similarity(sim_mat.flatten().reshape(1, -1), emp_mat.flatten().reshape(1, -1))[0][0]
                subject_scores[f'Simulation_{i+1}_Cosine_Similarity'] = cosine_sim

                # Compute Pearson Correlation
                pearson_corr, _ = pearsonr(sim_mat.flatten(), emp_mat.flatten())
                subject_scores[f'Simulation_{i+1}_Pearson'] = pearson_corr

                # Compute Spearman Correlation
                spearman_corr, _ = spearmanr(sim_mat.flatten(), emp_mat.flatten())
                subject_scores[f'Simulation_{i+1}_Spearman'] = spearman_corr

                # Compute KL Divergence (between the probability distributions of the matrices)
                sim_flat_norm = sim_mat.flatten() / np.sum(sim_mat.flatten())
                emp_flat_norm = emp_mat.flatten() / np.sum(emp_mat.flatten())
                kl_divergence = entropy(sim_flat_norm, emp_flat_norm)
                subject_scores[f'Simulation_{i+1}_KL_Divergence'] = kl_divergence

            # Append the results for this subject and frequency
            scores_list.append(subject_scores)

    # Create a DataFrame from the results and save it to CSV
    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(output_csv, index=False)

    return scores_df



def compute_average_coherence_matrix_score(output_csv, *coherence_matrices):
    """
    Compute various comparison scores (Pearson, Spearman, Frobenius norm, Cosine similarity, MAE) 
    between averaged coherence matrices (averaged across subjects) of simulations and empirical data.

    Parameters:
        output_csv (str): Path to save the averaged coherence matrix scores CSV file.
        *coherence_matrices: Variable number of coherence matrices.
                             Expected order: (simulated_1, simulated_2, ..., empirical)
                             All matrices should have the shape (nsubjects, nregions, nregions).
    
    Returns:
        A CSV file with scores comparing averaged coherence matrices.
    """
    # Unpack the coherence matrices
    *simulations, empirical_data = coherence_matrices

    # Average the coherence matrices across subjects
    avg_empirical = np.mean(empirical_data, axis=0)
    
    # Prepare a dictionary to hold the scores
    scores_dict = {}

    for i, sim_mat in enumerate(simulations):
        # Average the simulated coherence matrix across subjects
        avg_sim = np.mean(sim_mat, axis=0)

        # Flatten the matrices for correlation and similarity calculations
        avg_emp_flat = avg_empirical.flatten()
        avg_sim_flat = avg_sim.flatten()

        # --------- Pearson Correlation ---------
        pearson_corr, _ = pearsonr(avg_emp_flat, avg_sim_flat)
        scores_dict[f'Simulation_{i+1}_Pearson'] = pearson_corr

        # --------- Spearman Correlation ---------
        spearman_corr, _ = spearmanr(avg_emp_flat, avg_sim_flat)
        scores_dict[f'Simulation_{i+1}_Spearman'] = spearman_corr

        # --------- Frobenius Norm (Euclidean Distance) ---------
        # Compute the Frobenius norm on the matrices directly
        frobenius_norm = norm(avg_empirical - avg_sim, 'fro')
        scores_dict[f'Simulation_{i+1}_Frobenius_Norm'] = frobenius_norm

        # --------- Cosine Similarity ---------
        # Reshape flattened vectors to 2D for cosine similarity
        cos_sim = cosine_similarity(avg_emp_flat.reshape(1, -1), avg_sim_flat.reshape(1, -1))[0][0]
        scores_dict[f'Simulation_{i+1}_Cosine_Similarity'] = cos_sim

        # --------- Mean Absolute Error (MAE) ---------
        mae = mean_absolute_error(avg_emp_flat, avg_sim_flat)
        scores_dict[f'Simulation_{i+1}_MAE'] = mae

    # Create a DataFrame from the scores dictionary
    scores_df = pd.DataFrame(scores_dict, index=['Score']).T

    # Save the DataFrame to CSV
    scores_df.to_csv(output_csv, index=True)

    return scores_df


def compute_dynamic_average_coherence_matrix_scores(output_csv, *coherence_matrices):
    """
    Compute scores for averaged coherence matrices across different frequencies and save them in a CSV file.
    
    Parameters:
        output_csv (str): Path to save the output CSV file containing scores.
        *coherence_matrices: 4D coherence matrices of shape (nsubjects, nregions, nregions, nfreq).
                             Expected order: (simulated_1, simulated_2, ..., empirical)
    
    Returns:
        A CSV file with scores for each frequency band.
    """
    # Unpack the coherence matrices (simulations, then empirical data)
    *simulations, empirical_data = coherence_matrices

    # Get the number of regions and frequencies
    nregions = empirical_data.shape[1]
    nfreq = empirical_data.shape[3]  # Assumes 4D input: nsubjects x nregions x nregions x nfreq

    # Average the coherence matrices across subjects for each frequency
    avg_empirical_data = np.mean(empirical_data, axis=0)  # Average across subjects: nregions x nregions x nfreq
    avg_simulations = [np.mean(sim, axis=0) for sim in simulations]  # List of averaged simulated matrices

    # Prepare a list to store the scores
    scores_list = []

    # Iterate over frequency bands
    for freq in range(nfreq):
        freq_scores = {'Frequency': freq + 1}

        # Get the averaged empirical matrix for the current frequency
        avg_emp_mat = avg_empirical_data[:, :, freq]

        # Iterate over the simulations (Sim1, Sim2)
        for i, avg_sim_mat in enumerate(avg_simulations):
            # Get the averaged simulation matrix for the current frequency
            avg_sim_mat = avg_sim_mat[:, :, freq]

            # Compute Mean Absolute Error (MAE)
            mae = mean_absolute_error(avg_sim_mat.flatten(), avg_emp_mat.flatten())
            freq_scores[f'Simulation_{i+1}_MAE'] = mae

            # Compute Frobenius Norm
            frobenius_norm = np.linalg.norm(avg_sim_mat - avg_emp_mat, ord='fro')
            freq_scores[f'Simulation_{i+1}_Frobenius'] = frobenius_norm

            # Compute Cosine Similarity
            cosine_sim = cosine_similarity(avg_sim_mat.flatten().reshape(1, -1), avg_emp_mat.flatten().reshape(1, -1))[0][0]
            freq_scores[f'Simulation_{i+1}_Cosine_Similarity'] = cosine_sim

            # Compute Pearson Correlation
            pearson_corr, _ = pearsonr(avg_sim_mat.flatten(), avg_emp_mat.flatten())
            freq_scores[f'Simulation_{i+1}_Pearson'] = pearson_corr

            # Compute Spearman Correlation
            spearman_corr, _ = spearmanr(avg_sim_mat.flatten(), avg_emp_mat.flatten())
            freq_scores[f'Simulation_{i+1}_Spearman'] = spearman_corr

            # Compute KL Divergence (between the probability distributions of the matrices)
            sim_flat_norm = avg_sim_mat.flatten() / np.sum(avg_sim_mat.flatten())
            emp_flat_norm = avg_emp_mat.flatten() / np.sum(avg_emp_mat.flatten())
            kl_divergence = entropy(sim_flat_norm, emp_flat_norm)
            freq_scores[f'Simulation_{i+1}_KL_Divergence'] = kl_divergence

        # Append the results for this frequency band
        scores_list.append(freq_scores)

    # Create a DataFrame from the results and save it to CSV
    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(output_csv, index=False)

    return scores_df


def compute_subject_specific_dynamic_coherence_matrix_scores_stats(output_csv, subject_ids, *coherence_matrices):
    """
    Compute subject-specific scores for coherence matrices across different frequencies and save them in a CSV file.
    Perform paired T-tests for each subject (Simulation vs Empirical) and save the results for each subject.
    
    Parameters:
        output_csv (str): Path to save the output CSV file containing scores.
        subject_ids (list): List of subject IDs corresponding to the matrices.
        *coherence_matrices: 4D coherence matrices of shape (nsubjects, nregions, nregions, nfreq).
                             Expected order: (simulated_1, simulated_2, ..., empirical)
    
    Returns:
        A CSV file with scores for each subject and frequency band and another CSV for the T-test results.
    """
    # Unpack the coherence matrices (simulations, then empirical data)
    *simulations, empirical_data = coherence_matrices

    # Get the number of subjects, regions, and frequencies
    nsubjects = empirical_data.shape[0]
    nregions = empirical_data.shape[1]
    nfreq = empirical_data.shape[3]  # Assumes 4D input: nsubjects x nregions x nregions x nfreq

    # Prepare a list to store the scores for all subjects and frequencies
    all_scores = []
    
    # Containers for the t-test results (subject-specific) between each simulation and empirical
    ttest_results_list = []

    # Iterate over each subject
    for subject_idx in range(nsubjects):
        subject_id = subject_ids[subject_idx]

        # Prepare lists to hold the scores for the paired t-tests for each simulation and the empirical data
        sim_scores = {i: [] for i in range(len(simulations))}
        emp_scores = []

        # Iterate over each frequency
        for freq in range(nfreq):
            freq_scores = {'Subject_ID': subject_id, 'Frequency': freq + 1}

            # Get the empirical matrix for this subject and frequency
            emp_mat = empirical_data[subject_idx, :, :, freq]
            emp_mat = np.triu(emp_mat)

            # Store empirical data scores (we'll use these for paired t-tests with each simulation)
            emp_flat = emp_mat.flatten()

            # Iterate over the simulations (Sim1, Sim2, etc.)
            for i, sim in enumerate(simulations):
                # Get the simulation matrix for this subject and frequency
                sim_mat = sim[subject_idx, :, :, freq]

                sim_mat = np.triu(sim_mat)

                # Compute Mean Absolute Error (MAE)
                mae = mean_absolute_error(sim_mat.flatten(), emp_flat)
                freq_scores[f'Simulation_{i+1}_MAE'] = mae

                # Compute Frobenius Norm
                frobenius_norm = np.linalg.norm(sim_mat - emp_mat, ord='fro')
                freq_scores[f'Simulation_{i+1}_Frobenius'] = frobenius_norm

                # Compute Cosine Similarity
                cosine_sim = cosine_similarity(sim_mat.flatten().reshape(1, -1), emp_flat.reshape(1, -1))[0][0]
                freq_scores[f'Simulation_{i+1}_Cosine_Similarity'] = cosine_sim

                # Compute Pearson Correlation
                pearson_corr, _ = pearsonr(sim_mat.flatten(), emp_flat)
                freq_scores[f'Simulation_{i+1}_Pearson'] = pearson_corr

                # Compute Spearman Correlation
                spearman_corr, _ = spearmanr(sim_mat.flatten(), emp_flat)
                freq_scores[f'Simulation_{i+1}_Spearman'] = spearman_corr

                # Compute KL Divergence
                sim_flat_norm = sim_mat.flatten() / np.sum(sim_mat.flatten())
                emp_flat_norm = emp_flat / np.sum(emp_flat)
                kl_divergence = entropy(sim_flat_norm, emp_flat_norm)
                freq_scores[f'Simulation_{i+1}_KL_Divergence'] = kl_divergence

                # Append simulation scores for later t-test
                sim_scores[i].append([mae, frobenius_norm, cosine_sim, pearson_corr, spearman_corr, kl_divergence])

            # Append empirical scores for later t-test
            emp_scores.append([mae, frobenius_norm, cosine_sim, pearson_corr, spearman_corr, kl_divergence])

            # Append the scores for this subject and frequency to the list
            all_scores.append(freq_scores)

        # Now perform paired t-tests between the empirical and each simulation for this subject
        ttest_results = {'Subject_ID': subject_id}
        score_names = ['MAE', 'Frobenius', 'Cosine_Similarity', 'Pearson', 'Spearman', 'KL_Divergence']

        # Iterate over each simulation to compare with empirical data
        for i, sim in enumerate(simulations):
            sim_scores_np = np.array(sim_scores[i])
            emp_scores_np = np.array(emp_scores)

            # Perform t-tests for each score type (MAE, Frobenius, etc.)
            for j, score_name in enumerate(score_names):
                t_stat, p_value = ttest_rel(sim_scores_np[:, j], emp_scores_np[:, j])
                ttest_results[f'Simulation_{i+1}_{score_name}_tstat'] = t_stat
                ttest_results[f'Simulation_{i+1}_{score_name}_pvalue'] = p_value

        # Store the t-test results for this subject
        ttest_results_list.append(ttest_results)

    # Create a DataFrame from the scores and save it to CSV
    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(output_csv, index=False)

    # Create a DataFrame from the t-test results and save it to a CSV
    ttest_df = pd.DataFrame(ttest_results_list)
    ttest_df.to_csv(output_csv.replace('.csv', '_ttest.csv'), index=False)

    return scores_df, ttest_df


def compute_dynamic_average_coherence_matrix_scores_stats(output_csv, *coherence_matrices):
    """
    Compute scores for averaged coherence matrices across different frequencies and save them in a CSV file.
    Perform paired T-tests between Simulation 1 and Simulation 2 across frequencies for each score.
    
    Parameters:
        output_csv (str): Path to save the output CSV file containing scores.
        *coherence_matrices: 4D coherence matrices of shape (nsubjects, nregions, nregions, nfreq).
                             Expected order: (simulated_1, simulated_2, ..., empirical)
    
    Returns:
        A CSV file with scores for each frequency band and a separate CSV for T-test results.
    """
    # Unpack the coherence matrices (simulations, then empirical data)
    *simulations, empirical_data = coherence_matrices

    # Get the number of regions and frequencies
    nregions = empirical_data.shape[1]
    nfreq = empirical_data.shape[3]  # Assumes 4D input: nsubjects x nregions x nregions x nfreq

    # Average the coherence matrices across subjects for each frequency
    avg_empirical_data = np.mean(empirical_data, axis=0)  # Average across subjects: nregions x nregions x nfreq
    avg_simulations = [np.mean(sim, axis=0) for sim in simulations]  # List of averaged simulated matrices

    # Prepare a list to store the scores
    scores_list = []
    sim1_scores = []
    sim2_scores = []

    # Iterate over frequency bands
    for freq in range(nfreq):
        freq_scores = {'Frequency': freq + 1}

        # Get the averaged empirical matrix for the current frequency
        avg_emp_mat = avg_empirical_data[:, :, freq]

        avg_emp_mat = np.triu(avg_emp_mat)

        # Iterate over the simulations (Sim1, Sim2)
        for i, avg_sim_mat in enumerate(avg_simulations):
            # Get the averaged simulation matrix for the current frequency
            avg_sim_mat = avg_sim_mat[:, :, freq]

            avg_sim_mat = np.triu(avg_sim_mat)

            # Compute Mean Absolute Error (MAE)
            mae = mean_absolute_error(avg_sim_mat.flatten(), avg_emp_mat.flatten())
            freq_scores[f'Simulation_{i+1}_MAE'] = mae

            # Compute Frobenius Norm
            frobenius_norm = np.linalg.norm(avg_sim_mat - avg_emp_mat, ord='fro')
            freq_scores[f'Simulation_{i+1}_Frobenius'] = frobenius_norm

            # Compute Cosine Similarity
            cosine_sim = cosine_similarity(avg_sim_mat.flatten().reshape(1, -1), avg_emp_mat.flatten().reshape(1, -1))[0][0]
            freq_scores[f'Simulation_{i+1}_Cosine_Similarity'] = cosine_sim

            # Compute Pearson Correlation
            pearson_corr, _ = pearsonr(avg_sim_mat.flatten(), avg_emp_mat.flatten())
            freq_scores[f'Simulation_{i+1}_Pearson'] = pearson_corr

            # Compute Spearman Correlation
            spearman_corr, _ = spearmanr(avg_sim_mat.flatten(), avg_emp_mat.flatten())
            freq_scores[f'Simulation_{i+1}_Spearman'] = spearman_corr

            # Compute KL Divergence (between the probability distributions of the matrices)
            sim_flat_norm = avg_sim_mat.flatten() / np.sum(avg_sim_mat.flatten())
            emp_flat_norm = avg_emp_mat.flatten() / np.sum(avg_emp_mat.flatten())
            kl_divergence = entropy(sim_flat_norm, emp_flat_norm)
            freq_scores[f'Simulation_{i+1}_KL_Divergence'] = kl_divergence

        # Store the scores in separate lists for Sim1 and Sim2 for the paired T-test
        sim1_scores.append([freq_scores['Simulation_1_MAE'], freq_scores['Simulation_1_Frobenius'], 
                            freq_scores['Simulation_1_Cosine_Similarity'], freq_scores['Simulation_1_Pearson'], 
                            freq_scores['Simulation_1_Spearman'], freq_scores['Simulation_1_KL_Divergence']])
        
        sim2_scores.append([freq_scores['Simulation_2_MAE'], freq_scores['Simulation_2_Frobenius'], 
                            freq_scores['Simulation_2_Cosine_Similarity'], freq_scores['Simulation_2_Pearson'], 
                            freq_scores['Simulation_2_Spearman'], freq_scores['Simulation_2_KL_Divergence']])

        # Append the results for this frequency band
        scores_list.append(freq_scores)

    # Create a DataFrame from the results and save it to CSV
    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(output_csv, index=False)

    # Perform paired T-tests between Sim1 and Sim2 scores across frequencies
    sim1_scores = np.array(sim1_scores)
    sim2_scores = np.array(sim2_scores)

    ttest_results = {}
    score_names = ['MAE', 'Frobenius', 'Cosine_Similarity', 'Pearson', 'Spearman', 'KL_Divergence']

    for i, score_name in enumerate(score_names):
        t_stat, p_value = ttest_rel(sim1_scores[:, i], sim2_scores[:, i])
        ttest_results[f'Ttest_{score_name}_tstat'] = t_stat
        ttest_results[f'Ttest_{score_name}_pvalue'] = p_value

    # Save T-test results to a new CSV
    ttest_df = pd.DataFrame([ttest_results])
    ttest_df.to_csv(output_csv.replace('.csv', '_ttest.csv'), index=False)

    return scores_df, ttest_df


def save_region_mae_to_txt(mae_mf_regions_avg, mae_ww_regions_avg, mae_wc_regions_avg, mae_go_regions_avg, outdir):
    """
    Save the region MAE data for MF, WW, WC, and GO in separate txt files.
    Each file will contain the region MAE data in a column format.
    """
    
    # Define the file names
    filenames = ['mf_region_mae.txt', 'ww_region_mae.txt', 'wc_region_mae.txt', 'go_region_mae.txt']
    arrays = [mae_mf_regions_avg, mae_ww_regions_avg, mae_wc_regions_avg, mae_go_regions_avg]
    
    for filename, array in zip(filenames, arrays):
        # Save each array in column format
        np.savetxt(outdir + filename, np.array(array), fmt='%.6f')  # Save with 6 decimal precision
        print(f'Saved {filename} successfully!')


def save_region_mae_to_txt_wb(mae_mf_regions_avg, mae_ww_regions_avg, outdir):
    """
    Save the region MAE data for MF, WW in separate txt files.
    Each file will contain the region MAE data in a column format.
    """
    
    # Define the file names
    filenames = ['wb_mf_region_mae.txt', 'wb_ww_region_mae.txt']
    arrays = [mae_mf_regions_avg, mae_ww_regions_avg]
    
    for filename, array in zip(filenames, arrays):
        # Save each array in column format
        np.savetxt(outdir + filename, np.array(array), fmt='%.6f')  # Save with 6 decimal precision
        print(f'Saved {filename} successfully!')


# Create a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Apply bandpass filter to the data
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y


# Compute coherence between two signals
def compute_coherence_matrix(data1, data2, fs):
    n_regions = data1.shape[0]
    coherence_matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            f, Cxy = coherence(data1[i, :], data2[j, :], fs=fs, nperseg = 512)
            coherence_matrix[i, j] = np.mean(Cxy)
    return coherence_matrix


# Similarity metrics
def compute_similarity(empirical, simulated):
    pcc = np.corrcoef(empirical.flatten(), simulated.flatten())[0, 1]
    cosine_sim = np.dot(empirical.flatten(), simulated.flatten()) / (
        np.linalg.norm(empirical.flatten()) * np.linalg.norm(simulated.flatten())
    )
    kl_divergence = np.sum(empirical * np.log(empirical / simulated + 1e-10)) # Add small constant to avoid log(0)
    mae = mean_absolute_error(empirical, simulated)
    return pcc, cosine_sim, kl_divergence,mae








