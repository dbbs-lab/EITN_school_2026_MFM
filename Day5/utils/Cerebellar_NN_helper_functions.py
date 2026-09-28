import numpy as np
import brian2 as b2

def connect_population(pre, post, K, Q, T, delay, var_name, name=None):

    N_pre = len(pre)
    p = K / N_pre
    #p=1 # testing all to all connections
    
    w = (Q * np.e / T) * b2.nS / b2.ms

    S = b2.Synapses(pre, post, on_pre=f'{var_name} += w', namespace={'w':w}, name=name)
    S.connect(p=p)
    S.delay = delay * b2.ms

    # print(f"{S.name}: {pre.name} -> {post.name} | p={p:.4f} | Nsyn={len(S)}")
    # print(f"w: {w}")
    return S

def extract_params(p_model, suffix):
    return (
        p_model[f'K_{suffix}'],
        p_model[f'Q_{suffix}'],
        p_model[f'T_{suffix}'],
        p_model[f'delay_{suffix}']
    )

def build_connections(idx, params_dict, populations, connectivity_map):
    synapses = []

    for pop_name, params in params_dict.items():
        post = populations[pop_name]
        p_model = params[0][idx]['model']

        rules = connectivity_map.get(pop_name, {})

        # --- Excitatory ---
        if 'exc' in rules and 'K_e' in p_model:
            K, Q, T, delay = extract_params(p_model, 'e')

            for pre_name in rules['exc']:
                pre = populations[pre_name]

                syn_name = f"S_{pre_name}_{pop_name}_E"
                print(f"Setting connection: {pre_name} -> {pop_name}")

                synapses.append(
                    connect_population(pre, post, K, Q, T, delay, 'xE', syn_name)
                )

        # --- Mossy-specific excitatory ---
        if 'exc_m' in rules and 'K_e_m' in p_model:
            K, Q, T, delay = extract_params(p_model, 'e_m')

            for pre_name in rules['exc_m']:
                pre = populations[pre_name]

                syn_name = f"S_{pre_name}_{pop_name}_Em"
                print(f"Setting connection: {pre_name} -> {pop_name}")

                synapses.append(
                    connect_population(pre, post, K, Q, T, delay, 'xE', syn_name)
                )

        # --- Inhibitory ---
        if 'inh' in rules and 'K_i' in p_model:
            K, Q, T, delay = extract_params(p_model, 'i')

            for pre_name in rules['inh']:
                pre = populations[pre_name]

                syn_name = f"S_{pre_name}_{pop_name}_I"
                print(f"Setting connection: {pre_name} -> {pop_name}")

                synapses.append(
                    connect_population(pre, post, K, Q, T, delay, 'xI', syn_name)
                )

    return synapses
