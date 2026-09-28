import json
import numpy as np
import brian2 as b2

implemented_neuron_models = ['DCNi_cell', 'DCNp_cell',
                             'PC_cell', 'GrC_cell', 'GoC_cell',
                             'MLI_cell', 'stellate_cell', 'basket_cell', 'GrC_cell_PLOS',
                             'GoC_cell_PLOS', 'MLI_cell_PLOS', 'PC_cell_PLOS']

# Defining the E-GLIF model #
EGLIF_eqs = '''
dV/dt = (gl*(V - El) - Ia + Id + Ie + Is + gE*(Ee - V) + gI*(Ei - V)) / Cm : volt (unless refractory)
dIa/dt = kadap*(V - El) - k2*Ia : amp
dId/dt = -k1*Id : amp
dgE/dt = xE - gE/Te : siemens
dxE/dt = -xE/Te : siemens/second
dgI/dt = xI - gI/Ti : siemens
dxI/dt = -xI/Ti : siemens/second
Is = external_current(t) : amp
Ie : amp
gl : siemens
Cm : farad
El : volt
Vmin: volt
kadap : 1/henry
k2 : 1/second
k1 : 1/second
Ee : volt
Ei : volt
Te : second
Ti : second
'''


def sinusoidal_input(time, amplitude=1.0, length=None, period=None, offset=0.0, phase=0.0):
    """Create a sinusoidal input signal over a time array."""
    time = np.asarray(time, dtype=float)

    if length is None:
        length = time[-1] - time[0] if time.size > 1 else 1.0
    if period is None:
        period = length if length > 0 else 1.0

    if length <= 0:
        raise ValueError("length must be positive")
    if period <= 0:
        raise ValueError("period must be positive")

    signal = np.zeros_like(time, dtype=float)
    mask = (time >= 0) & (time <= length)
    signal[mask] = offset + amplitude * np.sin(2 * np.pi * time[mask] / period + phase)
    return signal

#-------------------------------------------------------------------
def setting_EGLIF_simulation_Brian_membrane_fluc(idx = None, N_cell = None, neuron_model = None, json_file_name = None, curr_inj = None):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Please, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Please, specify the json_file_name containing the model parameters")
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    print(f'Imported data: {json_file_name}')
    
    if neuron_model in implemented_neuron_models:
        print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_th']
        V_reset_value = data[0][idx]['model']['V_reset']
        lambda_0_value = data[0][idx]['model']['lambda_0']
        tau_V_value = data[0][idx]['model']['tau_V']
        t_ref_value = data[0][idx]['model']['t_ref']
        A2_value = data[0][idx]['model']['A_2']
        A1_value = data[0][idx]['model']['A_1']


        G_neuron = b2.NeuronGroup(N_cell, model=EGLIF_eqs,
                     threshold=f'rand() < dt * {lambda_0_value} / ms * exp((V - {V_th_value} * mV) / ({tau_V_value} * mV))',
                     reset = f'''
                     V = {V_reset_value} * mV
                     Ia += {A2_value} * pA
                     Id = {A1_value} * pA
                     ''',
                     refractory=f'{t_ref_value} * ms',
                     #method='euler')
            		 method = 'rk4')
		

        #init variables:
        #G_neuron.V = data[0][idx]['init']['V']*b2.mV
        
        #Randomly initialize membrane potential
        EL = data[0][idx]['model']['E_L']
        Vreset = data[0][idx]['model']['V_reset']
        Vth = data[0][idx]['model']['V_th']

        lower = EL + (Vreset - EL)
        upper = EL + (Vth - EL)/2

        G_neuron.V = (lower + (upper - lower) * b2.rand(N_cell)) * b2.mV
        
        G_neuron.Ia = data[0][idx]['init']['I_a']*b2.nA
        G_neuron.Id = data[0][idx]['init']['I_d']*b2.nA
        
        G_neuron.gE = 0*b2.nS
        G_neuron.gI = 0*b2.nS
        G_neuron.xE = 0*b2.nS/b2.ms
        G_neuron.xI = 0*b2.nS/b2.ms

        #parameter values:
        G_neuron.Cm = data[0][idx]['model']['C_m'] * b2.pF
        G_neuron.gl = data[0][idx]['model']['g_L'] * b2.nS
        G_neuron.El = data[0][idx]['model']['E_L'] * b2.mV
        G_neuron.kadap = data[0][idx]['model']['k_a'] * (b2.amp / (b2.volt * b2.second))
        G_neuron.k1 = data[0][idx]['model']['k_1'] / b2.ms
        G_neuron.k2 = data[0][idx]['model']['k_2'] / b2.ms
        G_neuron.Ie = data[0][idx]['model']['I_e'] * b2.nA

		       
        G_neuron.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_neuron.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_neuron.Te = data[0][idx]['model']['T_e'] * b2.ms
        G_neuron.Ti = data[0][idx]['model']['T_i'] * b2.ms

        if 'V_min' in data[0][idx]['model'].keys():
            G_neuron.Vmin = data[0][idx]['model']['V_min'] * b2.mV

        return G_neuron, data

#-------------------------------------------------------------------
def setting_EGLIF_simulation_Brian(idx = None, N_cell = None, neuron_model = None, json_file_name = None, curr_inj = None):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Please, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Please, specify the json_file_name containing the model parameters")
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    print(f'Imported data: {json_file_name}')
    
    if neuron_model in implemented_neuron_models:
        print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_th']
        V_reset_value = data[0][idx]['model']['V_reset']
        lambda_0_value = data[0][idx]['model']['lambda_0']
        tau_V_value = data[0][idx]['model']['tau_V']
        t_ref_value = data[0][idx]['model']['t_ref']
        A2_value = data[0][idx]['model']['A_2']
        A1_value = data[0][idx]['model']['A_1']


        G_neuron = b2.NeuronGroup(N_cell, model=EGLIF_eqs,
                     threshold=f'rand() < dt * {lambda_0_value} / ms * exp((V - {V_th_value} * mV) / ({tau_V_value} * mV))',
                     reset = f'''
                     V = {V_reset_value} * mV
                     Ia += {A2_value} * pA
                     Id = {A1_value} * pA
                     ''',
                     refractory=f'{t_ref_value} * ms',
                     #method='euler')
            		 method = 'rk4')
		

        #init variables:
        G_neuron.V = data[0][idx]['init']['V']*b2.mV
        G_neuron.Ia = data[0][idx]['init']['I_a']*b2.nA
        G_neuron.Id = data[0][idx]['init']['I_d']*b2.nA
        
        G_neuron.gE = 0*b2.nS
        G_neuron.gI = 0*b2.nS
        G_neuron.xE = 0*b2.nS/b2.ms
        G_neuron.xI = 0*b2.nS/b2.ms

        #parameter values:
        G_neuron.Cm = data[0][idx]['model']['C_m'] * b2.pF
        G_neuron.gl = data[0][idx]['model']['g_L'] * b2.nS
        G_neuron.El = data[0][idx]['model']['E_L'] * b2.mV
        G_neuron.kadap = data[0][idx]['model']['k_a'] * (b2.amp / (b2.volt * b2.second))
        G_neuron.k1 = data[0][idx]['model']['k_1'] / b2.ms
        G_neuron.k2 = data[0][idx]['model']['k_2'] / b2.ms
        G_neuron.Ie = data[0][idx]['model']['I_e'] * b2.nA

        G_neuron.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_neuron.Ei = data[0][idx]['model']['E_i'] * b2.mV        
        G_neuron.Te = data[0][idx]['model']['T_e'] * b2.ms
        G_neuron.Ti = data[0][idx]['model']['T_i'] * b2.ms

        if 'V_min' in data[0][idx]['model'].keys():
            G_neuron.Vmin = data[0][idx]['model']['V_min'] * b2.mV

        return G_neuron, data

#-------------------------------------------------------------------
def setting_EGLIF_simulation_Brian_Vmin(idx = None, N_cell = None, neuron_model = None, json_file_name = None, curr_inj = None):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Please, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Please, specify the json_file_name containing the model parameters")
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    print(f'Imported data: {json_file_name}')

    if neuron_model in implemented_neuron_models:
        print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_th']
        V_reset_value = data[0][idx]['model']['V_reset']
        lambda_0_value = data[0][idx]['model']['lambda_0']
        tau_V_value = data[0][idx]['model']['tau_V']
        t_ref_value = data[0][idx]['model']['t_ref']
        A2_value = data[0][idx]['model']['A_2']
        A1_value = data[0][idx]['model']['A_1']


        G_neuron = b2.NeuronGroup(
            N_cell,
            model=EGLIF_eqs,
            threshold=f'rand() < dt * {lambda_0_value} / ms * exp((V - {V_th_value} * mV) / ({tau_V_value} * mV))',
            reset=f'''
                V = {V_reset_value} * mV
                Ia += {A2_value} * pA
                Id = {A1_value} * pA
            ''',
            refractory=f'{t_ref_value} * ms',
            method='rk4',
            events={'vmin_event': 'V < Vmin'}  # define extra event
        )

        #init variables:
        G_neuron.V = data[0][idx]['init']['V']*b2.mV
        G_neuron.Ia = data[0][idx]['init']['I_a']*b2.nA
        G_neuron.Id = data[0][idx]['init']['I_d']*b2.nA
        G_neuron.gE = 0*b2.nS
        G_neuron.gI = 0*b2.nS
        G_neuron.xE = 0*b2.nS/b2.ms
        G_neuron.xI = 0*b2.nS/b2.ms

        #parameter values:
        G_neuron.Cm = data[0][idx]['model']['C_m'] * b2.pF
        G_neuron.gl = data[0][idx]['model']['g_L'] * b2.nS
        G_neuron.El = data[0][idx]['model']['E_L'] * b2.mV
        G_neuron.kadap = data[0][idx]['model']['k_a'] * (b2.amp / (b2.volt * b2.second))
        G_neuron.k1 = data[0][idx]['model']['k_1'] / b2.ms
        G_neuron.k2 = data[0][idx]['model']['k_2'] / b2.ms
        G_neuron.Ie = data[0][idx]['model']['I_e'] * b2.nA

		       
        G_neuron.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_neuron.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_neuron.Te = data[0][idx]['model']['T_e'] * b2.ms
        G_neuron.Ti = data[0][idx]['model']['T_i'] * b2.ms

        if 'V_min' in data[0][idx]['model'].keys():
            G_neuron.Vmin = data[0][idx]['model']['V_min'] * b2.mV

        return G_neuron, data


# -------------------------------------------------------------------
def setting_EGLIF_simulation_Brian_Vmin_membrane_fluc(idx=None, N_cell=None, neuron_model=None, json_file_name=None, curr_inj=None):
    if N_cell == None:
        N_cell = 1

    if neuron_model == None:
        raise ValueError("Please, specify the neuron_model you wish to simulate")
    if neuron_model not in implemented_neuron_models:
        raise ValueError(f"neuron_model must be one of {implemented_neuron_models}, but got '{neuron_model}'.")

    if json_file_name == None:
        raise ValueError("Please, specify the json_file_name containing the model parameters")

    with open(json_file_name, 'r') as file:
        data = json.load(file)
    print(f'Imported data: {json_file_name}')

    if neuron_model in implemented_neuron_models:
        print(f'neuron model: {neuron_model}')
        V_th_value = data[0][idx]['model']['V_th']
        V_reset_value = data[0][idx]['model']['V_reset']
        lambda_0_value = data[0][idx]['model']['lambda_0']
        tau_V_value = data[0][idx]['model']['tau_V']
        t_ref_value = data[0][idx]['model']['t_ref']
        A2_value = data[0][idx]['model']['A_2']
        A1_value = data[0][idx]['model']['A_1']

        G_neuron = b2.NeuronGroup(
            N_cell,
            model=EGLIF_eqs,
            threshold=f'rand() < dt * {lambda_0_value} / ms * exp((V - {V_th_value} * mV) / ({tau_V_value} * mV))',
            reset=f'''
                V = {V_reset_value} * mV
                Ia += {A2_value} * pA
                Id = {A1_value} * pA
            ''',
            refractory=f'{t_ref_value} * ms',
            method='rk4',
            events={'vmin_event': 'V < Vmin'}  # define extra event
        )

        # init variables:
        # G_neuron.V = data[0][idx]['init']['V']*b2.mV

        # Randomly initialize membrane potential
        EL = data[0][idx]['model']['E_L']
        Vreset = data[0][idx]['model']['V_reset']
        Vth = data[0][idx]['model']['V_th']

        lower = EL + (Vreset - EL)
        upper = EL + (Vth - EL) / 2

        G_neuron.V = (lower + (upper - lower) * b2.rand(N_cell)) * b2.mV

        G_neuron.Ia = data[0][idx]['init']['I_a'] * b2.nA
        G_neuron.Id = data[0][idx]['init']['I_d'] * b2.nA
        G_neuron.gE = 0 * b2.nS
        G_neuron.gI = 0 * b2.nS
        G_neuron.xE = 0 * b2.nS / b2.ms
        G_neuron.xI = 0 * b2.nS / b2.ms

        # parameter values:
        G_neuron.Cm = data[0][idx]['model']['C_m'] * b2.pF
        G_neuron.gl = data[0][idx]['model']['g_L'] * b2.nS
        G_neuron.El = data[0][idx]['model']['E_L'] * b2.mV
        G_neuron.kadap = data[0][idx]['model']['k_a'] * (b2.amp / (b2.volt * b2.second))
        G_neuron.k1 = data[0][idx]['model']['k_1'] / b2.ms
        G_neuron.k2 = data[0][idx]['model']['k_2'] / b2.ms
        G_neuron.Ie = data[0][idx]['model']['I_e'] * b2.nA

        G_neuron.Ee = data[0][idx]['model']['E_e'] * b2.mV
        G_neuron.Ei = data[0][idx]['model']['E_i'] * b2.mV
        G_neuron.Te = data[0][idx]['model']['T_e'] * b2.ms
        G_neuron.Ti = data[0][idx]['model']['T_i'] * b2.ms

        if 'V_min' in data[0][idx]['model'].keys():
            G_neuron.Vmin = data[0][idx]['model']['V_min'] * b2.mV

        return G_neuron, data
    
#-------------------------------------------------------------------    
def get_params_SI(p):
    params_SI={}
    params_SI['C_m'] = p['C_m'] * 10**-12
    params_SI['g_L'] = p['g_L'] * 10**-9
    params_SI['E_L'] = p['E_L'] * 10**-3
    params_SI['V_th'] = p['V_th'] * 10**-3
    params_SI['V_reset'] = p['V_reset'] * 10**-3
    params_SI['V_spike'] = p['V_spike'] * 10**-3
    params_SI['delta_V'] = p['delta_V'] * 10**-3
    params_SI['tau_V'] = p['tau_V'] * 10**-3
    params_SI['lambda_0'] = p['lambda_0']
    params_SI['t_ref'] = p['t_ref'] * 10**-3
    params_SI['k_a'] = p['k_a']
    params_SI['k_2'] = p['k_2'] * 10**3
    params_SI['k_1'] = p['k_1'] * 10**3
    params_SI['A_2'] = p['A_2'] * 10**-12
    params_SI['A_1'] = p['A_1'] * 10**-12
    params_SI['I_e'] = p['I_e'] * 10**-9
    params_SI['E_e'] = p['E_e'] * 10**-3
    params_SI['K_e'] = p['K_e']
    params_SI['T_e'] = p['T_e'] * 10**-3
    params_SI['Q_e'] = p['Q_e'] * 10**-9
    params_SI['E_i'] = p['E_i'] * 10**-3
    params_SI['K_i'] = p['K_i']
    params_SI['T_i'] = p['T_i'] * 10**-3
    params_SI['Q_i'] = p['Q_i'] * 10**-9

    if 'Q_e_m' in p.keys():
        params_SI['K_e_m'] = p['K_e_m']
        params_SI['T_e_m'] = p['T_e_m'] * 10**-3
        params_SI['Q_e_m'] = p['Q_e_m'] * 10**-9      
    
    if 'Q_e_i' in p.keys():
        params_SI['K_e_i'] = p['K_e_i']
        params_SI['T_e_i'] = p['T_e_i'] * 10**-3
        params_SI['Q_e_i'] = p['Q_e_i'] * 10**-9        
    
    return params_SI
    
#-------------------------------------------------------------------
def get_input_config(idx = None, json_file_name = None):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the input configuration parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)
        net_comp = data[0]["network_composition"]
        return data[0], net_comp

#-------------------------------------------------------------------
def print_connectivity(connectivity_map):
    for neuron, conn in connectivity_map.items():
        print(f"\n{neuron} receives:")
        
        if "inh" in conn:
            print(f"  inhibitory from: {', '.join(conn['inh'])}")
        
        if "exc" in conn:
            print(f"  excitatory from: {', '.join(conn['exc'])}")
        
        if "exc_m" in conn:
            print(f"  and excitatory mossy input")
       

