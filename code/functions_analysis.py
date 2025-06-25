##### FUNCTIONS FOR ANALYSIS OF THE MACAQUE RIPPLE BAND #####

import os
import numpy as np
import pandas as pd
import elephant
from scipy.signal import hilbert
import neo
from scipy.stats import zscore
import quantities as pq

########## DATA LOADING AND HANDLING ##########


def load_block(monkey,array,type_rec,type_sig,date,data_folder=''):
    """
    Loading the monkey data (already preprocessed as in the snake files).
    
    type_rec: RS, OG
    type_sig: LFP, RB, tMUA, spikes
    """
    io = neo.NixIO(f'{data_folder}/macaque{monkey}_{type_rec}_{date}/{type_sig}/macaque{monkey}_{type_rec}_{date}_Array{array}_{type_sig}.nix','ro')
    block = io.read_block()
    return block


def sig_block_to_arr(sig_block,result_type):
    """
    Converts the block of nix data to just numpy array with 64 rows and T columns (units are as in nix, usually ms)

    This function works for continuous sig. types (result_type):
    - LFP (signal directly from the filter, non-normalised)
    - LFP_zsc (LFP further zscored per channel)
    - RB_filtered (this is in the units resulting from filtering, non-normalised)
    - RB_filtered_zsc (further applied zscore on every row of the signal)
    - RB_phase
    - RB_envelope_norm (envelope in the units of its STD)
    - RB_envelope_phase (phase of the zscored RB envelope)

    WARNING - FUNCTION IS BASED ON THE INDEXING IN THE SNAKE PROCESSED FILE.

    If we want to return LFP or LFP_zsc, the LFP type of block has to be loaded.
    For other result_types, load RB block in the sig_block.
    """
    if result_type=='LFP':
        LFP_aux = sig_block.segments[0].analogsignals[0]
        arr = np.array(LFP_aux.T.magnitude)
    elif result_type=='LFP_zsc':
        LFP_aux = sig_block.segments[0].analogsignals[0]
        arr = np.array(LFP_aux.T.magnitude)
        arr = zscore(arr,axis=1)
    elif result_type=='RB_filtered':
        RB_aux = sig_block.segments[0].analogsignals[0]
        arr = np.array(RB_aux.T.magnitude)
    elif result_type=='RB_filtered_zsc':
        RB_aux = sig_block.segments[0].analogsignals[0]
        arr = np.array(RB_aux.T.magnitude)
        arr = zscore(arr,axis=1)
    elif result_type=='RB_phase':
        RB_aux = sig_block.segments[0].analogsignals[1]
        arr = np.array(RB_aux.T.magnitude)
    elif result_type=='RB_envelope_norm':
        RB_aux = sig_block.segments[0].analogsignals[3]
        arr = np.array(RB_aux.T.magnitude)
    elif result_type=='RB_envelope_phase':
        RB_aux = sig_block.segments[0].analogsignals[4]
        arr = np.array(RB_aux.T.magnitude)
    else:
        print('Wrong result type entered.')
        return        
    return arr

"""
OLD VERSION 25.06.25
def spike_block_to_arr(spike_block):
    ""
    Takes into nix spike train, and bins the spikes into 1 ms bins.
    Returns np.array with as many rows as cells/channels.

    Input can be both tMUA and SUA.
    ""
    ### create a spike vector, not cutting any spikes out
    start_t_ms = np.float64(spike_block.segments[0].spiketrains[0].t_start.magnitude)*1000
    stop_t_ms = np.float64(spike_block.segments[0].spiketrains[0].t_stop.magnitude)*1000
    dur_rec_ms = np.int64(np.ceil(stop_t_ms - start_t_ms)) ### rounded up to the integer
    num_cells = len(spike_block.segments[0].spiketrains)
    arr = np.zeros([num_cells,dur_rec_ms])
    for cell in range(num_cells):
        spikes_out = 0
        sp_all = np.zeros(dur_rec_ms)
        spike_train = spike_block.segments[0].spiketrains[cell]
        for spike_time in spike_train:
            spike_time = spike_time.magnitude*1000
            spike_time_rel = spike_time - start_t_ms
            try:
                sp_all[int(np.floor(spike_time_rel))]+=1
            except:
                spikes_out+=1
        if spikes_out>0:
            print(f'Spikes out, cell idx. {cell}: {spikes_out}.')
        arr[cell,:] = sp_all
    return arr
"""

def spike_block_to_arr(spike_block,bin_size=1*pq.ms):
    """
    Binning spikes in the block with a given bin.
    Defauls 1 ms, returns np.array
    """
    bst = elephant.conversion.BinnedSpikeTrain(spike_block.segments[0].spiketrains, bin_size=bin_size)
    spike_arr = bst.to_array()
    
    return spike_arr


def cut_abs_times(aux_arr,start_t_arr,monkey,rec_type='RS',date=None,params={}):
    """
    Cuts out only the part of the recording where all of the arrays were on.
    Preserves all rows of array.

    start_t_arr (int) gives the time indice of the first column in abs. time.
    """
    if monkey in ['N','F']: ### this function only makes sense for these monkeys, RS
        start_t_all_arr = params['times_all_arr'][monkey][rec_type][date][0]
        stop_t_all_arr = params['times_all_arr'][monkey][rec_type][date][1]
        duration_all_arr = stop_t_all_arr - start_t_all_arr
        diff_start = start_t_all_arr - start_t_arr ### should be always non-negative
        if diff_start<0:
            print('Wrong start alignment.')
            print(f'Diff.: {diff_start}')
            print(f'Start all.: {start_t_all_arr}')
            print(f'Start arr.: {start_t_arr}')
            return
        aux_arr_cut = aux_arr[:,diff_start:(diff_start+duration_all_arr)]
        return aux_arr_cut
    else:
        return aux_arr ### A and L have data aligned

    
def aux_electrodeID_to_ch_order(monkey,date,electrode_ID,array_ID,data_folder='',type_rec='RS'):
    """
    Returns channel order in 0 to 63 for a given electrode ID on an array. 
    (channel indexing as in tMUA files)
    """
    ### we also open tMUA block, to find the correct row corresponding to ElectrodeID
    tMUA_block = load_block(monkey,array_ID,type_rec,'tMUA',date,data_folder=data_folder)
    ### looking for a correct row
    ch = -1
    for sp_tr_idx in range(64):
        sp_tr = tMUA_block.segments[0].spiketrains[sp_tr_idx]
        el_id = sp_tr.annotations['Electrode_ID']
        if el_id == electrode_ID:
            ch = sp_tr_idx
    return ch


def aux_convert_cols_to_list(df,col_names=[],before_number=None):
    """
    For column in col_names converts each value from string to the list.
    (i.e. in each row of df col. we assume string, that has to be splitted into numbers)

    Before number should be set to '(' in order to extract numbers in brackets or in the form np.float(num).
    """
    for name in col_names:
        vals_converted = [] #### list of lists
        data_col = df[name]
        for idx in data_col.index:
            str_row = data_col.loc[idx][1:-1] ### string not including the brackets
            list_row = str_row.split(',')
            if before_number is None:
                list_row = [float(value) for value in list_row]
            else:
                list_row =  [float(value[(value.find(before_number)+1):-1]) for value in list_row] ### erases the part before the given symbol
            vals_converted.append(np.array(list_row))
        df[name] = vals_converted
    return df


def ensure_dir_exists(dirpath):
    """
    Creates folder on the path if missing.
    """
    if not os.path.isdir(dirpath):
        print('Creating', dirpath)
        os.makedirs(dirpath)
    return


def create_indicator(df,start_col='t_start',stop_col='t_stop',state_col='state',data_col=None,
                     positive_state='Closed_eyes',mult_factor=1000):
    """    
    Creates vector with 1 and 0 for the timestamps in the dataframe.
    Positive state times are 1.
    The time in df can be multiplied by factor to define a column of the indicator.
    i.e. if there are seconds in df, and ms in indicator, factor is 1000.

    If the data_col is not None, the value from the data col will be used instead of the number 1 in the indicator.
    """
    indic_arr = np.zeros(np.int64(df.loc[df.index[-1]][stop_col])*mult_factor)
    for idx in df.index:
        start_t = np.int64(df.loc[idx][start_col]*mult_factor)
        stop_t = np.int64(df.loc[idx][stop_col]*mult_factor)
        if data_col is not None:
            val = df.loc[idx][data_col]
        if df.loc[idx][state_col]==positive_state:
            if data_col is not None:
                if val==0:
                    val=1 #### just proxy for when the orientation of a grating is 0 (or generaly val. in data_col), to make it count in the indicator
                indic_arr[start_t:stop_t] = val
            else:
                indic_arr[start_t:stop_t] = 1
        else:
            pass
    return indic_arr

    
########## MISC. AUX. FUNCTIONS ##########


def moving_window(arr, window_size=1000, step=1000): 
    """
    Returns the list of the cut data from the moving window with the given size, 
    and indices of starts and stops wrt original array.
    
    Works on the rows of the input array.
    """
    if arr.ndim==1:
        data_sliced = [arr[i:i + window_size] for i in np.arange(0,len(arr) - window_size + 1,step)]
        starts = [i for i in np.arange(0,len(arr) - window_size + 1,step)]
        stops = [i + window_size for i in np.arange(0,len(vec) - window_size + 1,step)]
    elif arr.ndim>1:
        data_sliced = [arr[:,i:i + window_size] for i in np.arange(0,arr.shape[1] - window_size + 1,step)]
        starts = [i for i in np.arange(0,arr.shape[1] - window_size + 1,step)]
        stops = [i + window_size for i in np.arange(0,arr.shape[1] - window_size + 1,step)]
    else:
        print('Wrong dimensionality array.')
        return
    return data_sliced, starts, stops


def list_merge(list_of_lists):
    """
    Merging list of lists into one list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def count_zero_intervals(data_vec):
    """
    Count how long are intervals with 0.
    
    data_vec is assumed to be one-dimensional
    """
    intervals = []
    count = 0
    
    for num in data_vec:
        if num == 0:
            count += 1
        else:
            if count > 0:
                intervals.append(count)
                count = 0
    if count > 0:
        intervals.append(count)
    return intervals


def bin_arr(arr, bin_width=1000,step=1000):
    """
    Bin array, works on individual rows.
    Step gives the distance between consecutive bin centers.
    """
    if arr.ndim>1:
        num_rows = arr.shape[0]
        for row in range(num_rows):
            vec = arr[row,:]
            data_sliced = [vec[i:i + bin_width] for i in np.arange(0,len(vec) - bin_width + 1,step)]
            if row==0:
                num_cols = len(data_sliced)
                binned_arr = np.zeros([num_rows,num_cols])
            sums_bins = [np.sum(vec) for vec in data_sliced]
            binned_arr[row,:] = sums_bins ### resulting in the array with same number of rows, and num_col as number of bins
        return binned_arr
    elif arr.ndim==1:
        vec = arr #[np.newaxis,:]
        data_sliced = [vec[i:i + bin_width] for i in np.arange(0,len(vec) - bin_width + 1,step)]
        num_cols = len(data_sliced)
        binned_arr = np.zeros(num_cols)
        sums_bins = [np.sum(vec) for vec in data_sliced]
        binned_arr = np.array(sums_bins) ### resulting in the one dimensional vector length is the number of bins
    else:
        print('Wrong input array.')
        return
    return binned_arr


########## CIRCULAR OPERATIONS ##########


def circular_avg(data_vec,bins=25,x_range=(-np.pi,np.pi)):
    """
    Creates histogram based on the data vec and finds the angle and the size of the normalised circular average.
    In default assumes data from -pi to pi.
    Resulting R is from 0 to 1.
    """
    bin_vals, border_bins = np.histogram(data_vec,bins=bins,range=x_range)
    centre_bins = np.diff(border_bins)/2+border_bins[:-1] ### move from the left edge of the bin to its centre
    try:
        bin_vals = bin_vals/np.sum(bin_vals) ### normalising histogram to density
        r, phi = event_to_vec_weighted(angles=centre_bins,weights=bin_vals)
    except:
        r, phi = np.nan, np.nan
    return r, phi


def event_to_vec_weighted(angles,weights):
    """
    Computes vector sum of vectors (circulat avg. in polar coor.).

    angles: vector of angles
    weights: length of the vectors in each direction

    Returns vector sum in polar coordinates (r, phi).
    No further normalisation is used in this function.
    """
    import cmath

    event = np.vstack((weights, angles)).T # we add the other polar coordinate - the length
    size = event.shape[0] # dimension of the event vector
    vector = np.zeros(size, complex)
    for row in range(size):  # each row is a complex cartesian repr.
        vector[row] = cmath.rect(event[row][0], event[row][1]) # takes argument r, phi
    vector_sum = np.sum(vector[:]) # sum of complex numbers
    vec_sum_polar = cmath.polar(vector_sum)
    
    return vec_sum_polar # (r, phi)


########## SUA DATAFRAME CREATE FUNCTIONS ##########


def spike_train_prop_vec(spike_vector,rb_phase,rb_envelope,rb_env_phase,channel_prop=None,indicator=None,indicator_name=None):
    """
    TODO better description
    
    Calculates properties of one spiketrain (already binned in the spike_vector).

    spike_train is a spike train directly from the nix file, with all metadata included
    """
    from copy import deepcopy 

    if indicator is not None:
        mask = indicator>0
        spike_vector = spike_vector[mask]
        rb_phase = rb_phase[mask]
        rb_envelope = rb_envelope[mask]
        rb_env_phase = rb_env_phase[mask]
        
    ### lists of RB phases and envs. of spikes - CAREFUL WITH MULTIPLE SPIKES IN ONE MS
    phases_list = []
    env_list = []
    phases_env_list = []
    aux_sp = deepcopy(spike_vector)
    while np.sum(aux_sp)>0:
        non_zero_idx = np.where(aux_sp>0)[0]
        for idx in non_zero_idx:
            phases_list.append(rb_phase[idx]) 
            env_list.append(rb_envelope[idx]) 
            phases_env_list.append(rb_env_phase[idx]) 
            aux_sp[idx]-=1 ### subtracting spikes that we have already used

    ### phase preference
    r, phi = circular_avg(np.array(phases_list),bins=30)
    r_env, phi_env = circular_avg(np.array(phases_env_list),bins=30)

    ### phases of high and low envelope spikes 
    high_env_mask = np.array(env_list)>=np.median(rb_envelope) #### mask from the envelope values for each spike (NOT IN SHAPE OF INPUT ARRAYS, only spikes)
    low_env_mask = np.array(env_list)<np.median(rb_envelope)
    
    list_phases_high_env = np.array(phases_list)[high_env_mask]
    list_phases_low_env = np.array(phases_list)[low_env_mask]

    list_env_phases_high_env = np.array(phases_env_list)[high_env_mask]
    list_env_phases_low_env = np.array(phases_env_list)[low_env_mask]

    ### firing rate 
    dur_rec_ms = spike_vector.shape[0]
    dur_rec_s = dur_rec_ms/1000
    fr = np.sum(spike_vector)/dur_rec_s
    fr_high_env = len(list_phases_high_env)/dur_rec_s*2 ### we normalise by 2, because this only considers spikes above median env.
    fr_low_env = len(list_phases_low_env)/dur_rec_s*2
    
    ### CV ISI
    len_intervals = count_zero_intervals(spike_vector) 
    CV_ISI = np.std(np.array(len_intervals))/np.mean(np.array(len_intervals))

    if indicator is None:
        ind_string = ''
    else:
        if indicator_name is not None:
            ind_string = f'_{indicator_name}'
        else:
            print('No indicator name given.')
            return
        
    prop_dict = {f'FR{ind_string}':fr, 
                f'CV_ISI{ind_string}': CV_ISI,
                f'ISI{ind_string}': len_intervals,

                f'env_th_median{ind_string}':np.median(rb_envelope), ### the median value of RB envelope on this channel
                 
                f'list_phases{ind_string}': phases_list,
                f'list_env{ind_string}':env_list,
                f'list_env_phases{ind_string}':phases_env_list,

                f'list_phases_high_env{ind_string}':list_phases_high_env,
                f'list_env_phases_high_env{ind_string}':list_env_phases_high_env,
                 
                f'list_phases_low_env{ind_string}':list_phases_low_env,
                f'list_env_phases_low_env{ind_string}':list_env_phases_low_env,

                f'FR_high_env_median{ind_string}':fr_high_env,
                f'FR_low_env_median{ind_string}':fr_low_env, 
                f'FR_high_env_low_env_median_ratio{ind_string}':fr_high_env/fr_low_env, 
                 
                f'pref_phase_spikes{ind_string}':phi, 
                f'norm_RB_phase_selectivity_spikes{ind_string}':r, 
                f'pref_env_phase_spikes{ind_string}': phi_env,
                f'norm_RB_env_phase_selectivity_spikes{ind_string}': r_env,
                }
    # adding other percentile TH values, so the spikes can be splitted into high/low env. in different ways later
    for perc in [10,20,30,40,50,60,70,80,90,95]:
        prop_dict[f'env_th_perc_{perc}{ind_string}'] = np.percentile(rb_envelope,perc)
    
    if channel_prop is not None:
        for k in channel_prop.keys():
            prop_dict[k] = channel_prop[k]
    return prop_dict


















