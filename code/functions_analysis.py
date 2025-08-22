##### FUNCTIONS FOR ANALYSIS OF THE MACAQUE RIPPLE BAND #####

import os
import numpy as np
import pandas as pd
import elephant
from scipy.signal import hilbert
import neo
from scipy.stats import zscore
import quantities as pq
from scipy.signal import welch
import pickle
from scipy.stats import skew
from scipy.signal import find_peaks


########## DATA LOADING AND HANDLING ##########


def load_block(monkey,array,type_rec,type_sig,date,data_folder=''):
    """
    Loading the monkey data (already preprocessed as in the snake files).
    
    type_rec: RS, OG
    type_sig: LFP, RB, tMUA, spikes
    """
    path = f'{data_folder}/macaque{monkey}_{type_rec}_{date}/{type_sig}/macaque{monkey}_{type_rec}_{date}_Array{array}_{type_sig}.nix'
    try:
        io = neo.NixIO(path,'ro')
        block = io.read_block()
        return block
    except:
        print(path)


def load_prop_df(monkey_list,condition='RS',params={},df_folder='',exclude_noisy=True):
    """
    Loads DF with all the SUA properties, for all dates in params, all monkeys in monkey list merged.
    """
    df_list = []
    for monkey in monkey_list:
        print(monkey)
        all_dates = params['dates'][monkey][condition]
        for date in all_dates:
            print(date)
            if condition=='RS':
                name_folder = 'sua_prop_all'
            elif condition=='OG':
                name_folder = 'sua_prop_all_OG'
            with open(f'{df_folder}/{name_folder}/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
                 df_sua = pickle.load(file)
            df_list.append(df_sua)
    
    df_merged = pd.concat(df_list,ignore_index=True)

    a_list = []
    for idx in df_merged.index:
        a = df_merged.loc[idx,'area']
        if a in ['V4','IT']:
            a_list.append(a)
        else:
            a_list.append('V12')
    
    df_merged['area_merged'] = a_list
    
    if exclude_noisy:
        df_merged = df_merged[~df_merged['ch_is_noisy_100Hz']]
        df_merged = df_merged[~df_merged['ch_is_noisy_120Hz']]
        
    return df_merged

def load_trig_df(monkey_list,condition='RS',params = {},df_folder='',exclude_noisy=True):
    """
    Loads DF with the SUA triggered data.
    """
    df_list = []
    for monkey in monkey_list:
        print(monkey)
        all_dates = params['dates'][monkey][condition]
        for date in all_dates:
            print(date)
            if condition=='RS':
                name_folder = 'sua_prop_triggered'
            elif condition=='OG':
                name_folder = 'sua_prop_triggered_OG'
            with open(f'{df_folder}/{name_folder}/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
                 df_sua = pickle.load(file)
            df_list.append(df_sua)
    
    df_merged = pd.concat(df_list,ignore_index=True)
    
    if exclude_noisy:
        df_merged = df_merged[~df_merged['ch_is_noisy_100Hz']]
        df_merged = df_merged[~df_merged['ch_is_noisy_120Hz']]
    return df_merged


def load_ripple_trig_df(monkey_list,dual_th=[2.5,3.5],condition='RS',params = {},df_folder='',trigg_option='peak'):
    """
    Loading dataframes with thresholded ripples triggered statistics.

    trigg_option: peak, start, stop
    """
    df_list = []
    for monkey in monkey_list:
        print(monkey)
        all_dates = params['dates'][monkey][condition]
        for date in all_dates:
            print(date)
            if condition=='RS':
                name_folder = 'ripple_prop_triggered'
            elif condition=='OG':
                name_folder = 'ripple_prop_triggered_OG'
            df_ripp = pd.read_pickle(f'{df_folder}/{name_folder}/th__{int(dual_th[0]*10)}_{int(dual_th[1]*10)}/{trigg_option}_monkey{monkey}_all_arrays_date_{date}.pkl')
            if df_ripp.shape[0]>0:
                df_list.append(df_ripp)
    
    df_merged = pd.concat(df_list,ignore_index=True)
    return df_merged


def load_shuffle_df(monkey_list,condition='RS',params = {},df_folder='',exclude_noisy=True):
    """
    Loads DF with the shuffled selectivity th. data.
    """
    df_list = []
    for monkey in monkey_list:
        print(monkey)
        all_dates = params['dates'][monkey][condition]
        for date in all_dates:
            print(date)
            if condition=='RS':
                name_folder = 'sua_shuffled_phases'
            elif condition=='OG':
                name_folder = 'sua_shuffled_phases_OG'
            with open(f'{df_folder}/{name_folder}/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
                 df_sua = pickle.load(file)
            df_list.append(df_sua)
    
    df_merged = pd.concat(df_list,ignore_index=True)
    
    if exclude_noisy:
        df_merged = df_merged[~df_merged['ch_is_noisy_100Hz']]
        df_merged = df_merged[~df_merged['ch_is_noisy_120Hz']]
    return df_merged
    

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
    However, if the value in the col. is 0, number 1 will be put instead.
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


def aux_RB_noise_prop(sig_vec,fs=1000):
    """
    100 Hz noise presence: 
    For a given vector of signal, compares the avg. PSD in the interval [98,102) and in [90,95). 
    If the power in the [98,102) interval is stonger, returns noise100 = True (there is 100 Hz noise present).

    For 120 Hz the same idea is used on intervals [118,122) and [110,115).

    Remark: In our analysis is used on spectra of RB signal, already filtered in [80,150].
    """
    frequencies, psd = welch(sig_vec, fs=fs, nperseg=fs*2)  # 2-second segments

    mask_100 = (frequencies>=98) & (frequencies<102)
    mask_before_100 = (frequencies>=90) & (frequencies<95)

    mask_120 = (frequencies>=118) & (frequencies<122)
    mask_before_120 = (frequencies>=110) & (frequencies<115)
    
    psd_around100 = np.mean(psd[mask_100])
    psd_before100 = np.mean(psd[mask_before_100])
    
    psd_around120 = np.mean(psd[mask_120])
    psd_before120 = np.mean(psd[mask_before_120])

    noise100 = psd_around100 > psd_before100
    noise120 = psd_around120 > psd_before120
    
    return noise100, noise120


def spike_train_prop_vec(spike_vector,rb_sig,LFP_sig,rb_phase,rb_envelope,rb_env_phase,channel_prop=None,indicator=None,indicator_name=None):
    """
    TODO better description
    
    Calculates properties of one spiketrain (already binned in the spike_vector).

    spike_train is a spike train directly from the nix file, with all metadata included
    """
    from copy import deepcopy 

    if indicator is not None:
        mask = indicator>0
        try:
            spike_vector = spike_vector[mask]
        except:
            spike_vector = spike_vector[mask[:-1]]  # sometimes the spike vector is one bin shorter
        rb_sig = rb_sig[mask]
        LFP_sig = LFP_sig[mask]
        rb_phase = rb_phase[mask]
        rb_envelope = rb_envelope[mask]
        rb_env_phase = rb_env_phase[mask]
        
    ### lists of RB phases and envs. of spikes - CAREFUL WITH MULTIPLE SPIKES IN ONE MS
    phases_list = []
    env_list = []
    phases_env_list = []
    LFP_list = []
    aux_sp = deepcopy(spike_vector)
    while np.sum(aux_sp)>0:
        non_zero_idx = np.where(aux_sp>0)[0]
        for idx in non_zero_idx:
            phases_list.append(rb_phase[idx]) 
            env_list.append(rb_envelope[idx]) 
            phases_env_list.append(rb_env_phase[idx]) 
            LFP_list.append(LFP_sig[idx])
            aux_sp[idx]-=1 ### subtracting spikes that we have already used

    ### phase preference
    r, phi = circular_avg(np.array(phases_list),bins=30)
    r_env, phi_env = circular_avg(np.array(phases_env_list),bins=30)

    ### phases of high and low envelope spikes 
    high_env_mask = np.array(env_list)>=np.median(rb_envelope)  # mask from the envelope values for each spike (NOT IN SHAPE OF INPUT ARRAYS, only spikes)
    low_env_mask = np.array(env_list)<np.median(rb_envelope)
    
    list_phases_high_env = np.array(phases_list)[high_env_mask]
    list_phases_low_env = np.array(phases_list)[low_env_mask]

    list_env_phases_high_env = np.array(phases_env_list)[high_env_mask]
    list_env_phases_low_env = np.array(phases_env_list)[low_env_mask]

    ### firing rate 
    dur_rec_ms = spike_vector.shape[0]
    dur_rec_s = dur_rec_ms/1000
    fr = np.sum(spike_vector)/dur_rec_s
    fr_high_env = len(list_phases_high_env)/dur_rec_s*2  # we normalise by 2, because this only considers spikes above median env.
    fr_low_env = len(list_phases_low_env)/dur_rec_s*2
    
    ### CV ISI
    len_intervals = count_zero_intervals(spike_vector) 
    CV_ISI = np.std(np.array(len_intervals))/np.mean(np.array(len_intervals))

    ### line noise presence in RB
    noise100, noise120 = aux_RB_noise_prop(rb_sig,fs=1000)

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

                f'LFP_list{ind_string}':LFP_list,

                f'FR_high_env_median{ind_string}':fr_high_env,
                f'FR_low_env_median{ind_string}':fr_low_env, 
                f'FR_high_env_low_env_median_ratio{ind_string}':fr_high_env/fr_low_env, 
                 
                f'pref_RB_phase_spikes{ind_string}':phi, 
                f'norm_RB_phase_selectivity_spikes{ind_string}':r, 
                f'pref_RB_env_phase_spikes{ind_string}': phi_env,
                f'norm_RB_env_phase_selectivity_spikes{ind_string}': r_env,

                f'ch_is_noisy_100Hz{ind_string}': noise100,
                f'ch_is_noisy_120Hz{ind_string}': noise120,
                }
    # adding other percentile TH values, so the spikes can be splitted into high/low env. in different ways later
    for perc in [10,20,30,40,50,60,70,80,90,95]:
        prop_dict[f'env_th_perc_{perc}{ind_string}'] = np.percentile(rb_envelope,perc)
    
    if channel_prop is not None:
        for k in channel_prop.keys():
            prop_dict[k] = channel_prop[k]
    return prop_dict


def aux_add_up_down_classes(df_sua):
    """
    Clasifies whether the peak is UP or DOWN (bigger in abs. val above, or below 0), in the zscored waveform.
    """
    df_added = df_sua
    aux_classes = []
    for idx in df_added.index:
        wf = df_added.loc[idx]['avg_wf']
        if np.abs(np.max(wf))>np.abs(np.min(wf)):
            aux_classes.append('UP')
        else:
            aux_classes.append('DOWN')

    df_added['wf_direction'] = aux_classes
    return df_added


def aux_add_waveform_prop(df_sua):
    """
    From the dataframe with formated waveform properties calculates waveform width and height (amplitude).
    """
    df_added = df_sua
    ### amplitude
    waveforms = df_sua['avg_wf'].values
    df_added['amp_wf'] = [np.max(wf) - np.min(wf) for wf in waveforms]
    ### distance from peak to trough
    min_idcs = [np.argmin(wf) for wf in waveforms]
    # measuring distance from the trough, to the right hand side maximum
    df_added['width_wf'] = [np.abs(np.argmax(wf[min_idx:])+min_idx - np.argmin(wf)) for wf, min_idx in zip(waveforms,min_idcs)]
    return df_added


def aux_add_zscored_avg_waveform(df_sua):
    """
    From a dataframe with formated waveforms, saves one more column with each waveform zscored, 
    and another column with its zscored amplitude.
    """
    from scipy.stats import zscore
    
    waveforms = df_sua['avg_wf'].values
    df_added = df_sua
    df_added['avg_wf_zscored'] =  [zscore(wf.magnitude) for wf in waveforms]
    wfs_zsc = df_added['avg_wf_zscored'].values
    df_added['amp_wf_zscored'] = [np.max(wf) - np.min(wf) for wf in wfs_zsc]
    
    return df_added


def aux_add_width_classes(df_sua,width_intervals):
    """
    Adding width class info, based on the measured width of a waveform (peak to the right max.).
    """
    names_widths = ['narrow','medium','wide']
    df_added = df_sua
    ### adding column with spike width classification into narrow, medium, wide
    aux_classes = []
    for idx in df_added.index:
        width_row = df_added.loc[idx]['width_wf']
        for i in range(len(width_intervals)):
            interval = width_intervals[i]
            if (width_row>=interval[0]) & (width_row<=interval[1]):
                aux_classes.append(names_widths[i])
    
    df_added['width_wf_class'] = np.array(aux_classes)
    return df_added


def aux_add_selectivity(df_sua,sel_th=0.06):
    """
    Adds a column with selectivity RB info (selectivity wrt. RB, bool)
    """
    select_mask = df_sua['norm_RB_phase_selectivity_spikes'].values>sel_th
    df_sua['is_RB_phase_selective'] = select_mask
    return df_sua


def aux_add_final_classes(df_sua,final_classes,peak_height_th):
    """
    Final classification by the waveform width,
    narrow cells moreover split into two subclasses.

    This classification is added to the inputed dataframe.
    """
    
    df_sua['final_class'] = 'NO_CLASS'
    dict_cl_indices = {}
    for cl in final_classes:
        if cl=='DOWN_narrow_shallow':
            aux_df = df_sua[df_sua['wf_direction']=='DOWN']
            aux_df = aux_df[aux_df['width_wf_class']=='narrow']
            idx_list = []
            for idx in aux_df.index:
                max_before_tr=np.max(aux_df.loc[idx,'avg_wf_zscored'][:30])
                if max_before_tr<peak_height_th:
                    idx_list.append(idx)
            dict_cl_indices['DOWN_narrow_shallow'] = idx_list
        elif cl=='DOWN_narrow_sharp':
            aux_df = df_sua[df_sua['wf_direction']=='DOWN']
            aux_df = aux_df[aux_df['width_wf_class']=='narrow']
            idx_list = []
            for idx in aux_df.index:
                max_before_tr=np.max(aux_df.loc[idx,'avg_wf_zscored'][:30])
                if max_before_tr>=peak_height_th:
                    idx_list.append(idx)
            dict_cl_indices['DOWN_narrow_sharp'] = idx_list
        elif cl=='DOWN_medium_shallow':
            aux_df = df_sua[df_sua['wf_direction']=='DOWN']
            aux_df = aux_df[aux_df['width_wf_class']=='medium']
            idx_list = []
            for idx in aux_df.index:
                max_before_tr=np.max(aux_df.loc[idx,'avg_wf_zscored'][:30])
                if max_before_tr<peak_height_th:
                    idx_list.append(idx)
            dict_cl_indices['DOWN_medium_shallow'] = idx_list
        elif cl=='DOWN_medium_sharp':
            aux_df = df_sua[df_sua['wf_direction']=='DOWN']
            aux_df = aux_df[aux_df['width_wf_class']=='medium']
            idx_list = []
            for idx in aux_df.index:
                max_before_tr=np.max(aux_df.loc[idx,'avg_wf_zscored'][:30])
                if max_before_tr>=peak_height_th:
                    idx_list.append(idx)
            dict_cl_indices['DOWN_medium_sharp'] = idx_list
        elif cl=='DOWN_wide':
            aux_df = df_sua[df_sua['wf_direction']=='DOWN']
            aux_df = aux_df[aux_df['width_wf_class']=='wide']
            dict_cl_indices['DOWN_wide'] = aux_df.index
        elif cl=='UP':
            aux_df = df_sua[df_sua['wf_direction']=='UP']
            dict_cl_indices['UP'] = aux_df.index
        else:
            print('Undefined cell type.')
    for cl in final_classes:
        for i in dict_cl_indices[cl]:
            df_sua.loc[i,'final_class'] = cl
    return df_sua


def fill_short_intervals(binary_array, threshold=100):
    """Modify a binary array by turning short zero intervals into ones.
    
    Args:
        binary_array (np.ndarray): The input binary array.
        threshold (int): The maximum length of zero intervals to replace with ones.
    
    Returns:
        np.ndarray: The modified binary array.
    """
    binary_array = binary_array.copy()  # Avoid modifying the original array
    zero_regions = np.where(binary_array == 0)[0]  # Get indices of zeros
    
    # Detect continuous zero intervals
    split_indices = np.where(np.diff(zero_regions) != 1)[0] + 1
    zero_groups = np.split(zero_regions, split_indices)
    
    # Replace short zero intervals with ones
    for group in zero_groups:
        if len(group) < threshold:
            binary_array[group] = 1
    
    return binary_array


def create_no_stim_indic(indic0,indic1,threshold=250):
    """
    Out of two binary vectors, creates their merge, puts all intervals with 0 shorter than N to one, and then negate.
    """
    sum_vec = indic0+indic1
    mask = sum_vec>0 ### indices of non-zero digits
    sum_vec[mask] = 1
    sum_vec = fill_short_intervals(sum_vec, threshold=threshold)
    no_stim_vec = ~(sum_vec.astype(bool))
    no_stim_vec = no_stim_vec.astype(np.int64)
    return no_stim_vec


########## TRIGGERED STATISTICS ##########

def trigg_list(trigg_vec,sig_vec,width_idx=2000):
    """
    Auxiliary for triggered prop.
    Takes non-zero indices in trigg_vec and cut out the pieces of width width_idx from the sig_vec around these indices.
    """
    non_zero_idx = np.where(trigg_vec>0)[0]  # indices of triggers in 1D array
    non_zero_idx = non_zero_idx[non_zero_idx>width_idx]  # discarding the triggers that do not have enough space on the left
    non_zero_idx = non_zero_idx[non_zero_idx<sig_vec.shape[0]-width_idx] # discarding the triggers that do not have enough space on the right
    trigg_list = [sig_vec[idx-width_idx:idx+width_idx] for idx in non_zero_idx]
    return trigg_list


def triggered_prop(spike_vec,rb_sig_vec,rb_env_vec,LFP_vec,
                   indicator_list = [], indicator_names = ['EC','EO'],width_cut=2000,channel_prop=None):
    """
    Auxiliary for triggered prop.
    Creating dictionary with trig. pieces of the signal cut out.

    Indicator list should contain bool masks.
    """
    prop_dict = {}
    prop_dict['RB_sig_trigg_sum'] = np.sum(trigg_list(spike_vec,rb_sig_vec,width_idx=width_cut),axis=0)
    prop_dict['RB_env_trigg_sum'] = np.sum(trigg_list(spike_vec,rb_env_vec,width_idx=width_cut),axis=0)
    prop_dict['LFP_trigg_sum'] = np.sum(trigg_list(spike_vec,LFP_vec,width_idx=width_cut),axis=0)
    prop_dict['RB_sig_trigg_mean'] = np.mean(trigg_list(spike_vec,rb_sig_vec,width_idx=width_cut),axis=0)
    prop_dict['RB_env_trigg_mean'] = np.mean(trigg_list(spike_vec,rb_env_vec,width_idx=width_cut),axis=0)
    prop_dict['LFP_trigg_mean'] = np.mean(trigg_list(spike_vec,LFP_vec,width_idx=width_cut),axis=0)
    if len(indicator_list)>0:
        if len(indicator_list)==len(indicator_names):
            for i in range(len(indicator_list)):
                indicator = indicator_list[i]
                spike_vec_new = spike_vec[indicator]
                ind_name = indicator_names[i]
                prop_dict[f'RB_sig_trigg_sum_{ind_name}'] = np.sum(trigg_list(spike_vec_new,rb_sig_vec,width_idx=width_cut),axis=0)
                prop_dict[f'RB_env_trigg_sum_{ind_name}'] = np.sum(trigg_list(spike_vec_new,rb_env_vec,width_idx=width_cut),axis=0)
                prop_dict[f'LFP_trigg_sum_{ind_name}'] = np.sum(trigg_list(spike_vec_new,LFP_vec,width_idx=width_cut),axis=0)
                prop_dict[f'RB_sig_trigg_mean_{ind_name}'] = np.mean(trigg_list(spike_vec_new,rb_sig_vec,width_idx=width_cut),axis=0)
                prop_dict[f'RB_env_trigg_mean_{ind_name}'] = np.mean(trigg_list(spike_vec_new,rb_env_vec,width_idx=width_cut),axis=0)
                prop_dict[f'LFP_trigg_mean_{ind_name}'] = np.mean(trigg_list(spike_vec_new,LFP_vec,width_idx=width_cut),axis=0)
        else:
            print('Wrong number of names for indicators given.')
            return
    
    if channel_prop is not None:
        for k in channel_prop.keys():
            prop_dict[k] = channel_prop[k]
    return prop_dict


def ripple_triggered_prop(ripple_center_vec,EC_ripple_center_vec,EO_ripple_center_vec,
                          HIGH_ripple_center_vec,ZERO_ripple_center_vec,LOW_ripple_center_vec,
                          rb_sig_vec,rb_env_vec,LFP_vec,
                          spikes_sum_dict_array={},spikes_sum_dict_ch={},
                          width_cut=2000,channel_prop=None,cell_classes=[]):
    """
    Auxiliary for triggered prop of ripples.
    Creating dictionary with trig. pieces of the signal cut out.

    Spikes sum vec is a sum of all the SUA spikes on the given channel, regardless on the class.
    """
    prop_dict = {}

    for option in ['','_EC','_EO','_HIGH','_ZERO','_LOW']:
        if option=='_EC':
            trig_vec = EC_ripple_center_vec
        elif option=='_EO':
            trig_vec = EO_ripple_center_vec
        elif option=='_HIGH':
            trig_vec = HIGH_ripple_center_vec
        elif option=='_ZERO':
            trig_vec = ZERO_ripple_center_vec
        elif option=='_LOW':
            trig_vec = LOW_ripple_center_vec
        else:
            trig_vec = ripple_center_vec
        
        prop_dict[f'RB_sig_trigg_sum{option}'] = np.sum(trigg_list(trig_vec,rb_sig_vec,width_idx=width_cut),axis=0)
        prop_dict[f'RB_env_trigg_sum{option}'] = np.sum(trigg_list(trig_vec,rb_env_vec,width_idx=width_cut),axis=0)
        prop_dict[f'LFP_trigg_sum{option}'] = np.sum(trigg_list(trig_vec,LFP_vec,width_idx=width_cut),axis=0)
        
        prop_dict[f'RB_sig_trigg_mean{option}'] = np.mean(trigg_list(trig_vec,rb_sig_vec,width_idx=width_cut),axis=0)
        prop_dict[f'RB_env_trigg_mean{option}'] = np.mean(trigg_list(trig_vec,rb_env_vec,width_idx=width_cut),axis=0)
        prop_dict[f'LFP_trigg_mean{option}'] = np.mean(trigg_list(trig_vec,LFP_vec,width_idx=width_cut),axis=0)
    
        for cl in cell_classes:
            prop_dict[f'spikes_{cl}_trigg_sum_array{option}'] = np.sum(trigg_list(trig_vec,spikes_sum_dict_array[cl],width_idx=width_cut),axis=0)
            prop_dict[f'spikes_{cl}_trigg_mean_array{option}'] = np.mean(trigg_list(trig_vec,spikes_sum_dict_array[cl],width_idx=width_cut),axis=0)
            prop_dict[f'spikes_{cl}_trigg_sum_ch{option}'] = np.sum(trigg_list(trig_vec,spikes_sum_dict_ch[cl],width_idx=width_cut),axis=0)
            prop_dict[f'spikes_{cl}_trigg_mean_ch{option}'] = np.mean(trigg_list(trig_vec,spikes_sum_dict_ch[cl],width_idx=width_cut),axis=0)
    
    if channel_prop is not None:
        for k in channel_prop.keys():
            prop_dict[k] = channel_prop[k]
    return prop_dict


########### SHUFFLE PHASES ANALYSIS ##########


def shuffle_distrib(spike_vec,rb_phase_vec,rb_env_phase_vec,channel_prop={},num_repeat=100):
    """
    Creating dictionary with shuffle distribution data of a given cell,
    the dict. contains the RB phase shuffle, as well as RB env. phase shuffle, and some additional information about the cell. 
    """
    ### generating repeatedly shifted vectors
    np.random.seed(42)
    spike_indices = np.where(spike_vec>0)[0]
    shifts = np.random.uniform(low=0,high=len(spike_vec),size=num_repeat)
    
    rolled_phases = [np.roll(rb_phase_vec,shift) for shift in shifts]
    rolled_env_phases = [np.roll(rb_env_phase_vec,shift) for shift in shifts]
        
    spike_phases_list = [vec[spike_indices] for vec in rolled_phases]
    spike_env_phases_list = [vec[spike_indices] for vec in rolled_env_phases]

    # r, phi = circular_avg(np.array(phases_list),bins=25)
    select_phases = []
    angle_phases = []
    for l in spike_phases_list: # itterating in thee list of lists
        r, phi = circular_avg(np.array(l),bins=25) # it was 30 here previously, TODO RUN AGAIN AND ERASE
        select_phases.append(r)
        angle_phases.append(phi)
    select_phases = np.array(select_phases)
    angle_phases = np.array(angle_phases)

    select_env_phases = []
    angle_env_phases = []
    for l in spike_env_phases_list:
        r, phi = circular_avg(np.array(l),bins=25) # 30 as well here in the last run TODO ERASE
        select_env_phases.append(r)
        angle_env_phases.append(phi)
    select_env_phases = np.array(select_env_phases)
    angle_env_phases = np.array(angle_env_phases)

    ### creating distribution of phases
    prop_dict = {}

    phase_counts, bin_edges = np.histogram(select_phases, bins=100, range=(0, 1))
    env_phase_counts, _ = np.histogram(select_env_phases, bins=100, range=(0, 1))
    phase_angle_counts, bin_edges_angle = np.histogram(angle_phases, bins=100, range=(-np.pi, np.pi))
    env_phase_angle_counts, _ = np.histogram(angle_env_phases, bins=100, range=(-np.pi, np.pi))

    prop_dict['bin_edges'] = bin_edges
    prop_dict['bin_edges_angle'] = bin_edges_angle
    prop_dict['phase_counts'] = phase_counts
    prop_dict['env_phase_counts'] = env_phase_counts
    prop_dict['phase_angles_counts'] = phase_angle_counts
    prop_dict['env_phase_angles_counts'] = env_phase_angle_counts
    
    ### adding channel prop
    if channel_prop is not None:
        for k in channel_prop.keys():
            prop_dict[k] = channel_prop[k]
        
    return prop_dict


########## RIPPLE ANALYSIS ##########

### threshold detection
### modified from https://neurodsp-tools.github.io/neurodsp/generated/neurodsp.burst.detect_bursts_dual_threshold.html

"""The dual threshold algorithm for detecting oscillatory bursts in a neural signal."""
#from neurodsp.utils.core import get_avg_func
#from neurodsp.utils.checks import check_param_options
#from neurodsp.utils.decorators import multidim
#from neurodsp.timefrequency.hilbert import amp_by_time


def get_avg_func(avg_type):
    """Select a function to use for averaging.

    Parameters
    ----------
    avg_type : {'mean', 'median', 'sum'}
        The type of averaging function to use.

    Returns
    -------
    func : callable
        Requested function.
    """

    if avg_type == 'mean':
        func = np.mean
    elif avg_type == 'median':
        func = np.median
    elif avg_type == 'std':
        func = np.std

    return func


def detect_bursts_dual_th(sig_magnitude, fs, dual_thresh, f_range=None,
                                 min_n_cycles=None, min_burst_duration=0.04,
                                 avg_type='std', magnitude_type='amplitude', EC_indicator=None):
    """
    Detect bursts in a signal using the dual threshold algorithm.

    Parameters
    ----------
    sig_magnitude : 1d array
        Time series of envelope of the signal. This will be thresholded.
    fs : float
        Sampling rate, in Hz.
    dual_thresh : tuple of (float, float)
        Low and high threshold values for burst detection.
        Units are normalized by the average signal magnitude.
    f_range : tuple of (float, float), optional
        Frequency range, to filter signal to, before running burst detection.
        If f_range is None, then no filtering is applied prior to running burst detection.
    min_n_cycles : float, optional, default None
        Minimum burst duration in to keep.
        Only used if `f_range` is defined, and is used as the number of cycles at f_range[0].
    min_burst_duration : float, optional, default: 0.04 s
        Minimum length of a burst, in seconds. Must be defined if not filtering.
        Only used if `f_range` is not defined, or if `min_n_cycles` is set to None.
    avg_type : {'median', 'mean', 'std'}, optional
        Averaging method to use to normalize the magnitude that is used for thresholding.
    magnitude_type : {'amplitude', 'power'}, optional
        Metric of magnitude used for thresholding.
    EC_indicator : Vector with 1 for every time step where the eyes were closed. Time steps as in the signal.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    is_burst : 1d array
        Boolean indication of where bursts are present in the input signal.
        True indicates that a burst was detected at that sample, otherwise False.

    Notes
    -----
    The dual-threshold burst detection algorithm was originally proposed in [1].

    References
    ----------
    .. [1] Feingold, J., Gibson, D. J., DePasquale, B., & Graybiel, A. M. (2015).
           Bursts of beta oscillation differentiate postperformance activity in
           the striatum and motor cortex of monkeys performing movement tasks.
           Proceedings of the National Academy of Sciences, 112(44), 13687–13692.
           DOI: https://doi.org/10.1073/pnas.1517629112
    """

    if len(dual_thresh) != 2:
        raise ValueError("Invalid number of elements in 'dual_thresh' parameter")

    if magnitude_type == 'power':
        sig_magnitude = sig_magnitude**2  

    # Calculate normalized magnitude
    if avg_type in ['std','median','mean']:
        sig_magnitude = sig_magnitude / get_avg_func(avg_type)(sig_magnitude)
    elif avg_type=='std_EC':
        sig_magnitude = sig_magnitude / np.std(sig_magnitude[EC_indicator>0])
    elif avg_type=='std_EO':
        sig_magnitude = sig_magnitude / np.std(sig_magnitude[EC_indicator==0])

    # Identify time periods of bursting using the 2 thresholds
    is_burst = _dual_threshold_split(sig_magnitude, dual_thresh[1], dual_thresh[0])

    # Remove bursts detected that are too short
    # Use a number of cycles defined on the frequency range, if available
    if f_range is not None and min_n_cycles is not None:
        min_burst_samples = int(np.ceil(min_n_cycles * fs / f_range[0]))
    # Otherwise, make sure minimum duration is set, and use that
    else:
        if min_burst_duration is None:
            raise ValueError("Minimum burst duration must be defined if not filtering "
                             "and using a number of cycles threshold.")
        min_burst_samples = int(np.ceil(min_burst_duration * fs))

    is_burst = _rmv_short_periods(is_burst, min_burst_samples)

    return is_burst.astype(bool)


def _dual_threshold_split(sig, thresh_hi, thresh_lo):
    """
    Identify periods that are above thresh_lo and have at least one value above thresh_hi.
    """

    # Find all values above thresh_hi
    # To avoid bug in later loop, do not allow first or last index to start off as 1
    sig[[0, -1]] = 0
    idx_over_hi = np.where(sig >= thresh_hi)[0]

    # Initialize values in identified period
    positive = np.zeros(len(sig))
    positive[idx_over_hi] = 1

    # Iteratively test if a value is above thresh_lo if it is not currently in an identified period
    sig_len = len(sig)

    for ind in idx_over_hi:

        j_down = ind - 1
        if positive[j_down] == 0:
            j_down_done = False
            while j_down_done is False:
                if sig[j_down] >= thresh_lo:
                    positive[j_down] = 1
                    j_down -= 1
                    if j_down < 0:
                        j_down_done = True
                else:
                    j_down_done = True

        j_up = ind + 1
        if positive[j_up] == 0:
            j_up_done = False
            while j_up_done is False:
                if sig[j_up] >= thresh_lo:
                    positive[j_up] = 1
                    j_up += 1
                    if j_up >= sig_len:
                        j_up_done = True
                else:
                    j_up_done = True
    return positive


def _rmv_short_periods(sig, n_samples):
    """
    Remove periods that are equal to 1 for less than n_samples.
    """

    if np.sum(sig) == 0:
        return sig

    osc_changes = np.diff(1 * sig)
    osc_starts = np.where(osc_changes == 1)[0]
    osc_ends = np.where(osc_changes == -1)[0]

    if len(osc_starts) == 0:
        osc_starts = [0]
    if len(osc_ends) == 0:
        osc_ends = [len(osc_changes)]

    if osc_ends[0] < osc_starts[0]:
        osc_starts = np.insert(osc_starts, 0, 0)
    if osc_ends[-1] < osc_starts[-1]:
        osc_ends = np.append(osc_ends, len(osc_changes))

    osc_length = osc_ends - osc_starts
    osc_starts_long = osc_starts[osc_length >= n_samples]
    osc_ends_long = osc_ends[osc_length >= n_samples]

    is_osc = np.zeros(len(sig))
    for ind in range(len(osc_starts_long)):
        is_osc[osc_starts_long[ind]:osc_ends_long[ind]] = 1

    return is_osc


def aux_from_env_to_data(vec,mult_factor=1000):
    """
    Skewness is not calculated on the density, but on the data. 
    We convert the envelopes (float) into integer type of data for each T point, as if converting to historam and bin count.
    We then calculate skewness on the binned version.

    Inputs 1d vector.
    Returns list of data points.
    """
    vec_int = (vec*mult_factor).astype(int)
    data_repeat = np.repeat(range(len(vec)),vec_int)
        
    return data_repeat


def find_first_peak_height(signal_vec):
    """
    Finds the first peak value of the signal.  
    Both negative and positive peaks are considered.
    """
    positive_peaks = find_peaks(signal_vec)[0]
    negative_peaks = find_peaks(-signal_vec)[0]

    first_idx = np.min([positive_peaks[0],negative_peaks[0]])
    peak_height = signal_vec[first_idx]
    return peak_height
    

def ripple_prop(monkey, array, date, dual_th=[2.5,3.5], f_range= [80,150], 
                type_rec='RS', magnitude_type='amplitude', avg_type='std', 
                EC_indicator=None, min_burst_duration = 0.04, fs=1000,data_folder='',params={}):
    """
    Args:
        monkey: monkey, needed for epochs dataframe input
        array: array from 1 to 16
        date: date of recording
        dual_th: parameter dual_thresh from detect_burst_dual_threshold
        f_range: ripple band range (WILL NOT BE USED FOR FILTERING, JUST FOR MINIMAL CYCLES SETTING, MAYBE ERASE)
        min_burst_duration: Minimal duration of detected bursts. Defaults to 0.04 s
        method: 'std', 'mean','median' - normalisation method for units of threshold
        bad_channel_list: list of indices of bad channels, ordering as in the RB sig. array. From 0 to 63.

    Returns: pd.DataFrame with statistics about individual ripples
    """

    RB_block = load_block(monkey,array,type_rec=type_rec,type_sig='RB',date=date,data_folder=data_folder)
    RB_sig_array = sig_block_to_arr(RB_block,'RB_filtered_zsc')
    RB_env_array = sig_block_to_arr(RB_block,'RB_envelope_norm')
    n_channels = 64
    
    list_ripples = []

    start_rec = float(RB_block.segments[0].analogsignals[0].t_start.rescale('ms').magnitude)
    stop_rec = float(RB_block.segments[0].analogsignals[0].t_stop.rescale('ms').magnitude)
    #print(start_rec)

    ### shifting indicator for arrays in N and F (there it does not go from 0)
    common_time_start = params['times_all_arr'][monkey][type_rec][date][0]
    diff_times = common_time_start - start_rec

    if EC_indicator is not None:
        if diff_times>0:
            print(f'Extending EC indicator by {diff_times} ms.')
            EC_indicator = np.hstack([np.zeros(int(diff_times)),EC_indicator])
    
    for ch in range(n_channels):
        ripple_sig_ch = RB_sig_array[ch,:]
        ripple_env_ch = RB_env_array[ch,:]

        noise100, noise120 = aux_RB_noise_prop(ripple_sig_ch,fs=1000)
        bad_channel = noise100 | noise120
        
        burst_vector = detect_bursts_dual_th(ripple_env_ch, fs=1000, dual_thresh=dual_th, 
                                             min_n_cycles=None,min_burst_duration=min_burst_duration, avg_type=avg_type, 
                                             magnitude_type=magnitude_type, EC_indicator = EC_indicator)

        change = np.diff(burst_vector)  # locates the starts and ends of bursts
        indcs = change.nonzero()[0]
        indcs += 1  # Get indices following the change.
        # the following 6 lines are copyied from detect_bursts_dual_threshold function in neurodsp.burst
        if burst_vector[0]:  # If the first sample is part of a burst, prepend a zero.
            indcs = np.r_[0, indcs]
        if burst_vector[-1]:    # If the last sample is part of a burst,
                                # append an index corresponding to the length of signal.
            indcs = np.r_[indcs, burst_vector.size]
        ch_starts = indcs[0::2]  # indices of beginnings of bursts for channel ch, indices are in units of time steps (for 500 Hz sample 1 unit is 2 ms)
        ch_ends = indcs[1::2]  # indices of ends of bursts for channel ch, indices are in units of time steps (for 500 Hz sample 1 unit is 2 ms)
        ch_durations = (ch_ends-ch_starts)*1000/fs

        for r in range(len(ch_starts)):  # for each ripple in the signal of the given electrode
            r_start = ch_starts[r] #ripple start, indices are in units of time steps (for 500 Hz sample 1 unit is 2 ms)
            r_end = ch_ends[r] #ripple stop, indices are in units of time steps (for 500 Hz sample 1 unit is 2 ms)
            r_duration = ch_durations[r]
            one_ripple = ripple_sig_ch[r_start:r_end]
            one_ripple_envelope = ripple_env_ch[r_start:r_end]
            signs = np.sign(one_ripple)  # array with signs of a given point of the signal
            
            # amplitude of the ripple
            amplitude = np.max(one_ripple_envelope) # amplitude in the units of standard deviations of envelope
            # time of the peak (the first one if multiple occur)
            max_index = np.argmax(np.abs(one_ripple))
            max_index_positive = np.argmax(one_ripple)
            # skewness of the Ripple envelope
            aux_skew_data = aux_from_env_to_data(one_ripple_envelope,mult_factor=1000)
            skew_r = skew(aux_skew_data)
            
            # if 0 is a peak, asign it a positive/negative sign, so it would be detected later as a crossing 
            for l in range(len(signs)-2):
                if False not in (signs[[l,l+1,l+2]]==[-1,0,-1]):
                    signs[l+1]=1
                if False not in (signs[[l,l+1,l+2]]==[1,0,1]):
                    signs[l+1]=-1
            
            sign_changes = np.zeros(len(signs))  # binary array with 1 for zero crossing
            # we assume there are no consecutive 0 in the signal
            for idx in range(0, len(signs) - 1):
                if idx>0:
                    zero_cross_with_zero = (signs[idx] == 0 and signs[idx-1]<0 and signs[idx+1]>0) or (signs[idx] == 0 and signs[idx-1]>0 and signs[idx+1]<0)
                else: 
                    zero_cross_with_zero = False
                if signs[idx] > 0 > signs[idx + 1] or signs[idx] < 0 < signs[idx + 1] or zero_cross_with_zero: #(signs[idx] == 0) in more simple way
                    sign_changes[idx] = 1
                    
            num_zero_crossings = sign_changes.sum() # number of ZC in the ripple
            
            # eyes closeed/open indicator
            # indicates if the ripple started during eyes closed period
            if type_rec == 'RS':
                try:
                    if EC_indicator[int(r_start*(1000/fs))]>0:
                        eyes_closed_r = True
                    else:
                        eyes_closed_r = False
                except:
                    eyes_closed_r = np.nan
            else:
                eyes_closed_r = np.nan ### this place can be potentially used for other indicators, for example grating trial indic.
            
            if sign_changes.sum() > 1:      # detect the first and the last zero crossing in the ripple
                                            # if there are at least two crossings
                zero_cross_idx = np.nonzero(sign_changes)[0]  # array with indices of non-zero elements
                idx_first_zero_c = zero_cross_idx[0]  # index of the first zero crossing of the ripple signal
                idx_last_zero_c = zero_cross_idx[-1]  # index of the last zero crossing of the ripple signal
                length_intercross = idx_last_zero_c - idx_first_zero_c  # the number of elements between
                                                                        # the first and the last crossing
                intercross_time = length_intercross / fs  # time between these crossings
                
                num_cycles = (num_zero_crossings - 1) / 2  # number of cycles in the ripple (approximation)
                freq = num_cycles / intercross_time  # frequency of the ripple (approximation)
                time_first_zero_c = (idx_first_zero_c+r_start)/fs # time of the first zero crossing
                time_last_zero_c = (idx_last_zero_c+r_start)/fs # time of the last zero crossing 

                # statistics of intercrossing intervals in the ripple r
                len_intercross_intervals = np.diff(zero_cross_idx)  # lengths of intercrossing intervals
                dur_intercross_intervals = len_intercross_intervals /fs
                mean = np.mean(dur_intercross_intervals)  # mean intercrossing duration
                std = np.std(dur_intercross_intervals)  # variation of intercrossing duration
                cv = std / mean  # coefficient of variation
                pos_peak_t_ms = (r_start+max_index_positive)*(1000/fs)
                
                dict_ripple_ch = {'channel_0_63': ch, # channels are numbered from 1 to 64
                                  'th_baseline': avg_type,
                                  'ICI time': intercross_time,
                                  'freq': freq,
                                  'skew_env':skew_r,
                                  'first_peak_sig_val': find_first_peak_height(one_ripple),
                                  'time_0_sig':one_ripple[0],
                                  'start_time_ms': r_start*(1000/fs),
                                  'stop_time_ms': r_end*(1000/fs),
                                  'eyes_closed': eyes_closed_r, 
                                  'duration_ms': r_duration,
                                  'amplitude': amplitude,
                                  'peak_time_ms': (r_start+max_index)*(1000/fs), 
                                  'positive_peak_time_ms': pos_peak_t_ms,
                                  'num_of_zero_cross': int(num_zero_crossings),
                                  'time_first_zero_cross': np.round(time_first_zero_c*1000),
                                  'time_last_zero_cross': np.round(time_last_zero_c*1000),
                                  'diff_after_before_peak':(r_end*(1000/fs)-pos_peak_t_ms)-(pos_peak_t_ms-r_start*(1000/fs)),
                                  'peak_position_ratio':(pos_peak_t_ms-r_start*(1000/fs))/r_duration,
                                  'mean ICI': mean,
                                  'std ICI': std,
                                  'CV ICI': cv,
                                  'bad_channel': bad_channel,
                                  'start_rec_ms': start_rec,
                                  'stop_rec_ms': stop_rec, 
                                  'date': date,
                                  'monkey': monkey,
                                  }
            else:  # if we have less than 2 zero crossings in the ripple
                num_zero_crossings = sign_changes.sum()
                dict_ripple_ch = {'channel_0_63': ch,
                                  'th_baseline': avg_type,
                                  'ICI time': np.nan,
                                  'freq': np.nan,
                                  'skew_env': np.nan,
                                  'first_peak_sig_val': np.nan,
                                  'time_0_sig':np.nan,
                                  'start_time_ms': r_start*(1000/fs),
                                  'stop_time_ms': r_end*(1000/fs),
                                  'eyes_closed': eyes_closed_r,
                                  'duration_ms': r_duration,
                                  'amplitude': amplitude,
                                  'peak_time_ms': (r_start+max_index)*(1000/fs), 
                                  'positive_peak_time_ms': (r_start+max_index_positive)*(1000/fs),
                                  'num_of_zero_crossings': int(num_zero_crossings),
                                  'time_first_zero_cross': np.nan,
                                  'time_last_zero_cross': np.nan,
                                  'diff_after_before_peak':np.nan,
                                  'peak_position_ratio':np.nan,
                                  'mean ICI': np.nan,
                                  'std ICI': np.nan,
                                  'CV ICI': np.nan,
                                  'bad_channel': bad_channel,
                                  'start_rec_ms': start_rec,
                                  'stop_rec_ms': stop_rec, 
                                  'date': date,
                                  'monkey': monkey,
                                  }
            list_ripples.append(dict_ripple_ch)
    df = pd.DataFrame(list_ripples)
    return df


def load_ripples_df(monkey_list,dual_th,date=None,area=None,condition='RS',params={},df_folder='',exclude_noisy=True,verbose=False):
    """
    Loading ripple dataframes for a given monkey list, area and condition, in one concatenated DF.
    """
    df_list = []
    for monkey in monkey_list:
        if verbose:
            print(monkey)
        if date is None:
            all_dates = params['dates'][monkey][condition]
        else:
            all_dates = [date]
        for aux_date in all_dates:
            if verbose:
                print(aux_date)
            if area is not None:
                if area in ['V4','IT','V1','V2']:
                    all_arrays = np.where(np.array(params['areas'][monkey])==area)[0] + 1  # shifting indexing from 0 to 1
                if area in ['V12']:
                    all_arrays = np.where(np.isin(np.array(params['areas'][monkey]),['V1','V2']))[0] + 1 
            else:
                all_arrays = range(1,17)
            for array in all_arrays:
                if monkey=='N' and array==5:
                    pass
                else:
                    folder_name = f'{df_folder}/{condition}_ripples/{monkey}/{aux_date}'               
                    df_ripp = pd.read_csv(f'{folder_name}/th__{int(dual_th[0]*10)}_{int(dual_th[1]*10)}_{monkey}_{aux_date}_arr{array}_ripples.csv')
                    df_ripp['area'] = params['areas'][monkey][array-1]
                    df_ripp['array'] = array
                    df_list.append(df_ripp)
    df_merged = pd.concat(df_list,ignore_index=True)
    
    if exclude_noisy:
        df_merged = df_merged[~df_merged['bad_channel']]
    
    return df_merged


def find_classes_cells(spike_block,sua_df):
    """
    For a spike block, returns the list of classes corresponding to final class of each of the cells.
    """
    num_cells = len(spike_block.segments[0].spiketrains)
    cl_list = []
    for idx in range(num_cells):
        sp_train = spike_block.segments[0].spiketrains[idx]
        cell_name = sp_train.annotations['nix_name']
        df_cell = sua_df[sua_df['cell_name']==cell_name]
        cl_list.append(df_cell['final_class'])
    return cl_list


def find_channels_cells(spike_block,sua_df):
    """
    For a spike block, returns the list of channel corresponding to index in RB or LFP file.
    """
    num_cells = len(spike_block.segments[0].spiketrains)
    cl_list = []
    for idx in range(num_cells):
        sp_train = spike_block.segments[0].spiketrains[idx]
        cell_name = sp_train.annotations['nix_name']
        df_cell = sua_df[sua_df['cell_name']==cell_name]
        cl_list.append(df_cell['channel_order'])
    return cl_list





