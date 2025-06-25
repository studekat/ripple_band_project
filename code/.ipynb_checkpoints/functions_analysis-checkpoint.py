##### FUNCTIONS FOR ANALYSIS OF THE MACAQUE RIPPLE BAND #####

import numpy as np
import pandas as pd
import elephant
from scipy.signal import hilbert
import neo
from scipy.stats import zscore

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
        vec = arr[np.newaxis,:]
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
    """
    bin_vals, border_bins = np.histogram(data_vec,bins=bins,range=x_range)
    centre_bins = np.diff(border_bins)/2+border_bins[:-1] ### move from the left edge of the bin to its centre
    bin_vals = bin_vals/np.sum(bin_vals) ### normalising histogram to density
    r, phi = event_to_vec_weighted(angles=centre_bins,weights=bin_vals)
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


















