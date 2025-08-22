from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

##### PARAMETERS #####
with open("/CSNG/studekat/ripple_band_project/code/params_analysis.yml") as f:
    params = yaml.safe_load(f)

DATA_FOLDER = params['data_folder'] ### folder with all the preprocessed data
DATES = params['dates']
MAIN_FOLDER = '/CSNG/studekat/ripple_band_project/'

DF_FOLDER = '/CSNG/studekat/ripple_band_project/dataframes' ### here the resulting dataframes will be saved
MONKEY_LIST = ['L','N','F','A']

create_EC_EO_indicators = True # creates dictionaries with EC, EO indicators, and saves them
calculate_SUA_prop = True  # computationaly expensive, calculates all phase properties and saves pkl
calculate_other_prop = True  # modifies pkl by adding more properties

WIDTH_INTERVALS = params['width_intervals']
FINAL_CLASSES = params['final_classes']
PEAK_HEIGHT = params['first_peak_height']

#### NOT YET CHOSEN FOR GOOD, NEED TO DECIDE ON ARGUMENTS TO FIX THIS
SEL_TH = 0.06  # threshold for selectivity of cells, based on the histogram of all selectivities


##### CREATING INDICATORS #####
if create_EC_EO_indicators:
    print('Calculating EC, EO indicators.')
    for monkey in MONKEY_LIST:
        for date in DATES[monkey]['RS']:
            df_eyes = pd.read_csv(f'{DATA_FOLDER}/macaque{monkey}_RS_{date}/epochs_macaque{monkey}_RS_{date}.csv')
            duration_rec = params['times_all_arr'][monkey]['RS'][date][1] - params['times_all_arr'][monkey]['RS'][date][0] ### duration when all arrays are on
            ### loading indicator for the whole duration of array 1 rec time, sometimes not even that time is covered, a bit shorter
            EC_indic = create_indicator(df_eyes,start_col='t_start',stop_col='t_stop',state_col='state',
                                 positive_state='Closed_eyes',mult_factor=1000)
            EO_indic = create_indicator(df_eyes,start_col='t_start',stop_col='t_stop',state_col='state',
                                 positive_state='Open_eyes',mult_factor=1000)
            ### cutting only the common time
            cut_EC_indic = np.zeros(duration_rec)
            cut_EC_indic[:EC_indic.shape[0]] = EC_indic[:duration_rec]
            cut_EO_indic = np.zeros(duration_rec)
            cut_EO_indic[:EO_indic.shape[0]] = EO_indic[:duration_rec] 
    
            ### saving as a dictionary
            eyes_indic_dict = {}
            eyes_indic_dict['EC'] = cut_EC_indic
            eyes_indic_dict['EO'] = cut_EO_indic
    
            name = f'eyes_indic_monkey_{monkey}_RS_date_{date}_common_times'
            ensure_dir_exists(f'{MAIN_FOLDER}/metadata/EC_EO_indicators/')
            with open(f'{MAIN_FOLDER}/metadata/EC_EO_indicators/{name}.pkl', 'wb') as file:
                pickle.dump(eyes_indic_dict, file)


##### CALCULATING MAIN SUA PROPERTIES DATAFRAMES #####
if calculate_SUA_prop:
    print('Calculating SUA properties, main dataframes.')
    for monkey in MONKEY_LIST:
        print(monkey)
        for date in params['dates'][monkey]['RS']:
            print(date)
            prop_list = []
            for array in range(1,17): 
                print(array)
                try:
                    try:
                        spike_block = load_block(monkey,array,type_rec='RS',type_sig='spikes',date=date,data_folder=DATA_FOLDER)  # SUA
                        RB_block = load_block(monkey,array,type_rec='RS',type_sig='RB',date=date,data_folder=DATA_FOLDER)  # Ripple band
                        LFP_block = load_block(monkey,array,type_rec='RS',type_sig='LFP',date=date,data_folder=DATA_FOLDER)  # LFP
                        num_cells = len(spike_block.segments[0].spiketrains)
                        start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.magnitude)*1000))
                        start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
                        start_t_LFP_ms = int(np.floor(np.float64(LFP_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
                        print(f'Start t RB: {start_t_RB_ms}')
                        print(f'Start t spikes: {start_t_spikes_ms}')
                        print(f'Start t LFP: {start_t_LFP_ms}')
                        if start_t_spikes_ms!=start_t_RB_ms:
                            print('Spikes and ripples do not have the same start time.')
                    except:
                        print(f'Cannot read the spike file for date {date}, monkey {monkey}, array {array}.')
                    try:
                        df_OP = pd.read_csv(f'{DATA_FOLDER}/metadata/OP_maps_dataframes/{monkey}/OP_prop_OG_array{array}.csv')
                    except:
                        print(f'Cannot read OP maps for date {date}, monkey {monkey}, array {array}.')
                    
                    for cell in range(num_cells):
                        spike_train = spike_block.segments[0].spiketrains[cell]
                        cell_name = spike_train.annotations['nix_name']
                        electrode_ID = spike_train.annotations['Electrode_ID']
                        
                        ### channel prop - additional info for a channel, such as OP, bad channel ID, array and area
                        channel_prop = {}
                        channel_prop['cell_name'] = cell_name
                        ### OP
                        try:
                            ch_OP = df_OP[df_OP['Electrode_ID']==electrode_ID]
                            if ch_OP['selectivity_01'].values[0]>0.2 and ch_OP['num_f0_high_jump'].values[0]<3:
                                channel_prop['pref_OP'] = ch_OP['pref_OP'].values[0]
                                channel_prop['selectivity_OP_01'] = ch_OP['selectivity_01'].values[0]
                            else:
                                channel_prop['pref_OP'] = np.nan
                                channel_prop['selectivity_OP_01'] = ch_OP['selectivity_01'].values[0]
                        except:
                            channel_prop['pref_OP'] = np.nan
                            channel_prop['norm_selectivity_OP'] = np.nan
                        ### channel order
                        ch = aux_electrodeID_to_ch_order(monkey,date,electrode_ID,array,data_folder=DATA_FOLDER,type_rec='RS')
                        channel_prop['channel_order'] = ch
                        ### array
                        channel_prop['array'] = array
                        ### area
                        if monkey in ['N','F']:
                            name_area = 'Area'
                        else:
                            name_area = 'cortical_area'
                        ch_area = spike_train.annotations[name_area]
                        channel_prop['area'] = ch_area
                        channel_prop['train_order'] = cell # order in the spike train

                        avg_waveform = np.mean(spike_train.waveforms,axis=0)
                        channel_prop['avg_wf'] = avg_waveform

                        rb_sig_arr = sig_block_to_arr(RB_block,'RB_filtered_zsc')
                        LFP_sig_arr = sig_block_to_arr(LFP_block,'LFP_zsc')
                        rb_phase_arr = sig_block_to_arr(RB_block,'RB_phase')
                        rb_envelope_arr = sig_block_to_arr(RB_block,'RB_envelope_norm')
                        rb_env_phase_arr = sig_block_to_arr(RB_block,'RB_envelope_phase')

                        spike_arr = spike_block_to_arr(spike_block)

                        # cutting out common times only for N and F
                        if monkey in ['N','F']:
                            rb_sig_arr = cut_abs_times(rb_sig_arr,start_t_RB_ms,monkey,rec_type='RS',date=date,params=params)
                            LFP_sig_arr = cut_abs_times(LFP_sig_arr,start_t_LFP_ms,monkey,rec_type='RS',date=date,params=params)
                            rb_phase_arr = cut_abs_times(rb_phase_arr,start_t_RB_ms,monkey,rec_type='RS',date=date,params=params)
                            rb_envelope_arr = cut_abs_times(rb_envelope_arr,start_t_RB_ms,monkey,rec_type='RS',date=date,params=params)
                            rb_env_phase_arr = cut_abs_times(rb_env_phase_arr,start_t_RB_ms,monkey,rec_type='RS',date=date,params=params)
                            
                            spike_arr = cut_abs_times(spike_arr,start_t_spikes_ms,monkey,rec_type='RS',date=date,params=params)

                        rb_sig = rb_sig_arr[ch,:]
                        LFP_sig = LFP_sig_arr[ch,:]
                        rb_phase = rb_phase_arr[ch,:]
                        rb_envelope = rb_envelope_arr[ch,:]
                        rb_env_phase = rb_env_phase_arr[ch,:]
                        
                        spike_vector = spike_arr[cell,:]

                        # EC and EO properties
                        file_path = f'{MAIN_FOLDER}/metadata/EC_EO_indicators/eyes_indic_monkey_{monkey}_RS_date_{date}_common_times.pkl'
                        with open(file_path, 'rb') as file:
                            eyes_dict = pickle.load(file)

                        EC_indic = eyes_dict['EC']
                        EO_indic = eyes_dict['EO']
                        EC_dict = spike_train_prop_vec(spike_vector,rb_sig,LFP_sig,rb_phase,rb_envelope,rb_env_phase,channel_prop=None,
                                                       indicator=EC_indic,indicator_name='EC')
                        EO_dict = spike_train_prop_vec(spike_vector,rb_sig,LFP_sig,rb_phase,rb_envelope,rb_env_phase,channel_prop=None,
                                                       indicator=EO_indic,indicator_name='EO')

                        # adding EC and EO properties to the channel_prop dict.
                        for k in EC_dict.keys():
                            channel_prop[k] = EC_dict[k]
                        for k in EO_dict.keys():
                            channel_prop[k] = EO_dict[k]

                        # whole rec. time properties
                        prop_dict = spike_train_prop_vec(spike_vector,rb_sig,LFP_sig,rb_phase,rb_envelope,rb_env_phase,channel_prop=channel_prop,
                                                        indicator=None,indicator_name=None) ### input already binned spikes
                        prop_list.append(prop_dict)
                except:
                    print(f'For array {array}, the SUA properties were not calculated.')
            df_prop = pd.DataFrame(prop_list)
            ensure_dir_exists(f'{DF_FOLDER}/sua_prop/')
            df_prop.to_pickle(f'{DF_FOLDER}/sua_prop/monkey{monkey}_all_arrays_date_{date}.pkl')

if calculate_other_prop:
    print('Calculating SUA properties - additional modification of main dataframes.')
    for monkey in MONKEY_LIST: 
        print(monkey)
        all_RS_dates = params['dates'][monkey]['RS']
        for date in all_RS_dates:
            print(date)
            with open(f'{DF_FOLDER}/sua_prop/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
                df_sua = pickle.load(file)
            df_added = aux_add_waveform_prop(df_sua)
            df_added = aux_add_zscored_avg_waveform(df_added)
            df_added = df_added[df_added['channel_order']>-1] ### erasing not working arrays
            df_added = aux_add_width_classes(df_added,width_intervals=WIDTH_INTERVALS)
            df_added = aux_add_up_down_classes(df_added)
            df_added = aux_add_final_classes(df_added,final_classes=FINAL_CLASSES,peak_height_th=PEAK_HEIGHT)
            df_added = aux_add_selectivity(df_added,sel_th=SEL_TH)
    
            #### saving new dataframes with properties as pickle
            ensure_dir_exists(f'{DF_FOLDER}/sua_prop_all/')
            df_added.to_pickle(f'{DF_FOLDER}/sua_prop_all/monkey{monkey}_all_arrays_date_{date}.pkl')
            ### the copy warning is there only for the case of empty arrays, no worries about it




