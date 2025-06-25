#!/home/morales/miniconda3/envs/amsterdam/bin/python
"""
Script to Spike Sort signals
"""
import os
import multiprocessing

# Set environment variables to use all cores
tot_num_cores = multiprocessing.cpu_count()
jobs = 16  # Number of parallel sorting jobs
num_cores = 64//jobs  # Cores per parallel process (usually 4 is good)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

import numpy as np
import pandas as pd
from scipy.linalg import svd
from elephant.signal_processing import butter
import probeinterface as pi
import spikeinterface as si
from spikeinterface.core import NumpyRecording
from spikeinterface.sorters import run_sorter
from spikeinterface import create_sorting_analyzer
import spikeinterface.qualitymetrics as qm
import spikeinterface.curation as scur
import neo
from utils import get_aligned_times, get_metadata, load_sig
import quantities as pq
from datetime import datetime
from joblib import Parallel, delayed

si.core.set_global_job_kwargs(n_jobs=1/jobs)

# Number of channels processed together
# Must be an integer divisor of the total number of channels!
chunk_size = 4

def compute_task(recording, out_dir):
    try:
        # When using few electrodes drift correction causes problems!
        sorting = run_sorter(sorter_name="kilosort4",
                             recording=recording,
                             folder=out_dir,
                             remove_existing_folder=True,
                             batch_size=30000*10,
                             whitening_range=chunk_size,
                             do_CAR=False,  # Deactivates median subtraction
                             Th_universal=7, ### originaly: 7
                             Th_learned=6, ### originaly: 6, we test 7,8
                             dmin=200,  # Y width of templates in um
                             dminx=200, # X width of templates in um
                             nearest_chans=np.min([chunk_size, 10]),
                             skip_kilosort_preprocessing=True,
                             verbose=False)
        print('Spike sorting completed')
    except Exception as e:
        print(e)
        # print(f'Probably no units found in channels {i*chunk_size} to {(i+1)*chunk_size}')
        sorting = None
    return sorting

if __name__ == '__main__':

    # Get inputs from snakemake
    ns6_file = snakemake.input.raw
    tracker_file = snakemake.output.tracker

    # Load data
    # Ingnoring the nev file makes the IO much faster!
    io = neo.BlackrockIO(ns6_file, load_nev=False)
    block = io.read_block(lazy=True)

    # Some cases have multiple segments, which need to be concatenated
    anasig_proxies = [seg.analogsignals[0] for seg in block.segments]
    sampling_rate = anasig_proxies[0].sampling_rate
    channels = anasig_proxies[0].array_annotations['channel_ids']
    print('Proxy loaded')

    # Get the aligned session times and offset
    alignment_offset, t_start, t_stop = get_aligned_times(snakemake)

    # Band-passed signal (iteration needed to not overflow memory)
    # Minimum memory needed: roughly 6 times the file size
    filtered_sigs = []
    for idx in range(np.min(anasig_proxies[-1].shape)):

        # Load signal
        sig = load_sig(anasig_proxies, idx, alignment_offset, t_stop)

        # Filtering
        sig = butter(sig,
                     lowpass_frequency=6000,
                     highpass_frequency=500,
                     sampling_frequency=sampling_rate)

        filtered_sigs.append(sig)
    anasig = np.stack(filtered_sigs).astype(np.float32)
    del filtered_sigs  # Release memory
    # Ensure the correct shape: (channels, timepoints)
    if anasig.shape[0] > anasig.shape[1]:
        anasig = anasig.T
    print('Signals loaded and band-pass filtered between 500 - 6000 Hz')

    # ZCA whitening
    covariance = np.cov(anasig)
    EPSILON = 1e-5
    U, S, _ = svd(covariance)  # Singular Value Decomposition
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + EPSILON)), U.T))
    anasig = np.dot(anasig.T, zca_matrix).T
    anasig = anasig.astype(np.float32)
    # Ensure the correct shape: (channels, timepoints)
    if anasig.shape[0] > anasig.shape[1]:
        anasig = anasig.T
    print('Signal whitened')

    # ---- RETRIEVE ELECTRODE METADATA ----
    metadata = get_metadata(snakemake, channels)

    # Get data chunks
    tot_channels = anasig.shape[0]
    recordings = []
    out_dirs = []
    for i in range(tot_channels//chunk_size):
        chunk_signal = anasig[i*chunk_size:(i+1)*chunk_size].astype(np.float32)
        num_channels, num_samples = chunk_signal.shape

        # Electrode metadata
        x_coords = metadata['schematic_X_position'].values[i*chunk_size:(i+1)*chunk_size]
        y_coords = metadata['schematic_Y_position'].values[i*chunk_size:(i+1)*chunk_size]
        electrode_positions = np.column_stack((x_coords, y_coords))  # Shape (num_channels, 2)

        # ---- CREATE PROBE BASED ON METADATA ----
        probe = pi.Probe(ndim=2)
        probe.set_contacts(electrode_positions)
        probe.set_device_channel_indices(np.arange(num_channels))
        probe.create_auto_shape()

        # ---- CREATE SPIKEINTERFACE RECORDING ----
        recording = NumpyRecording(traces_list=[chunk_signal.T], sampling_frequency=sampling_rate.magnitude)
        recording.set_probe(probe, in_place=True)

        # ---- RUN SPIKE SORTING ----
        data_dir = tracker_file.replace('.txt', f'_kilosort4/ch{i*chunk_size}-{(i+1)*chunk_size}/')
        out_dir = tracker_file.replace('.txt', f'_kilosort4/ch{i*chunk_size}-{(i+1)*chunk_size}/results/')
        recording = recording.save(folder=data_dir,
                                   overwrite=True,
                                   format="binary")
        recordings.append(recording)
        out_dirs.append(out_dir)

    # Run multiple sortings at once
    sortings = Parallel(n_jobs=jobs)(delayed(compute_task)(recording, out_dir) for recording, out_dir in zip(recordings, out_dirs))

    # Create spiketrains
    spiketrains = []
    for i, sorting, recording in zip(range(tot_channels//chunk_size), sortings, recordings):

        if sorting is None:
            continue

        # Remove potential spikes after recording ends
        clean_sorting = scur.remove_excess_spikes(sorting, recording)

        # ---- WAVEFORMS AND QUALITY METRICS ----
        sorting_analyzer = create_sorting_analyzer(
            sorting=clean_sorting,
            recording=recording,
            format="memory",
            sparse=False,  # Uses all waveforms
            compute_kwargs={"waveforms": {"max_spikes_per_unit": None}},
            progress_bar=True)

        # Calculate postprocessing quality metrics
        sorting_analyzer.compute("random_spikes", max_spikes_per_unit=np.inf)
        sorting_analyzer.compute("waveforms")
        sorting_analyzer.compute("templates")
        sorting_analyzer.compute("noise_levels")
        sorting_analyzer.compute("spike_amplitudes")
        sorting_analyzer.compute("template_metrics")
        sorting_analyzer.compute("spike_locations")
        metrics = qm.compute_quality_metrics(
            sorting_analyzer,
            qm_params={"presence_ratio": {"bin_duration_s": 10},
                       "sliding_rp_violation": {"bin_size_ms": 0.1}})
        sorting_analyzer.compute("unit_locations")
        unit_locations_ext = sorting_analyzer.get_extension("unit_locations")
        unit_locations = unit_locations_ext.get_data()
        extremum_channels = si.core.get_template_extremum_channel(sorting_analyzer, peak_sign='neg')

        # For waveform extraction
        waveform_extension = sorting_analyzer.get_extension("waveforms")
        waveforms = waveform_extension.get_data()
        channel_ids = sorting_analyzer.recording.get_channel_ids()
        unit_indices = clean_sorting.to_spike_vector()['unit_index']

        # ---- CREATE A LIST OF NEO SPIKETRAINS ----
        if ('macaqueN' in tracker_file) or ('macaqueF' in tracker_file):
            subset_metadata = metadata.set_index("instanceChannels_512", drop=False)
        elif ('macaqueL' in tracker_file) or ('macaqueA' in tracker_file):
            subset_metadata = metadata.set_index("within_NSP_electrode_ID", drop=False)
        subset_metadata = subset_metadata.iloc[i*chunk_size:(i+1)*chunk_size]
        subset_metadata.index = range(chunk_size)
        channel_metadata_dict = subset_metadata.to_dict(orient="index")
        for idx, unit_id in enumerate(clean_sorting.unit_ids):

            # Create a Neo SpikeTrain object
            spike_times = clean_sorting.get_unit_spike_train(unit_id) / sorting_analyzer.sampling_frequency
            spiketrain = neo.SpikeTrain(
                times=spike_times * pq.s + t_start,
                t_start=t_start,
                t_stop=t_stop,
                name=f"Unit {unit_id}")

            # Annotations
            annotations = metrics.loc[unit_id].to_dict()
            extremum_channel = extremum_channels[unit_id]
            annotations.update(channel_metadata_dict[extremum_channel])
            spiketrain.annotations.update(annotations)

            # Check that annotations are not illegal values (None, arrays, lists)
            for k in spiketrain.annotations.keys():
                if isinstance(spiketrain.annotations[k], (list, np.ndarray)):
                    spiketrain.annotations[k] = str(spiketrain.annotations[k])
                elif spiketrain.annotations[k] != spiketrain.annotations[k]:
                    spiketrain.annotations[k] = 'NaN'
                elif spiketrain.annotations[k] is None:
                    spiketrain.annotations[k] = 'None'

            # Waveforms
            ch_idx = np.where(np.array(channel_ids) == extremum_channel)[0][0]
            spiketrain.waveforms = waveforms[unit_indices == idx, :, ch_idx] * pq.uV

            # Keep
            spiketrains.append(spiketrain)
        print('Metadata annotations created')

    # ---- GET UNIQUE ELECTRODE ARRAYS ----
    arrays = [st.annotations['Array_ID'] for st in spiketrains]
    date = datetime.today().strftime('%Y-%m-%d')
    for arr_id in np.unique(arrays):
        # ---- CREATE A NEW NEO BLOCK FOR THIS ARRAY ----
        block = neo.Block(
            name=f"Spike Sorting Results - Array {arr_id} - Source file {ns6_file}",
            author="Aitor Morales-Gregorio",
            comment="Spike sorting data with full metadata",
            array=arr_id,
            date_of_creation=date
        )

        # ---- CREATE A SEGMENT ----
        segment = neo.Segment(name=f"Array {arr_id} Segment")
        block.segments.append(segment)

        # ---- FILTER SPIKETRAINS FOR THIS ARRAY ----
        array_spiketrains = [st for st in spiketrains if st.annotations["Array_ID"] == arr_id]

        if array_spiketrains:  # Prevents empty list issues
            segment.spiketrains.extend(array_spiketrains)
        if len(array_spiketrains) == 0:
            # skip if empty spiketrains
            continue

        # ---- CREATE OUTPUT FILE PATH ----
        out_file = tracker_file.replace(".txt", "_unfiltered.nix")
        instance = int(snakemake.wildcards.instance)
        if "macaqueL" in tracker_file or "macaqueA" in tracker_file or 'macaqueN_TVSD' in tracker_file:
            out_file = out_file.replace(f"instance{instance}", f"Array{arr_id}")
        elif "macaqueN" in tracker_file or "macaqueF" in tracker_file:
            hub = int(snakemake.wildcards.hub)
            out_file = out_file.replace(f"Hub{hub}-instance{instance}", f"Array{arr_id}")

        # ---- SAVE TO NIX FILE ----
        print(f"Saving NIX file for Array {arr_id}...\n")
        nix_io = neo.NixIO(out_file, mode="ow")
        nix_io.write(block)
        nix_io.close()

        # Thresholds on unit metrics
        thresholds = {
            'firing_rate_thr': 1,
            'waveform_snr_thr': 3,
            'isi_violations_thr': 0.9,
            'presence_thr': 0.5
        }

        # Filter units
        clean_spiketrains = []
        for st in array_spiketrains:
            # If the FR is less than 1 some of the other metrics could fail to be defined
            if st.annotations['firing_rate'] > thresholds['firing_rate_thr']:
                if st.annotations['snr'] > thresholds['waveform_snr_thr'] and \
                st.annotations['isi_violations_ratio'] < thresholds['isi_violations_thr'] and \
                st.annotations['presence_ratio'] > thresholds['presence_thr']:
                    clean_spiketrains.append(st)

        if len(clean_spiketrains) == 0:
            # skip if empty spiketrains
            continue

        # ---- CREATE A NEW NEO BLOCK ----
        block = neo.Block(
            name=f"Spike Sorting Results - Array {arr_id} - Source file {ns6_file}",
            author="Aitor Morales-Gregorio",
            comment="Spike sorting data with full metadata, only 'clean' units kept",
            date_of_creation=date,
            **thresholds)

        # Append spiketrains to block
        segment = neo.Segment()
        block.segments.append(segment)
        segment.spiketrains.extend(clean_spiketrains)

        # ---- SAVE TO NIX FILE ----
        filtered_file = out_file.replace("_unfiltered.nix", ".nix")
        nix_io = neo.NixIO(filtered_file, mode="ow")
        nix_io.write(block)
        nix_io.close()

        # ---- BLOCK WITHOUT WAVEFORMS ----
        block = neo.Block(
            name=f"Spike Sorting Results - Array {arr_id} - Source file {ns6_file}",
            author="Aitor Morales-Gregorio",
            comment="Spike sorting data with full metadata, only 'clean' units kept, no waveforms",
            date_of_creation=date,
            **thresholds)

        # Remove waveforms and append spiketrains to block
        for st in clean_spiketrains:
            st.waveforms = None
        segment = neo.Segment()
        block.segments.append(segment)
        segment.spiketrains.extend(clean_spiketrains)

        # ---- SAVE TO NIX FILE ----
        filtered_file = out_file.replace("_unfiltered.nix", "_filtered_noWaveforms.nix")
        nix_io = neo.NixIO(filtered_file, mode="ow")
        nix_io.write(block)
        nix_io.close()

    print("Spike sorting results saved for all arrays.")

    # ---- CREATE TRACKER FILE ----
    with open(tracker_file, "w", encoding="utf-8") as tfile:
        tfile.write(f"Spike sorting completed for arrays {metadata['Array_ID'].unique().tolist()}\n")
        tfile.write(f"Originating from {ns6_file} completed on {date}\n")
