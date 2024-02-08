# IMPORT packages
import os
from datetime import datetime
from pathlib import Path
from pdb import set_trace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subplots
import torch
from scipy import signal
from scipy.io import loadmat

from MUsim import MUsim

start_time = datetime.now()  # begin timer for script execution time


# define a function to convert a timedelta object to a pretty string representation
def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


# define a lowpass filter
def butter_lowpass(lowcut, fs, order=2):
    """
    Butterworth lowpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype="low")
    return b, a


# define a function to filter data with lowpass filter
def butter_lowpass_filter(data, lowcut, fs, order=2):
    """
    Butterworth lowpass filter.
    """
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# define a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


# define a function to filter data with bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Butterworth bandpass filter.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def sigmoid(x):
    """
    Sigmoid function.
    """
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


# function to blend arrays at the edges
def blend_arrays(array_list, Nsamp):
    """
    Blend arrays together overlapping at the edges using a sigmoid function.
    """
    if len(array_list) < 2:  # if only one array, return the unaltered array
        return array_list[0]
    blender_array = sigmoid(np.linspace(-12, 12, Nsamp))
    blended_array = np.zeros(
        len(np.concatenate(array_list)) - Nsamp * (len(array_list) - 1)
    )
    len_prev_arrays = 0
    for ii, iArray in enumerate(array_list):
        len_this_array = len(iArray)
        if iArray is array_list[0]:
            iArray[-Nsamp:] *= (
                1 - blender_array
            )  # only blend the first array at the end
            backward_shift = 0
        elif iArray is array_list[-1]:
            iArray[
                :Nsamp
            ] *= blender_array  # only blend the last array at the beginning
            backward_shift = Nsamp * ii
        else:  # blend both ends of the array for all other arrays
            iArray[:Nsamp] *= blender_array
            iArray[-Nsamp:] *= 1 - blender_array
            backward_shift = Nsamp * ii
        blended_array[
            len_prev_arrays
            - backward_shift : len_prev_arrays
            - backward_shift
            + len_this_array
        ] += iArray
        len_prev_arrays += len_this_array
    return blended_array


## MUsim class initialization and simulation function
def batch_run_MUsim(mu, force_profile, proc_num):
    """
    Initialize an MUsim object, set number of units, and apply the inverted filtered array as the force profile
    """
    mu.apply_new_force(force_profile)
    mu.simulate_spikes()
    print(f"({proc_num}) MU Thresholds " + str(mu.units[0]))
    print(f"({proc_num}) MU Poisson Lambdas " + str(mu.units[2]))
    return mu


# set analysis parameters
show_plotly_figures = False
show_matplotlib_figures = False
show_final_plotly_figure = False
save_final_plotly_figure = False
save_simulated_spikes = True
save_continuous_dat = True
multiprocess = True  # set to True to run on multiple processes
use_KS_templates = False  # set to True to use Kilosort templates to create waveform shapes, else load open ephys data, and use spike times to extract median waveform shapes for each unit
shift_MU_templates_along_channels = False
kinematics_fs = 125
ephys_fs = 30000
nt0 = 61  # 2.033 ms
SVD_dim = 9  # number of SVD components than were used in KiloSort
num_chans_in_recording = 9  # number of channels in the recording
num_chans_in_output = 8  # desired real number of channels in the output data
# number determines noise power added to channels (e.g. 50), or set None to disable
SNR_mode = "constant"  # 'power' to compute desired SNR with power,'from_data' simulates from the real data values, or 'constant' to add a constant amount of noise to all channels
# target SNR value if "power", or factor to adjust SNR by if "from_data", or set None to disable
adjust_SNR = 100  # None
# set 0 for no shape jitter, or a positive number for standard deviations of additive shape jitter
shape_jitter_amount = 0
# set None for random behavior, or a previous entropy int value to reproduce
random_seed_entropy = 218530072159092100005306709809425040261  # 218530072159092100005306709809425040261  # 75092699954400878964964014863999053929  # None
if random_seed_entropy is None:
    random_seed_entropy = np.random.SeedSequence().entropy
RNG = np.random.default_rng(random_seed_entropy)  # create a random number generator

# add eventplot of spike times to the last subplot, vertically spacing times and coloring by unit
MU_colors = [
    "royalblue",
    "firebrick",
    "forestgreen",
    "darkorange",
    "darkorchid",
    "darkgreen",
    "lightcoral",
    "rgb(116, 77, 37)",
    "cyan",
    "mediumpurple",
    "lightslategray",
    "gold",
    "lightpink",
    "darkturquoise",
    "darkkhaki",
    "darkviolet",
    "darkslategray",
    "darkgoldenrod",
    "darkmagenta",
    "darkcyan",
    "darkred",
    "darkblue",
    "darkslateblue",
    "darkolivegreen",
    "darkgray",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "maroon",
    "mediumblue",
    "mediumorchid",
    "mediumseagreen",
    "mediumslateblue",
    "mediumturquoise",
    "magenta",
    "forestgreen",
    "mediumvioletred",
    "midnightblue",
    "navy",
    "olive",
    "olivedrab",
]

# set plotting parameters
time_frame = [0, 1]  # time frame to plot, fractional bounds of 0 to 1
plot_template = "plotly_white"

# check inputs
assert (
    type(kinematics_fs) is int and kinematics_fs > 0
), "kinematics_fs must be a positive integer"
assert type(ephys_fs) is int and ephys_fs > 0, "ephys_fs must be a positive integer"
assert (
    type(shape_jitter_amount) in [int, float] and shape_jitter_amount >= 0
), "shape_jitter_amount must be a number greater than or equal to 0"
assert (type(adjust_SNR) in [int, float] and adjust_SNR >= 0) or (
    adjust_SNR is None
), "adjust_SNR must be a positive number or None"
assert time_frame[0] >= 0 and time_frame[1] <= 1 and time_frame[0] < time_frame[1], (
    "time_frame must be a list of two numbers between 0 and 1, "
    "with the first number smaller"
)


## first load a 1D kinematic array from a csv file into a numpy array
# format is YYYYMMDD-N, with N being the session number
anipose_sessions_to_load = [
    "20221116-3",
    "20221116-5",
    "20221116-7",
    "20221116-8",
    "20221116-9",
    "20221117-4",
    "20221117-5",
    "20221117-6",
    "20221117-8",
    "20221117-9",
]
# shuffle the list of sessions to load so different simulations have different kinematics
RNG.shuffle(
    anipose_sessions_to_load
)  # rat was not shuffled, monkey was shuffled once, any other should be shuffled N times
# format is bodypart_side_axis, with side being L or R, and axis being x, y, or z
chosen_bodypart_to_load = "palm_L_y"  # "wrist_L_y"
reference_bodypart_to_load = "tailbase_y"
# choose path to get anipose data from
anipose_folder = Path("/snel/share/data/anipose/analysis20230830_godzilla/")
# anipose_folder = Path("/snel/share/data/anipose/analysis20230829_inkblot+kitkat")

# load the csv file into a pandas dataframe and get numpy array of chosen bodypart
kinematic_csv_folder_path = anipose_folder.joinpath("pose-3d")
files = [
    f for f in kinematic_csv_folder_path.iterdir() if f.is_file() and f.suffix == ".csv"
]
kinematic_csv_file_paths = [
    f for f in files if any(s in f.name for s in anipose_sessions_to_load)
]
print(f"Taking kinematic data from: \n{[f.name for f in kinematic_csv_file_paths]}")
kinematic_dataframes = [pd.read_csv(f) for f in kinematic_csv_file_paths]
chosen_bodypart_arrays = [
    kinematic_dataframe[chosen_bodypart_to_load].to_numpy()
    for kinematic_dataframe in kinematic_dataframes
]
time_slices = [
    slice(
        round(len(arr) * time_frame[0]),
        round(len(arr) * time_frame[1]),
    )
    for arr in chosen_bodypart_arrays
]
chosen_bodypart_arrays = [
    chosen_bodypart_array[time_slice]
    for chosen_bodypart_array, time_slice in zip(chosen_bodypart_arrays, time_slices)
]

if show_plotly_figures:
    # plot each array with plotly express
    for iArray in chosen_bodypart_arrays:
        fig = px.line(iArray)
    # add title and axis labels
    fig.update_layout(
        title="Raw Kinematics-derived Force Profile",
        xaxis_title="Time",
        yaxis_title="Force Approximation",
    )
    fig.show()

## load, 2Hz lowpass, and subtract reference bodypart from chosen bodypart
reference_bodypart_arrays = [
    kinematic_dataframe[reference_bodypart_to_load].to_numpy()
    for kinematic_dataframe in kinematic_dataframes
]

# subtract filtered reference, lowpass filter, then combine all arrays into single arrays by using and overlap of N samples
# blend the arrays using a linear relationship at the edges
# lowpass filter the reference array
nyq = 0.5 * kinematics_fs
ref_lowcut = 2
ref_low = ref_lowcut / nyq
ref_order = 2
# bandpass filter parameters
lowcut = 1
highcut = 8
low = lowcut / nyq
high = highcut / nyq
order = 2
# apply filters
ref_filtered_array_list = []
ref_chosen_bodypart_array_list = []
inv_filtered_array_list = []
min_sub_array_list = []
Nsamp = 25  # this is the number of samples to blend at the edges
for ii in range(len(reference_bodypart_arrays)):
    iRef = reference_bodypart_arrays[ii]
    ref_filtered_array_list.append(
        butter_lowpass_filter(iRef, ref_lowcut, kinematics_fs, ref_order)
    )
    iSub = ref_filtered_array_list[ii][time_slices[ii]]
    # subtract reference bodypart from chosen bodypart
    ref_chosen_bodypart_array_list.append(chosen_bodypart_arrays[ii] - iSub)
    iFilt = ref_chosen_bodypart_array_list[ii]
    # invert the array to better simulate force from y kinematics
    inv_filtered_array_list.append(
        -butter_bandpass_filter(iFilt, lowcut, highcut, kinematics_fs, order)
    )

# blend the arrays together
blended_chosen_array = blend_arrays(inv_filtered_array_list, Nsamp)

if show_plotly_figures:
    fig = px.line(np.concatenate(ref_chosen_bodypart_array_list))
    # add title and axis labels
    fig.update_layout(
        title="Raw Kinematics-derived Force Profile, Reference Subtracted",
        xaxis_title="Time",
        yaxis_title="Force Approximation",
    )
    fig.show()

    # plot the filtered array with plotly express
    fig = px.line(np.concatenate(inv_filtered_array_list))
    # add title and axis labels
    fig.update_layout(
        title="Filtered and Inverted Kinematics-derived Force Profile",
        xaxis_title="Time",
        yaxis_title="Force Approximation",
    )
    fig.show()

    # plot the final array with plotly express
    fig = px.line(blended_chosen_array)
    # add title and axis labels
    fig.update_layout(
        title="Final Kinematics-derived Force Profile",
        xaxis_title="Time",
        yaxis_title="Force Approximation",
    )
    fig.show()

# interpolate final force array to match ephys sampling rate
interp_final_force_array = signal.resample(
    blended_chosen_array, round(len(blended_chosen_array) * (ephys_fs / kinematics_fs))
)

## load the spike history kernel csv's and plot them with plotly express to compare in 2 subplots
# load each csv file into a pandas dataframe
# MU_spike_history_kernel_path = Path(__file__).parent.joinpath("spike_history_kernel_basis_MU.csv")
orig_spike_history_kernel_path = Path(__file__).parent.joinpath(
    "spike_history_kernel_basis.csv"
)

# MU_spike_history_kernel_df = pd.read_csv(MU_spike_history_kernel_path)
orig_spike_history_kernel_df = pd.read_csv(orig_spike_history_kernel_path)

# if show_plotly_figures:
#     # now use plotly express to plot the two dataframes in two subplots
#     # make first subplot be the original spike history kernel,
#     # and second subplot be the MU spike history kernel
#     fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)

#     # add the original spike history kernel to the top subplot
#     for ii, iCol in enumerate(orig_spike_history_kernel_df.columns):
#         fig.add_trace(
#             go.Scatter(
#                 x=orig_spike_history_kernel_df.index / ephys_fs * 1000,
#                 y=orig_spike_history_kernel_df[iCol],
#                 name=f"Component {ii}",
#             ),
#             row=1,
#             col=1,
#         )

#     # add the MU spike history kernel to the bottom subplot
#     for ii, iCol in enumerate(MU_spike_history_kernel_df.columns):
#         fig.add_trace(
#             go.Scatter(
#                 x=MU_spike_history_kernel_df.index / ephys_fs * 1000,
#                 y=MU_spike_history_kernel_df[iCol],
#                 name=f"Component {ii}",
#             ),
#             row=2,
#             col=1,
#         )

#     # add title and axis labels, make sure x-axis title is only on bottom subplot
#     fig.update_layout(
#         title="<b>Spike History Kernel Bases</b>",
#         template=plot_template,
#     )
#     fig.update_yaxes(title_text="Original Kernel Values", row=1, col=1)
#     fig.update_yaxes(title_text="Motor Unit Kernel Values", row=2, col=1)
#     fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
#     fig.show()

## load proc.dat from a sort for each rat including 16 channels (zeroed-out channels replace noisy ones)
# paths to the folders containing the Kilosort data
paths_to_proc_dat = [
    # Path(
    #     "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/session20221117/2022-11-17_17-08-07_myo/sorted0_20231218_200926870049_rec-1,2,4,5,6"
    # ),
    # Path(
    #     "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/inkblot/session20230323/2023-03-23_14-41-46_myo/sorted0_20231218_202453598454_rec-3,5,7,8,9,10_Th,[10,4],spkTh,[-6]"
    # ),
    # Path(
    #     "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/kitkat/session20230420/2023-04-20_14-12-09_myo/sorted0_20231218_203140625455_rec-1,2,3,4,5,7,8,9,11,12,14,15,16,17,19,20,21"
    # ),
    Path(
        "/snel/share/data/rodent-ephys/open-ephys/monkey/sean-pipeline/session20231202/2022-12-02_10-14-45_myo/sorted0_20240131_172133542034_rec-1_11-good-of-20-total_Th,[10,4],spkTh,[-6]"
    ),
]

## load Kilosort data
# paths to the folders containing the Kilosort data
paths_to_KS_session_folders = [
    # Path(
    #     # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/session20221116/"
    #     "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/session20221117/"
    # ),
    # Path(
    #     "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/inkblot/session20230323/"
    # ),
    # Path(
    #     "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/kitkat/session20230420/"
    # ),
    Path(
        "/snel/share/data/rodent-ephys/open-ephys/monkey/sean-pipeline/session20231202/"
    ),
]
sorts_from_each_path_to_load = [
    # "20231027_163931",  # godzilla
    # "20231218_181442825759",  # inkblot
    # "20231214_104534576438",  # kitkat
    "20240131_172133542034",  # monkey
]  # ["20230924_151421"]  # , ["20230923_125645"], ["20230923_125645"]]

# find the folder name which ends in _myo and append to the paths_to_session_folders
paths_to_each_myo_folder = []
for iDir in paths_to_KS_session_folders:
    myo = [f for f in iDir.iterdir() if (f.is_dir() and f.name.endswith("_myo"))]
    assert len(myo) == 1, "There should only be one _myo folder in each session folder"
    paths_to_each_myo_folder.append(myo[0])
# inside each _myo folder, find the folder name which constains sort_from_each_path_to_load string
list_of_paths_to_sorted_folders = []
for iPath in paths_to_each_myo_folder:
    matches = [
        f
        for f in iPath.iterdir()
        if f.is_dir() and any(s in f.name for s in sorts_from_each_path_to_load)
    ]
    assert len(matches) == 1, (
        f"There needs to be one sort folder match in each _myo folder, but the number was: "
        f"{len(matches)}, for path {str(iPath)}"
    )
    list_of_paths_to_sorted_folders.append(matches[0])

rez_list = [
    loadmat(str(path_to_sorted_folder.joinpath("rez.mat")))
    for path_to_sorted_folder in list_of_paths_to_sorted_folders
]
ops_list = [
    loadmat(str(path_to_sorted_folder.joinpath("ops.mat")))
    for path_to_sorted_folder in list_of_paths_to_sorted_folders
]

spike_times_list = [
    np.load(str(path_to_sorted_folder.joinpath("spike_times.npy"))).flatten()
    for path_to_sorted_folder in list_of_paths_to_sorted_folders
]

spike_clusters_list = [
    np.load(str(path_to_sorted_folder.joinpath("spike_clusters.npy"))).flatten()
    for path_to_sorted_folder in list_of_paths_to_sorted_folders
]
# load and reshape into numchans x whatever (2d array) the data.bin file
ephys_data_list = [
    np.memmap(
        str(path_to_proc_folder.joinpath("proc.dat")),
        dtype="int16",
        mode="r",
    ).reshape(-1, num_chans_in_recording)
    for path_to_proc_folder in paths_to_proc_dat
]

chan_map_adj_list = [
    loadmat(str(path_to_sorted_folder.joinpath("chanMapAdjusted.mat")))
    for path_to_sorted_folder in list_of_paths_to_sorted_folders
]

amplitudes_df_list = [
    pd.read_csv(str(path_to_sorted_folder.joinpath("cluster_Amplitude.tsv")), sep="\t")
    for path_to_sorted_folder in list_of_paths_to_sorted_folders
]
# set index as the cluster_id column
amplitudes_df_list = amplitudes_df_list[0].set_index("cluster_id")

# list of lists of good clusters to take from each rez_list
# place units in order of total spike count, from highest to lowest
clusters_to_take_from = [
    # [26, 13, 10, 3, 22, 32, 1, 15, 40, 27],  # godzilla, 20231027_163931
    # [9, 7, 8, 13],  # [12, 8, 14, 1, 13],  # inkblot, 20231218_181442825759
    # [15, 52, 9, 20, 16, 5, 14, 23, 13, 8],  # kitkat, 20231214_104534576438
    [6, 13, 24, 1, 23, 14],  # monkey
]  # [[25, 3, 1, 5, 17, 18, 0, 22, 20, 30]]  # [[18, 2, 11, 0, 4, 10, 1, 9]]

num_motor_units = sum([len(i) for i in clusters_to_take_from])

# create N MUsim objects, and run them in parallel on N processes,
# one for each segment of the anipoise data
# initialize 1 MUsim object, then create identical copies of it for each process
mu = MUsim(random_seed_entropy)
mu.num_units = num_motor_units  # set same number of motor units as in the Kilosort data
mu.MUthresholds_dist = "exponential"  # set the distribution of motor unit thresholds
mu.MUspike_dynamics = "spike_history"
mu.sample_rate = ephys_fs  # 30000 Hz
# fixed minimum force threshold for the generated units' response curves. Tune this for lower
# bound of force thresholds sampled in the distribution of MUs during MU_sample()
mu.threshmin = np.percentile(interp_final_force_array, 40)
# fixed maximum force threshold for the generated units' response curves. Tune this for upper
# bound of force thresholds sampled in the distribution of MUs during MU_sample()
mu.threshmax = 2 * np.max(
    interp_final_force_array
)  # np.percentile(interp_final_force_array, 99)
mu.sample_MUs()

# now create N copies of the MUsim object
chunk_size = 7500 / 2  # number of samples to process in each multiprocessing process
if multiprocess:
    import multiprocessing

    N_processes = int(np.ceil(np.hstack(chosen_bodypart_arrays).shape[0] / chunk_size))
    if N_processes > 1:
        print(f"Using {N_processes} processes to simulate spikes in parallel")
    else:
        print(f"Using {N_processes} process to simulate spikes")
    mu_list = [
        mu.copy() for i in range(N_processes)
    ]  # identical copies of the MUsim object
    interp_final_force_array_segments = np.array_split(
        interp_final_force_array, N_processes
    )
    with multiprocessing.Pool(processes=N_processes) as pool:
        # cut interp_final_force_array into N processes segments
        # use starmap to pass multiple arguments to the batch_run_MUsim function
        results = pool.starmap(
            batch_run_MUsim,
            zip(
                mu_list,
                interp_final_force_array_segments,
                range(N_processes),
            ),
        )

    # now combine the results from each process into a single MUsim object
    mu = MUsim(random_seed_entropy)
    mu.num_units = num_motor_units
    mu.MUspike_dynamics = "spike_history"
    mu.force_profile = np.hstack([i.force_profile.flatten() for i in results])
    # make sure all units have the same thresholds (units[0])
    assert all([np.all(i.units[0] == results[0].units[0]) for i in results])
    mu.units[0] = results[0].units[0]  # then use the first units[0] as the new units[0]
    try:
        mu.units[1] = np.hstack(
            [i.units[1] for i in results]
        )  # stack the unit response curves
    except ValueError:
        # concatenate the unit response curves if they are different lengths using minimum length
        min_length = min([len(i.units[1]) for i in results])
        mu.units[1] = np.hstack([i.units[1][:min_length] for i in results])

    # make sure all units have the same poisson lambdas (units[2])
    assert all([np.all(i.units[2] == results[0].units[2]) for i in results])
    mu.units[2] = np.hstack([i.units[2] for i in results])  # stack the poisson lambdas
    mu.spikes = np.hstack([i.spikes for i in results])  # stack the spike responses
else:
    mu = batch_run_MUsim(mu, interp_final_force_array, 0)

if show_matplotlib_figures:
    mu.see("force")  # plot the force profile
    mu.see("curves")  # plot the unit response curves
    mu.see("spikes")  # plot the spike response
    # wait for user to press Enter or escape to continue
    input(
        "Press Enter to close all figures, save data, and exit... (Ctrl+C to exit without saving)"
    )

# save spikes from simulation if user does not ^C
kinematic_csv_file_name = "_".join(
    (
        kinematic_csv_file_paths[0].stem.split("-")[0],
        kinematic_csv_file_paths[0].stem.split("_")[1],
    )
)  # just get the date and rat name
if save_simulated_spikes:
    mu.save_spikes(
        # f"synthetic_spikes_from_{kinematic_csv_file_name}_using_{chosen_bodypart_to_load}.npy"
        f"spikes_{kinematic_csv_file_name}_SNR-{adjust_SNR}-{SNR_mode}_jitter-{shape_jitter_amount}std_files-{len(anipose_sessions_to_load)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.npy"
    )
    # also save a copy with name "most_recent_synthetic_spikes.npy"
    mu.save_spikes("most_recent_synthetic_spikes.npy")

## next section will place real multichannel electrophysiological spike waveform shapes at each
#  simulated spike time, onto multiple data channels. The final result will be an int16 binary file
#  called continuous.dat, which can be read by the spike sorting software Kilosort.
#  To do this, it will load the SVD matrixes from rez.mat, use those to produce multichannel
#  electrophysiological data snippets of length nt0 (2.033 ms) from the simulated spikes
#  these will then be placed at the simulated spike times for num_chans channels, and the resulting
#  array will be saved as continuous.dat
if use_KS_templates:
    num_dummy_chans = chan_map_adj_list[0]["numDummy"][0][0]
    num_chans_with_data = int(num_chans_in_recording - num_dummy_chans)
else:
    num_chans_with_data = num_chans_in_recording


# this chunk uses KS templates to create waveform shapes for each spike time
spike_counts_for_each_unit = mu.spikes[-1].sum(axis=0).astype(int)
if use_KS_templates:
    # W are the temporal components to be used to reconstruct unique temporal components
    # U are the weights of each temporal component distrubuted across channels
    W = rez_list[0]["W"]  # shape is (nt0, mu.num_units, SVD_dim)
    U = rez_list[0]["U"]  # shape is (SVD_dim, mu.num_units, num_chans)

    W_good = []
    U_good = []
    U_mean = []
    U_std = []
    # take the W and U matrixes from each recording in rez_list, and only take the good clusters
    # then get SVD_dim standard deviation values across all
    for ii, iRec in enumerate(rez_list):
        W_good.append(iRec["W"][:, clusters_to_take_from[ii], :])
        U_good.append(iRec["U"][:, clusters_to_take_from[ii], :])
        # take mean and std of all elements in U_good
        U_mean.append(np.mean(U_good[ii]))
        U_std.append(np.std(U_good[ii]))

    # extrapolate the U_std matrix to have num_chans columns, by taking mean and STD of all rows,
    # and Gaussian sampling with that mean and STD to fill in the rest of the new columns, using the
    # statistics found for each row of the original U_std matrix
    # U_std_mean = np.mean(U_std, axis=1)
    # U_std_std = np.std(U_std, axis=1)
    # U_vals_to_add = RNG3.normal(U_std_mean, U_std_std, (SVD_dim, num_chans - SVD_dim))
    # U_std_extrap = np.hstack((U_std, U_vals_to_add))

    # now create a new U matrix with SVD_dim rows and num_chans columns
    # make it a random matrix with values between -1 and 1
    # it will project from SVD space to the number of channels in the data
    # base_sim_U = RNG4.normal(0, U_std_extrap, (SVD_dim, num_chans_in_recording))

    # now create slightly jittered (by 20% of std) of waveform shapes for each MU, for each spike time in the simulation
    # first, create a new array of zeros to hold the new multichannel waveform shapes
    spike_snippets_to_place = np.zeros(
        (mu.num_units, np.max(spike_counts_for_each_unit), nt0, num_chans_in_recording)
    )
    for iUnit, iCount in enumerate(spike_counts_for_each_unit):
        # add the jitter to each U_good element, and create a new waveform shape for each spike time
        # use jittermat to change the waveform shape slightly for each spike example (at each time)
        jitter_mat = np.zeros((iCount, num_chans_in_recording, SVD_dim))
        jitter_mat_to_chans_with_data = RNG.normal(
            0,
            shape_jitter_amount * U_std[0],
            (iCount, num_chans_with_data, SVD_dim),
        )
        jitter_mat[:, :num_chans_with_data, :] = jitter_mat_to_chans_with_data
        # jitter and scale by amplitude for this unit
        iUnit_U = np.tile(
            U_good[0][:, iUnit, :], (iCount, 1, 1)
        ) * amplitudes_df_list.loc[clusters_to_take_from[0][iUnit]].Amplitude + (
            jitter_mat if shape_jitter_amount else 0
        )  # additive shape jitter
        for iSpike in range(iCount):
            iSpike_U = iUnit_U[
                iSpike, :, :
            ]  # get the weights for each channel for this spike
            # now project from the templates to create the waveform shape for each spike time
            spike_snippets_to_place[iUnit, iSpike, :, :] = np.dot(
                W_good[0][:, iUnit, :], iSpike_U.T
            )
else:  # this chunk uses the real data from proc.dat to create waveform shapes for each spike time,
    # use spike times for each cluster to extract median waveform shape, with -nt0//2 and +nt0//2 + 1
    ## first, extract all spikes at each corresponding spike time from each proc.dat file, and place them in a combined array
    median_spikes_list = []
    unit_counter = 0
    unit_start_offset = 0
    for ii, ephys_data in enumerate(ephys_data_list):
        # get the spike times for each cluster
        spike_times = spike_times_list[ii]
        spike_clusters = spike_clusters_list[ii]
        # get the spike times for each cluster
        spike_times_for_each_cluster = [
            spike_times[spike_clusters == iCluster]
            for iCluster in clusters_to_take_from[ii]
        ]
        # get the spike snippets for each cluster
        spike_snippets_for_each_cluster = [
            np.array(
                [
                    ephys_data[
                        int(iSpike_time - nt0 // 2) : int(iSpike_time + nt0 // 2 + 1), :
                    ]
                    for iSpike_time in iCluster_spike_times
                ]
            )
            for iCluster_spike_times in spike_times_for_each_cluster
        ]  # dimensions are (num_spikes, nt0, num_chans_in_recording)
        # get the median waveform shape for each cluster
        median_spike_snippets_for_each_cluster = [
            np.median(iCluster_snippets, axis=0)
            for iCluster_snippets in spike_snippets_for_each_cluster
        ]
        median_spikes_list.append(median_spike_snippets_for_each_cluster)
        # use RNG to randomize the channel order of the median waveform shape for each cluster
        # for iCluster in median_spike_snippets_for_each_cluster:
        #     RNG.shuffle(iCluster, axis=1)

        # now place the median waveform shape for each cluster into the spike_snippets_to_place array
        # for iUnit, iCount in enumerate(
        #     spike_counts_for_each_unit[
        #         unit_start_offset : len(np.unique(clusters_to_take_from[ii]))
        #         + unit_start_offset
        #     ]
        # ):
        #     for iSpike in range(iCount):
        #         spike_snippets_to_place[
        #             iUnit, iSpike, :, :
        #         ] = median_spike_snippets_for_each_cluster[iUnit]
        # unit_start_offset += len(np.unique(clusters_to_take_from[ii]))
        # that didn't work, so try placing the median waveform shape from each cluster into the spike_snippets_to_place array
        # make sure not to overwrite any of the previous waveform shapes across ephys_data_list iterations
        spike_snippets_to_place = np.zeros(
            (
                mu.num_units,
                np.max(spike_counts_for_each_unit),
                nt0,
                num_chans_in_recording,
            )
        )
        for jj in range(len(np.unique(clusters_to_take_from[ii]))):
            spike_snippets_to_place[unit_counter, :, :, :] = (
                median_spike_snippets_for_each_cluster[jj]
            )
            unit_counter += 1
        unit_start_offset += len(np.unique(clusters_to_take_from[ii]))

median_spikes_array = np.concatenate(
    median_spikes_list
)  # new shape is (num_units, nt0, num_chans_in_recording)
order_by_amplitude = np.max(np.abs(median_spikes_array), axis=(1, 2)).argsort()
if True:
    # plot the median waveform shape for each cluster in a different subplot, which is channel x unit
    # plot the unit by color

    # plot the median waveform shape for each cluster in a different subplot, which is channel x unit
    # plot the unit by color
    # concatenate the median waveform shapes for each cluster into a single array
    # sort the median_spikes_array in order of lowest unit amplitude to highest
    median_spikes_array = median_spikes_array[order_by_amplitude, :, :]

    fig = subplots.make_subplots(
        rows=num_chans_in_recording,
        cols=num_motor_units,
        subplot_titles=[
            f"Unit {iUnit} Ch. {iChan}"
            for iChan in range(num_chans_in_recording)
            for iUnit in range(num_motor_units)
        ],
        shared_xaxes=True,
        shared_yaxes=True,
    )
    for iChan in range(num_chans_in_recording):
        for iUnit in range(num_motor_units):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(nt0) / ephys_fs * 1000,
                    y=median_spikes_array[iUnit][:, iChan],
                    name=f"Unit {iUnit}",
                    marker_color=MU_colors[iUnit],
                ),
                row=iChan + 1,
                col=iUnit + 1,
            )
    # add title and axis labels
    fig.update_layout(
        title="<b>Median Waveform Shapes for Each Cluster</b>",
        template=plot_template,
    )
    fig.update_yaxes(title_text="Voltage (uV)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=num_chans_in_recording, col=1)
    # make y-axis range shared across all voltage subplots
    fig.update_yaxes(matches="y")
    fig.show()

# set_trace()

# multiply all waveforms by a Tukey window to make the edges go to zero
tukey_window = signal.windows.tukey(nt0, 0.25)
tukey_window = np.tile(tukey_window, (num_chans_in_recording, 1)).T
# order the spike snippets to place by spike by amplitude
spike_snippets_to_place = spike_snippets_to_place[
    order_by_amplitude, :, :, :
]  # new shape is (num_units, num_spikes, nt0, num_chans_in_recording)
for iUnit in range(mu.num_units):
    for iSpike in range(spike_counts_for_each_unit[iUnit]):
        spike_snippets_to_place[iUnit, iSpike, :, :] *= tukey_window

# # plot all waveforms for all units
# for iUnit in range(mu.num_units):
#     fig = px.line(
#         spike_snippets_to_place[iUnit, 0, :, :],
#         title=f"Unit {iUnit} Waveform Shapes",
#     )
#     fig.show()

# now create a new array (mu.spikes.shape[0], num_chans_in_recording) of zeros,
# and place the corresponding waveform shape at each 1 in mu.spikes for each unit
# after each iteration, sum continuous_dat with the previous iteration's result
if shift_MU_templates_along_channels:
    continuous_dat = np.zeros((mu.spikes[-1].shape[0], num_chans_in_output))
    zeros_to_append_along_channels = np.zeros(
        (
            spike_snippets_to_place.shape[0],
            spike_snippets_to_place.shape[1],
            spike_snippets_to_place.shape[2],
            abs(num_chans_in_output - num_chans_in_recording),
        ),
    )
    if zeros_to_append_along_channels.shape[3]:
        spike_snippets_to_place = np.concatenate(
            (spike_snippets_to_place, zeros_to_append_along_channels), axis=3
        )
    # now create an array of random shifts along channels for each MU template
    random_shifts_along_channels = RNG.integers(0, num_chans_in_output, mu.num_units)
    for iUnit in range(spike_snippets_to_place.shape[0]):
        spike_snippets_to_place[iUnit] = np.roll(
            spike_snippets_to_place[iUnit], random_shifts_along_channels[iUnit], axis=-1
        )
    # set_trace()
else:
    continuous_dat = np.zeros((mu.spikes[-1].shape[0], num_chans_in_recording))

for iUnit, iCount in enumerate(spike_counts_for_each_unit):
    iSpike_times = np.where(mu.spikes[-1][:, iUnit] == 1)[0]
    for iSpike in range(iCount):
        iSpike_time = iSpike_times[iSpike]
        # add the waveform shape to the continuous_dat array
        try:
            continuous_dat[
                iSpike_time - nt0 // 2 : iSpike_time + nt0 // 2 + 1, :
            ] += spike_snippets_to_place[iUnit, iSpike, :, :]
        except ValueError:
            # if the spike time is too close to the beginning or end of the recording,
            # only include the part of the waveform shape that fits in the recording
            if iSpike_time - nt0 // 2 < 0:
                underflow_amount = nt0 // 2 - iSpike_time
                continuous_dat[
                    : iSpike_time + nt0 // 2 + 1, :
                ] += spike_snippets_to_place[iUnit, iSpike, underflow_amount:, :]
            elif iSpike_time + nt0 // 2 + 1 > continuous_dat.shape[0]:
                overflow_amount = iSpike_time + nt0 // 2 + 1 - continuous_dat.shape[0]
                continuous_dat[iSpike_time - nt0 // 2 :, :] += spike_snippets_to_place[
                    iUnit, iSpike, :-overflow_amount, :
                ]
            else:
                print("Unknown error in placing waveform shape in continuous_dat")
                set_trace()
        except:
            raise

# get spike band power of the data using the bandpass filter on range (300,1000) Hz
# first, filter the data
spike_filtered_dat = np.zeros((len(continuous_dat), num_chans_with_data))
for iChan in range(num_chans_with_data):
    spike_filtered_dat[:, iChan] = butter_bandpass_filter(
        continuous_dat[:, iChan], 300, 1000, ephys_fs, order=2
    )  # 2nd order butterworth bandpass filter
# now square then average to get power
spike_band_power = np.mean(np.square(spike_filtered_dat), axis=0)
print(f"Spike Band Power: {spike_band_power}")
if adjust_SNR is not None:
    if SNR_mode == "from_data":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {device}")
        torch_continuous_dat = torch.tensor(continuous_dat, device=device)
        # get Gaussian_STDs variable from chanMapAdjusted.mat file
        Gaussian_STDs_of_data = torch.tensor(
            chan_map_adj_list[0]["Gaussian_STDs"][0], device=device
        )
        Gaussian_STDs_of_data = (
            Gaussian_STDs_of_data[0 : num_chans_with_data + num_dummy_chans]
            * adjust_SNR
        )  # multiply by factor of adjust_SNR
        Gaussian_STDs_of_data[num_chans_with_data:] = 0
        # target MAD of data should be to get within 1% of Gaussian_STDs_of_data values
        # initialize it
        MAD_k = torch.median(
            torch.abs(torch_continuous_dat - torch.mean(torch_continuous_dat, axis=0)),
            axis=0,
        )
        # get the Gaussian noise standard deviation of the data
        Gaussian_STDs_of_sim = MAD_k.values / 0.6745

        # make this shape of torch_continuous_dat
        new_noise_STD = torch.tensor(
            np.zeros(num_chans_in_recording), requires_grad=True, device=device
        )

        def forward(torch_continuous_dat):
            # add Gaussian noise to the data
            torch_continuous_dat_out = (
                torch_continuous_dat
                + torch.randn_like(torch_continuous_dat, device=device) * new_noise_STD
            )
            # get the median absolute deviation of the data
            MAD_k = torch.median(
                torch.abs(
                    torch_continuous_dat_out
                    - torch.mean(torch_continuous_dat_out, axis=0)
                ),
                axis=0,
            )
            # get the Gaussian noise standard deviation of the data
            Gaussian_STDs_of_sim = MAD_k.values / 0.6745
            return Gaussian_STDs_of_sim, torch_continuous_dat_out

        def criterion(Gaussian_STDs_of_sim, Gaussian_STDs_of_data):
            return torch.mean(torch.abs(Gaussian_STDs_of_sim - Gaussian_STDs_of_data))

        # back calculate the additional noise needed to make Gaussian_STDs_of_sim equal to
        # Gaussian_STDs_of_data
        # add noise gradually to be absolutely sure NOT to overshoot
        # goal is to make Gaussian_STDs_of_sim within 1% of Gaussian_STDs_of_data for all channels
        # now use optimizer to change learning rate
        print(f"Target Noise STD: {Gaussian_STDs_of_data}")
        optimizer = torch.optim.Adam([new_noise_STD], lr=0.001)
        loss_BGD = []

        for i in range(1000):
            Gaussian_STDs_of_sim, torch_continuous_dat_out = forward(
                torch_continuous_dat
            )
            loss = criterion(Gaussian_STDs_of_sim, Gaussian_STDs_of_data)
            loss_BGD.append(loss.item())
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad()
            if i % 100 == 0:
                print(f"Iteration: {i}")
                print(f"New Noise STD: {new_noise_STD}")
                print(f"Loss: {loss.item()}")
                print(f"Gaussian_STDs_of_sim: {Gaussian_STDs_of_sim}")
                print(f"Gaussian_STDs_of_data: {Gaussian_STDs_of_data}")
            # ignore nan values, but make sure all equal to or less than 1% of Gaussian_STDs_of_data
            if (
                torch.abs(Gaussian_STDs_of_sim - Gaussian_STDs_of_data)
                / Gaussian_STDs_of_data
                <= 0.01
            )[:num_chans_with_data].all():
                print("Gaussian_STDs_of_sim is less than 1% of Gaussian_STDs_of_data")
                print(f"Took {i} iterations to converge")
                break
        else:
            print(
                "Gaussian_STDs_of_sim values did not converge to less than 1% of Gaussian_STDs_of_data"
            )
        print(f"Final Loss: {loss.item()}")
        continuous_dat = torch_continuous_dat_out.detach().cpu().numpy()
    elif SNR_mode == "power":
        # back-calculate the noise needed to get the amplitude of Gaussian noise to add to the data
        # to get the desired SNR
        noise_power = spike_band_power / adjust_SNR
        # now calculate the standard deviation of the Gaussian noise to add to the data
        noise_std = np.sqrt(noise_power)
        print(f"Noise STD: {noise_std}")
        # now add Gaussian noise to the data
        noise_std_with_dummies = (
            np.zeros(num_chans_in_recording)
            if not shift_MU_templates_along_channels
            else np.zeros(num_chans_in_output)
        )
        noise_std_with_dummies[0:num_chans_with_data] = noise_std
        continuous_dat += RNG.normal(
            0, 2 * noise_std_with_dummies, continuous_dat.shape
        )
    elif SNR_mode == "constant":
        # just add a constant amount of Gaussian noise to all channels of the data
        # to get the desired SNR
        noise_std = adjust_SNR
        print(f"Noise STD: {noise_std}")
        # now add Gaussian noise to the data
        noise_std_with_dummies = (
            np.zeros(num_chans_in_recording)
            if not shift_MU_templates_along_channels
            else np.zeros(num_chans_in_output)
        )
        noise_std_with_dummies[0:num_chans_with_data] = noise_std
        continuous_dat += RNG.normal(0, noise_std_with_dummies, continuous_dat.shape)
    # elif SNR_mode == "mean_waveforms":
    #     # mean subtract each channel, take absolute value, then take median for each channel
    #     MAD_k = np.median(np.abs(continuous_dat - np.mean(continuous_dat, axis=0)), axis=0)
    #     Gaussian_noise_std_k = MAD / 0.6745
    #     # for SNR take max of each mean waveform, then divide by Gaussian_noise_std_k
    #     SNR_k = np.max(np.abs(mu.units[0]), axis=0) / Gaussian_noise_std_k

# Verify that the SNR is correct
# first, get the spike band power of the data
spike_filtered_dat_after_noise = np.zeros((len(continuous_dat), num_chans_with_data))
for iChan in range(num_chans_with_data):
    spike_filtered_dat_after_noise[:, iChan] = butter_bandpass_filter(
        continuous_dat[:, iChan], 300, 1000, ephys_fs, order=2
    )  # 2nd order butterworth bandpass filter
# now square then average to get power
spike_band_power = np.mean(np.square(spike_filtered_dat_after_noise), axis=0)
# compute power outside of spike band
# get lowband power
low_band_filtered_dat = np.zeros((len(continuous_dat), num_chans_with_data))
for iChan in range(num_chans_with_data):
    low_band_filtered_dat[:, iChan] = butter_bandpass_filter(
        continuous_dat[:, iChan], 5, 100, ephys_fs, order=2
    )  # 2nd order butterworth bandpass filter
# now square then average to get power
low_band_power = np.mean(np.square(low_band_filtered_dat), axis=0)
# get highband power
high_band_filtered_dat = np.zeros((len(continuous_dat), num_chans_with_data))
for iChan in range(num_chans_with_data):
    high_band_filtered_dat[:, iChan] = butter_bandpass_filter(
        continuous_dat[:, iChan], 10000, 14000, ephys_fs, order=2
    )  # 2nd order butterworth bandpass filter
# now square then average to get power
high_band_power = np.mean(np.square(high_band_filtered_dat), axis=0)
outside_spike_band_power = low_band_power + high_band_power
# compute SNR
computed_SNR = spike_band_power / outside_spike_band_power
print(f"Computed SNR: {computed_SNR}")

if use_KS_templates:
    # finally use ops variable channelDelays to reapply the original channel delays to the data
    channel_delays_to_apply = ops_list[0]["channelDelays"][0]
    for iChan in range(num_chans_with_data):
        continuous_dat[:, iChan] = np.roll(
            continuous_dat[:, iChan], -channel_delays_to_apply[iChan]
        )

    continuous_dat *= 200  # scale for Kilosort

if show_final_plotly_figure or save_final_plotly_figure:
    import colorlover as cl

    N_colors = num_chans_in_output
    CH_colors = cl.to_rgb(cl.interp(cl.scales["6"]["seq"]["Greys"], 2 * N_colors))[
        -1 : -(N_colors + 1) : -1
    ]
    # now plot the continuous.dat array with plotly graph objects
    # first, create a time vector for ephys to plot against
    ephys_time_axis = np.linspace(
        0, len(continuous_dat) / ephys_fs, len(continuous_dat)
    )
    # second, create a time vector for kinematics to plot against
    force_time_axis = np.linspace(
        0, len(blended_chosen_array) / kinematics_fs, len(blended_chosen_array)
    )
    # include the anipose force profile on the top subplot
    # include MUsim object spike time eventplot on the final subplot
    # create a figure with subplots, make x-axis shared,
    # make the layout tight (subplots, close together)
    # alot two rows for the spike eventplot
    chans_to_plot = list(range(num_chans_in_output))  # [1, 2, 4, 5, 13, 14]
    num_chans_to_plot = len(chans_to_plot)
    number_of_rows = num_chans_to_plot + 3  # num_chans_with_data + 3
    row_spec_list = number_of_rows * [[None]]
    for iRow in range(num_chans_to_plot + 1):
        row_spec_list[iRow] = [{"rowspan": 1, "secondary_y": True}]
    row_spec_list[-2] = [{"rowspan": 2}]

    sub_titles = number_of_rows * [""]
    sub_titles[0] = f"<b>Simulated Kinematics: {kinematic_csv_file_name}</b>"
    sub_titles[1] = f"<b>Simulated Motor Unit Activity: {kinematic_csv_file_name}</b>"
    sub_titles[-2] = f"<b>Simulated Spikes : {kinematic_csv_file_name}</b>"

    fig = subplots.make_subplots(
        rows=num_chans_to_plot + 3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        specs=row_spec_list,
        subplot_titles=sub_titles,
    )
    # add the force profile to the top subplot
    fig.add_trace(
        go.Scatter(
            x=force_time_axis,
            y=np.round(blended_chosen_array, decimals=2),
            name="Force Profile",
            marker=dict(color="cornflowerblue"),
        ),
        row=1,
        col=1,
    )

    # add traces, one for each channel
    for ii, iChan in enumerate(chans_to_plot):  # range(num_chans_with_data):
        fig.add_trace(
            go.Scatter(
                x=ephys_time_axis,
                y=np.round(continuous_dat[:, iChan], decimals=2),
                name=f"Channel {iChan}",
                line=dict(width=0.5),
                marker=dict(color=CH_colors[iChan]),
            ),
            row=ii + 2,
            col=1,
        )

    # add eventplot of spike times to the last subplot, vertically spacing times and coloring by unit
    MU_colors = [
        "royalblue",
        "firebrick",
        "forestgreen",
        "darkorange",
        "darkorchid",
        "darkgreen",
        "lightcoral",
        "rgb(116, 77, 37)",
        "cyan",
        "mediumpurple",
        "lightslategray",
        "gold",
        "lightpink",
        "darkturquoise",
        "darkkhaki",
        "darkviolet",
        "darkslategray",
        "darkgoldenrod",
        "darkmagenta",
        "darkcyan",
        "darkred",
        "darkblue",
        "darkslateblue",
        "darkolivegreen",
        "darkgray",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "maroon",
        "mediumblue",
        "mediumorchid",
        "mediumseagreen",
        "mediumslateblue",
        "mediumturquoise",
        "magenta",
        "forestgreen",
        "mediumvioletred",
        "midnightblue",
        "navy",
        "olive",
        "olivedrab",
    ]
    # flatten clusters_to_take_from into a single list
    clusters_to_take_from = [i for j in clusters_to_take_from for i in j]
    for iUnit in range(mu.num_units):
        fig.add_trace(
            go.Scatter(
                x=mu.spikes[-1][:, iUnit].nonzero()[0] / ephys_fs,
                y=np.ones(int(mu.spikes[-1][:, iUnit].sum())) * -iUnit,
                mode="markers",
                marker_symbol="line-ns",
                marker=dict(
                    color=MU_colors[iUnit],
                    line_color=MU_colors[iUnit],
                    line_width=1.2,
                    size=10,
                ),
                name=f"Unit {clusters_to_take_from[iUnit]}",
                showlegend=False,
            ),
            row=num_chans_to_plot + 2,
            col=1,
        )

    # add title and axis labels, make sure x-axis title is only on bottom subplot
    fig.update_layout(
        # title=f"<b>Simulated Data from {kinematic_csv_file_name} using {chosen_bodypart_to_load}</b>",
        template=plot_template,
    )
    fig.update_yaxes(
        title_text="<b>Channel #</b>",
        row=num_chans_to_plot // 2 + 2,
        col=1,
        side="left",
        secondary_y=True,
    )
    # round down to nearest thousand from largest-amplitude channel's 15th standard deviation
    max_ylim = 15 * 1000 * (np.max(np.std(continuous_dat, axis=0)) // 1000)
    fig.update_yaxes(
        row=2,
        col=1,
        side="right",
        tickvals=np.arange(-max_ylim, max_ylim + 1, 5000),
        title_text="<b>Voltage (μV)</b>",
    )
    for iRow in range(2, num_chans_to_plot + 2):
        # add np.nan so the secondary y-axis shows up
        fig.add_trace(
            go.Scatter(
                x=[np.nan],
                y=[np.nan],
                name="",
                showlegend=False,
            ),
            row=iRow,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            range=[-max_ylim, max_ylim],
            row=iRow,
            col=1,
        )
        if iRow > 2:
            fig.update_yaxes(showticklabels=False, row=iRow, col=1, side="right")
        fig.update_yaxes(
            tickvals=[0],
            ticktext=[f"{chans_to_plot[iRow - 2]}"],
            secondary_y=True,
            showticklabels=True,
            side="left",
            row=iRow,
            col=1,
        )
    fig.update_yaxes(
        row=2,
        col=1,
        side="left",
        title_text="",
        secondary_y=True,
    )

    fig.update_yaxes(title_text="<b>Force (a.u.)</b>", row=1, col=1, autorange=True)
    fig.update_yaxes(
        title_text="<b>Unit #</b>",
        row=num_chans_to_plot + 2,
        col=1,
        tickvals=list(range(-len(clusters_to_take_from) + 1, 1)),
        ticktext=list(reversed(clusters_to_take_from)),
        range=[-len(clusters_to_take_from) + 1, 1],
    )
    fig.update_xaxes(title_text="<b>Time (s)</b>", row=num_chans_to_plot + 2, col=1)

    if save_final_plotly_figure:
        # fig.write_image(
        #     f"{kinematic_csv_file_name}_using_{chosen_bodypart_to_load}.svg",
        #     width=1920,
        #     height=1080,
        # )
        fig.write_html(
            f"{kinematic_csv_file_name}_using_{chosen_bodypart_to_load}.html"
        )
    if show_final_plotly_figure:
        print("Showing final plotly figure...")
        fig.show()

# add dummy channels to the data to make it num_chans_in_output channels
# or remove channels if too many
if num_chans_in_output > num_chans_in_recording:
    continuous_dat = np.hstack(
        (
            continuous_dat,
            np.zeros(
                (len(continuous_dat), num_chans_in_output - num_chans_in_recording)
            ),
        )
    )
elif num_chans_in_output < num_chans_in_recording:
    continuous_dat = continuous_dat[:, :num_chans_in_output]

# now save the continuous.dat array as a binary file
# first, convert to int16
# continuous_dat *= 200  # scale for Kilosort
continuous_dat = continuous_dat.astype(np.int16)
print(f"Continuous.dat shape: {continuous_dat.shape}")
print(f"Overall recording length: {len(continuous_dat) / ephys_fs} seconds")

# fig_tmp = px.line(continuous_dat[:30000, :num_chans_with_data])
# fig_tmp.show()

# now save as binary file in int16 format, where elements are 2 bytes, and samples from each channel
# are interleaved, such as: [chan1_sample1, chan2_sample1, chan3_sample1, ...]
# save simulation properties in continuous.dat file name
if save_continuous_dat:
    continuous_dat.tofile(
        f"continuous_{kinematic_csv_file_name}_SNR-{adjust_SNR}-{SNR_mode}_jitter-{shape_jitter_amount}std_files-{len(anipose_sessions_to_load)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.dat"
    )
    # overwrite a copy of most recent continuous.dat file
    continuous_dat.tofile("most_recent_continuous.dat")
print("Synthetic Data Generated Successfully!")

# ## compare synthetic data to real data
# # first, load real data
# mu_real = MUsim()
# # mu_real.num_units = mu.num_units
# # mu.sample_rate = ephys_fs  # 30000 Hz
# # mu_real.sample_MUs()
# # mu_real.apply_new_force(interp_final_force_array)
# mu_real.load_MUs(
#     "/home/smoconn/git/rat-loco/20221116-3_godzilla_speed05_incline00_time.npy",
#     bin_width=0.00003333333333333333,
#     load_as="trial",
#     slice=time_frame,
# )

# # set_trace()
# # mu_real.see("force")
# # mu_real.see("curves")
# if show_matplotlib_figures:
#     mu.see("spikes")
#     mu_real.see("spikes")
#     input("Press Enter to close all figures, and exit... (or Ctrl+C)")

finish_time = datetime.now()
time_elapsed = finish_time - start_time
# use strfdelta to format time elapsed prettily
print(
    (
        "Time elapsed: "
        f"{strfdelta(time_elapsed, '{hours} hours, {minutes} minutes, {seconds} seconds')}"
    )
)
