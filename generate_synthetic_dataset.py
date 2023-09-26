# IMPORT packages
from pathlib import Path
from pdb import set_trace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subplots
from scipy import signal
from scipy.io import loadmat

from MUsim import MUsim

# set parameters
show_plotly_figures = False
show_matplotlib_figures = False
save_simulated_spikes = False
kinematics_fs = 125
ephys_fs = 30000
nt0 = 61  # 2.033 ms
num_chans = 16  # desired real number of channels in the output data
SVD_dim = 9  # number of SVD components than were used in KiloSort
num_chans_in_recording = 9  # number of channels in the recording
time_frame = [0, 0.25]  # time frame to plot, fractional bounds of 0 to 1

## first load a 1D kinematic array from a csv file into a numpy array
# format is YYYYMMDD-N, with N being the session number
session_to_load = "20221116-3"  # "20230323-4"  # "20221116-7" # "20230323-4"
# format is bodypart_side_axis, with side being L or R, and axis being x, y, or z
chosen_bodypart_to_load = "palm_L_y"  # "wrist_L_y"
reference_bodypart_to_load = "tailbase_y"
# choose path to get anipose data from
anipose_folder = Path("/snel/share/data/anipose/analysis20230830_godzilla/")
# anipose_folder = Path("/snel/share/data/anipose/analysis20230829_inkblot+kitkat")

# load the csv file into a pandas dataframe and get numpy array of chosen bodypart
kinematic_csv_folder_path = anipose_folder.joinpath("pose-3d")
files = [f for f in kinematic_csv_folder_path.iterdir() if f.is_file() and f.suffix == ".csv"]
kinematic_csv_file_path = [f for f in files if session_to_load in f.name][0]
print(kinematic_csv_file_path)
kinematic_dataframe = pd.read_csv(kinematic_csv_file_path)
chosen_bodypart_array = kinematic_dataframe[chosen_bodypart_to_load].to_numpy()
time_slice = slice(time_frame[0], round(len(chosen_bodypart_array) * time_frame[1]))
chosen_bodypart_array = chosen_bodypart_array[time_slice]

# plot the array with plotly express
fig = px.line(chosen_bodypart_array)
# add title and axis labels
fig.update_layout(
    title="Raw Kinematics-derived Force Profile",
    xaxis_title="Time",
    yaxis_title="Force Approximation",
)
if show_plotly_figures:
    fig.show()

## load, 2Hz lowpass, and subtract reference bodypart from chosen bodypart
reference_bodypart_array = kinematic_dataframe[reference_bodypart_to_load].to_numpy()[time_slice]
# lowpass filter the array
nyq = 0.5 * kinematics_fs
ref_lowcut = 2
ref_low = ref_lowcut / nyq
ref_order = 2
ref_b, ref_a = signal.butter(ref_order, ref_low, btype="low")
ref_filtered_array = signal.filtfilt(ref_b, ref_a, reference_bodypart_array)
# subtract reference bodypart from chosen bodypart
ref_chosen_bodypart_array = chosen_bodypart_array - ref_filtered_array

fig = px.line(ref_chosen_bodypart_array)
# add title and axis labels
fig.update_layout(
    title="Raw Kinematics-derived Force Profile, Reference Subtracted",
    xaxis_title="Time",
    yaxis_title="Force Approximation",
)
if show_plotly_figures:
    fig.show()

# bandpass filter the array
lowcut = 1
highcut = 8
low = lowcut / nyq
high = highcut / nyq
order = 2
b, a = signal.butter(order, [low, high], btype="band")
filtered_array = signal.filtfilt(b, a, ref_chosen_bodypart_array)
# invert the array to better simulate force from y kinematics
inverted_filtered_array = -filtered_array
# plot the filtered array with plotly express
fig = px.line(inverted_filtered_array)
# add title and axis labels
fig.update_layout(
    title="Filtered and Inverted Kinematics-derived Force Profile",
    xaxis_title="Time",
    yaxis_title="Force Approximation",
)
if show_plotly_figures:
    fig.show()

# make all postive
final_force_array = inverted_filtered_array - np.min(inverted_filtered_array)
# plot the final array with plotly express
fig = px.line(final_force_array)
# add title and axis labels
fig.update_layout(
    title="Final Kinematics-derived Force Profile",
    xaxis_title="Time",
    yaxis_title="Force Approximation",
)
if show_plotly_figures:
    fig.show()

# interpolate final force array to match ephys sampling rate
interp_final_force_array = signal.resample(
    final_force_array, round(len(final_force_array) * (ephys_fs / kinematics_fs))
)
truncated_bodypart_array = interp_final_force_array[: len(interp_final_force_array) // 2]

## initialize MU simulation object, set number of units,
#  and apply the inverted filtered array as the force profile
mu = MUsim()
mu.num_units = 8
mu.MUthresholds_dist = "exponential"
mu.sample_rate = ephys_fs  # 30000 Hz
# fixed minimum threshold for the generated units' response curves
mu.threshmin = np.percentile(truncated_bodypart_array, 30)
# fixed maximum threshold for the generated units' response curves
mu.threshmax = np.percentile(truncated_bodypart_array, 99)
mu.sample_MUs()
mu.apply_new_force(truncated_bodypart_array)
mu.simulate_spikes()
print("MU Thresholds " + str(mu.units[0]))
print("MU Poisson Lambdas " + str(mu.units[2]))
# pdb.set_trace()
if show_matplotlib_figures:
    mu.see("force")  # plot the force profile
    mu.see("curves")  # plot the unit response curves
    mu.see("spikes")  # plot the spike response
    # wait for user to press Enter or escape to continue
    input(
        "Press Enter to close all figures, save data, and exit... (Ctrl+C to exit without saving)"
    )

# save spikes from simulation if user does not ^C
kinematic_csv_file_name = kinematic_csv_file_path.stem
if save_simulated_spikes:
    mu.save_spikes(
        f"synthetic_spikes_from_{kinematic_csv_file_name}_using_{chosen_bodypart_to_load}.npy"
    )

## next section will place real multichannel electrophysiological spike waveform shapes at each
#  simulated spike time, onto multiple data channels. The final result will be an int16 binary file
#  called continuous.dat, which can be read by the spike sorting software Kilosort.
#  To do this, it will load the SVD matrixes from rez.mat, use those to produce multichannel
#  electrophysiological data snippets of length nt0 (2.033 ms) from the simulated spikes
#  these will then be placed at the simulated spike times for num_chans channels, and the resulting
#  array will be saved as continuous.dat

# load the SVD matrixes from rez.mat
paths_to_session_folders = Path(
    "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/session20221116/",
    # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/inkblot/session20230323",
    # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/kitkat/session20230420",
)
sorts_from_each_path_to_load = ["20230924_151421"]  # , ["20230923_125645"], ["20230923_125645"]]
# find the folder name which ends in _myo and append to the paths_to_session_folders
paths_to_each_myo_folder = [
    f for f in paths_to_session_folders.iterdir() if f.is_dir() and f.name.endswith("_myo")
]
# inside each _myo folder, find the folder name which constains sort_from_each_path_to_load string
for iPath in paths_to_each_myo_folder:
    path_to_sort_folders = [
        f
        for f in iPath.iterdir()
        if f.is_dir() and any(s in f.name for s in sorts_from_each_path_to_load)
    ]

rez_list = [
    loadmat(str(path_to_sort_folder.joinpath("rez.mat")))
    for path_to_sort_folder in path_to_sort_folders
]
ops_list = [
    loadmat(str(path_to_sort_folder.joinpath("ops.mat")))
    for path_to_sort_folder in path_to_sort_folders
]

# list of lists of good clusters to take from each rez_list
clusters_to_take_from = [[18, 2, 11, 0, 4, 10, 1, 9]]

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
# U_vals_to_add = np.random.normal(U_std_mean, U_std_std, (SVD_dim, num_chans - SVD_dim))
# U_std_extrap = np.hstack((U_std, U_vals_to_add))

# now create a new U matrix with SVD_dim rows and num_chans columns
# make it a random matrix with values between -1 and 1
# it will project from SVD space to the number of channels in the data
# base_sim_U = np.random.normal(0, U_std_extrap, (SVD_dim, num_chans_in_recording))
spike_counts_for_each_unit = mu.spikes[-1].sum(axis=0).astype(int)

# now create slightly jittered (by 20% of std) of waveform shapes for each MU, for each spike time in the simulation
# first, create a new array of zeros to hold the new multichannel waveform shapes
spike_snippets_to_place = np.zeros(
    (mu.num_units, np.max(spike_counts_for_each_unit), nt0, num_chans_in_recording)
)

for iUnit, iCount in enumerate(spike_counts_for_each_unit):
    # add the jitter to each U_good element, and create a new waveform shape for each spike time
    # use jittermat to change the waveform shape slightly for each spike example (at each time)
    jitter_mat = np.random.normal(
        1,
        U_std[0],
        (iCount, num_chans_in_recording, SVD_dim),
    )
    # jitter and scale by amplitude for this unit
    iUnit_U = np.tile(U_good[0][:, iUnit, :], (iCount, 1, 1)) * jitter_mat * mu.units[0][iUnit]
    for iSpike in range(iCount):
        iSpike_U = iUnit_U[iSpike, :, :]  # get the weights for each channel for this spike
        # now project from the templates to create the waveform shape for each spike time
        spike_snippets_to_place[iUnit, iSpike, :, :] = np.dot(W_good[0][:, iUnit, :], iSpike_U.T)

# multiply all waveforms by a Tukey window to make the edges go to zero
tukey_window = signal.tukey(nt0, 0.5)
tukey_window = np.tile(tukey_window, (num_chans_in_recording, 1)).T
for iUnit in range(mu.num_units):
    for iSpike in range(spike_counts_for_each_unit[iUnit]):
        spike_snippets_to_place[iUnit, iSpike, :, :] *= tukey_window


# now create a new array (mu.spikes.shape[0], num_chans_in_recording) of zeros, and place the corresponding
# waveform shape at each 1 in mu.spikes for each unit
# after each iteration, sum continuous_dat with the previous iteration's result
continuous_dat = np.zeros((mu.spikes[-1].shape[0], num_chans_in_recording))
for iUnit, iCount in enumerate(spike_counts_for_each_unit):
    iSpike_times = np.where(mu.spikes[-1][:, iUnit] == 1)[0]
    for iSpike in range(iCount):
        iSpike_time = iSpike_times[iSpike]
        # add the waveform shape to the continuous_dat array
        continuous_dat[
            iSpike_time - nt0 // 2 : iSpike_time + nt0 // 2 + 1, :
        ] += spike_snippets_to_place[iUnit, iSpike, :, :]


# now plot the continuous.dat array with plotly graph objects
# first, create a time vector to plot against
time_vector = np.linspace(0, len(continuous_dat) / ephys_fs, len(continuous_dat))
# create a figure with subplots, make x-axis shared, make the layout tight (subplots, close together)
fig = subplots.make_subplots(
    rows=num_chans_in_recording,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.005,
)

# add traces, one for each channel
for iChan in range(num_chans_in_recording):
    fig.add_trace(
        go.Scatter(x=time_vector, y=continuous_dat[:, iChan], name=f"Channel {iChan}"),
        row=iChan + 1,
        col=1,
    )

# add title and axis labels
fig.update_layout(
    title=f"Simulated Data from {kinematic_csv_file_name} using {chosen_bodypart_to_load}",
    xaxis_title="Time (s)",
    yaxis_title="Simulated Voltage",
)
# if show_plotly_figures:
fig.show()

# now save the continuous.dat array as a binary file
# first, convert to int16
continuous_dat = continuous_dat.astype(np.int16)
# now save as binary file
continuous_dat.tofile(f"continuous.dat")
