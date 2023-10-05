# IMPORT packages
import datetime
import multiprocessing
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

start_time = datetime.datetime.now()  # begin timer for script execution time


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
    blended_array = np.zeros(len(np.concatenate(array_list)) - Nsamp * (len(array_list) - 1))
    len_prev_arrays = 0
    for ii, iArray in enumerate(array_list):
        len_this_array = len(iArray)
        if iArray is array_list[0]:
            iArray[-Nsamp:] *= 1 - blender_array  # only blend the first array at the end
            backward_shift = 0
        elif iArray is array_list[-1]:
            iArray[:Nsamp] *= blender_array  # only blend the last array at the beginning
            backward_shift = Nsamp * ii
        else:  # blend both ends of the array for all other arrays
            iArray[:Nsamp] *= blender_array
            iArray[-Nsamp:] *= 1 - blender_array
            backward_shift = Nsamp * ii
        blended_array[
            len_prev_arrays - backward_shift : len_prev_arrays - backward_shift + len_this_array
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
save_final_plotly_figure = True
save_simulated_spikes = True
kinematics_fs = 125
ephys_fs = 30000
nt0 = 61  # 2.033 ms
SVD_dim = 9  # number of SVD components than were used in KiloSort
num_chans_in_recording = 9  # number of channels in the recording
num_chans_in_output = 17  # desired real number of channels in the output data
# number determines noise power added to channels (e.g. 50), or set None to disable
change_SNR_to = None  # 50
# factor to multiply U_std by to get amount of jitter to add to waveform shapes
shape_jitter_amount = 0
# set None for random behavior, or a previous entropy int value to reproduce
random_seed_entropy = 75092699954400878964964014863999053929  # None
if random_seed_entropy is None:
    random_seed_entropy = np.random.SeedSequence().entropy
RNG = np.random.default_rng(random_seed_entropy)  # create a random number generator

# set plotting parameters
time_frame = [0, 1]  # time frame to plot, fractional bounds of 0 to 1
plot_template = "plotly_white"

# check inputs
assert type(kinematics_fs) is int and kinematics_fs > 0, "kinematics_fs must be a positive integer"
assert type(ephys_fs) is int and ephys_fs > 0, "ephys_fs must be a positive integer"
assert (
    type(shape_jitter_amount) in [int, float] and shape_jitter_amount >= 0
), "shape_jitter_amount must be a positive number or None"
assert (type(change_SNR_to) in [int, float] and change_SNR_to >= 0) or (
    change_SNR_to is None
), "SNR_of_simulated_data must be a positive number or None"
assert time_frame[0] >= 0 and time_frame[1] <= 1 and time_frame[0] < time_frame[1], (
    "time_frame must be a list of two numbers between 0 and 1, " "with the first number smaller"
)


## first load a 1D kinematic array from a csv file into a numpy array
# format is YYYYMMDD-N, with N being the session number
anipose_sessions_to_load = [
    "20221116-3",
    # "20221116-5",
    # "20221116-7",
    # "20221116-8",
    # "20221116-9",
    # "20221116-9",
]
# format is bodypart_side_axis, with side being L or R, and axis being x, y, or z
chosen_bodypart_to_load = "palm_L_y"  # "wrist_L_y"
reference_bodypart_to_load = "tailbase_y"
# choose path to get anipose data from
anipose_folder = Path("/snel/share/data/anipose/analysis20230830_godzilla/")
# anipose_folder = Path("/snel/share/data/anipose/analysis20230829_inkblot+kitkat")

# load the csv file into a pandas dataframe and get numpy array of chosen bodypart
kinematic_csv_folder_path = anipose_folder.joinpath("pose-3d")
files = [f for f in kinematic_csv_folder_path.iterdir() if f.is_file() and f.suffix == ".csv"]
kinematic_csv_file_paths = [f for f in files if any(s in f.name for s in anipose_sessions_to_load)]
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
Nsamp = 25
for ii in range(len(reference_bodypart_arrays)):
    iRef = reference_bodypart_arrays[ii]
    ref_filtered_array_list.append(
        butter_lowpass_filter(iRef, ref_lowcut, kinematics_fs, ref_order)
    )
    iSub = ref_filtered_array_list[ii]
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
MU_spike_history_kernel_path = Path(__file__).parent.joinpath("spike_history_kernel_basis_MU.csv")
orig_spike_history_kernel_path = Path(__file__).parent.joinpath("spike_history_kernel_basis.csv")

MU_spike_history_kernel_df = pd.read_csv(MU_spike_history_kernel_path)
orig_spike_history_kernel_df = pd.read_csv(orig_spike_history_kernel_path)

if show_plotly_figures:
    # now use plotly express to plot the two dataframes in two subplots
    # make first subplot be the original spike history kernel,
    # and second subplot be the MU spike history kernel
    fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)

    # add the original spike history kernel to the top subplot
    for ii, iCol in enumerate(orig_spike_history_kernel_df.columns):
        fig.add_trace(
            go.Scatter(
                x=orig_spike_history_kernel_df.index / ephys_fs * 1000,
                y=orig_spike_history_kernel_df[iCol],
                name=f"Component {ii}",
            ),
            row=1,
            col=1,
        )

    # add the MU spike history kernel to the bottom subplot
    for ii, iCol in enumerate(MU_spike_history_kernel_df.columns):
        fig.add_trace(
            go.Scatter(
                x=MU_spike_history_kernel_df.index / ephys_fs * 1000,
                y=MU_spike_history_kernel_df[iCol],
                name=f"Component {ii}",
            ),
            row=2,
            col=1,
        )

    # add title and axis labels, make sure x-axis title is only on bottom subplot
    fig.update_layout(
        title="<b>Spike History Kernel Bases</b>",
        template=plot_template,
    )
    fig.update_yaxes(title_text="Original Kernel Values", row=1, col=1)
    fig.update_yaxes(title_text="Motor Unit Kernel Values", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.show()

## load Kilosort data
# paths to the folders containing the Kilosort data
paths_to_KS_session_folders = Path(
    "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/session20221116/",
    # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/inkblot/session20230323",
    # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/kitkat/session20230420",
)
sorts_from_each_path_to_load = ["20230924_151421"]  # , ["20230923_125645"], ["20230923_125645"]]

# find the folder name which ends in _myo and append to the paths_to_session_folders
paths_to_each_myo_folder = [
    f for f in paths_to_KS_session_folders.iterdir() if f.is_dir() and f.name.endswith("_myo")
]
# inside each _myo folder, find the folder name which constains sort_from_each_path_to_load string
for iPath in paths_to_each_myo_folder:
    path_to_sort_folders = [
        f
        for f in iPath.iterdir()
        if f.is_dir() and any(s in f.name for s in sorts_from_each_path_to_load)
    ]

# load the SVD matrixes from rez.mat
rez_list = [
    loadmat(str(path_to_sort_folder.joinpath("rez.mat")))
    for path_to_sort_folder in path_to_sort_folders
]
ops_list = [
    loadmat(str(path_to_sort_folder.joinpath("ops.mat")))
    for path_to_sort_folder in path_to_sort_folders
]

chan_map_adj_list = [
    loadmat(str(path_to_sort_folder.joinpath("chanMapAdjusted.mat")))
    for path_to_sort_folder in path_to_sort_folders
]

amplitudes_df_list = [
    pd.read_csv(str(path_to_sort_folder.joinpath("cluster_Amplitude.tsv")), sep="\t")
    for path_to_sort_folder in path_to_sort_folders
]
# set index as the cluster_id column
amplitudes_df_list = amplitudes_df_list[0].set_index("cluster_id")

# list of lists of good clusters to take from each rez_list
# place units in order of total spike count, from highest to lowest
clusters_to_take_from = [[18, 2, 11, 0, 4, 10, 1, 9]]
num_motor_units = len(clusters_to_take_from[0])

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
mu.threshmin = np.percentile(interp_final_force_array, 70)
# fixed maximum force threshold for the generated units' response curves. Tune this for upper
# bound of force thresholds sampled in the distribution of MUs during MU_sample()
mu.threshmax = 1.1 * np.max(interp_final_force_array)  # np.percentile(interp_final_force_array, 99)
mu.sample_MUs()

# now create N copies of the MUsim object
chunk_size = 500  # number of samples to process in each multiprocessing process
N_processes = len(anipose_sessions_to_load) * np.ceil(
    len(chosen_bodypart_arrays[0]) / chunk_size
).astype(int)
mu_list = [mu.copy() for i in range(N_processes)]  # identical copies of the MUsim object
interp_final_force_array_segments = np.array_split(interp_final_force_array, N_processes)
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
mu.units[1] = np.hstack([i.units[1] for i in results])  # stack the unit response curves
# make sure all units have the same poisson lambdas (units[2])
assert all([np.all(i.units[2] == results[0].units[2]) for i in results])
mu.units[2] = np.hstack([i.units[2] for i in results])  # stack the poisson lambdas
mu.spikes = np.hstack([i.spikes for i in results])  # stack the spike responses

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
kinematic_csv_file_name = "_".join(
    (kinematic_csv_file_paths[0].stem.split("-")[0], kinematic_csv_file_paths[0].stem.split("_")[1])
)  # just get the date and rat name
if save_simulated_spikes:
    mu.save_spikes(
        # f"synthetic_spikes_from_{kinematic_csv_file_name}_using_{chosen_bodypart_to_load}.npy"
        f"spikes_{kinematic_csv_file_name}_SNR-{change_SNR_to}_jitter-{shape_jitter_amount}std_files-{len(anipose_sessions_to_load)}.npy"
    )

## next section will place real multichannel electrophysiological spike waveform shapes at each
#  simulated spike time, onto multiple data channels. The final result will be an int16 binary file
#  called continuous.dat, which can be read by the spike sorting software Kilosort.
#  To do this, it will load the SVD matrixes from rez.mat, use those to produce multichannel
#  electrophysiological data snippets of length nt0 (2.033 ms) from the simulated spikes
#  these will then be placed at the simulated spike times for num_chans channels, and the resulting
#  array will be saved as continuous.dat
num_dummy_chans = chan_map_adj_list[0]["numDummy"][0][0]
num_chans_with_data = int(num_chans_in_recording - num_dummy_chans)

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
spike_counts_for_each_unit = mu.spikes[-1].sum(axis=0).astype(int)

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
    iUnit_U = np.tile(U_good[0][:, iUnit, :], (iCount, 1, 1)) * amplitudes_df_list.loc[
        clusters_to_take_from[0][iUnit]
    ].Amplitude + (
        jitter_mat if shape_jitter_amount else 0
    )  # additive shape jitter
    for iSpike in range(iCount):
        iSpike_U = iUnit_U[iSpike, :, :]  # get the weights for each channel for this spike
        # now project from the templates to create the waveform shape for each spike time
        spike_snippets_to_place[iUnit, iSpike, :, :] = np.dot(W_good[0][:, iUnit, :], iSpike_U.T)

# multiply all waveforms by a Tukey window to make the edges go to zero
tukey_window = signal.windows.tukey(nt0, 0.25)
tukey_window = np.tile(tukey_window, (num_chans_in_recording, 1)).T
for iUnit in range(mu.num_units):
    for iSpike in range(spike_counts_for_each_unit[iUnit]):
        spike_snippets_to_place[iUnit, iSpike, :, :] *= tukey_window


# now create a new array (mu.spikes.shape[0], num_chans_in_recording) of zeros,
# and place the corresponding waveform shape at each 1 in mu.spikes for each unit
# after each iteration, sum continuous_dat with the previous iteration's result
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
                continuous_dat[: iSpike_time + nt0 // 2 + 1, :] += spike_snippets_to_place[
                    iUnit, iSpike, underflow_amount:, :
                ]
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
if change_SNR_to is not None:
    # back-calculate the noise needed to get the amplitude of Gaussian noise to add to the data
    # to get the desired SNR
    noise_power = spike_band_power / change_SNR_to
    # now calculate the standard deviation of the Gaussian noise to add to the data
    noise_std = np.sqrt(noise_power)
    print(f"Noise STD: {noise_std}")
    # now add Gaussian noise to the data
    noise_std_with_dummies = np.zeros(num_chans_in_recording)
    noise_std_with_dummies[0:num_chans_with_data] = noise_std
    continuous_dat += RNG.normal(0, 2 * noise_std_with_dummies, continuous_dat.shape)

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

if show_final_plotly_figure or save_final_plotly_figure:
    # now plot the continuous.dat array with plotly graph objects
    # first, create a time vector for ephys to plot against
    ephys_time_axis = np.linspace(0, len(continuous_dat) / ephys_fs, len(continuous_dat))
    # second, create a time vector for kinematics to plot against
    force_time_axis = np.linspace(
        0, len(blended_chosen_array) / kinematics_fs, len(blended_chosen_array)
    )
    # include the anipose force profile on the top subplot
    # include MUsim object spike time eventplot on the final subplot
    # create a figure with subplots, make x-axis shared,
    # make the layout tight (subplots, close together)
    fig = subplots.make_subplots(
        rows=num_chans_with_data + 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.005,
    )
    # add the force profile to the top subplot
    fig.add_trace(
        go.Scatter(
            x=force_time_axis,
            y=np.round(blended_chosen_array, decimals=2),
            name="Force Profile",
        ),
        row=1,
        col=1,
    )

    # add traces, one for each channel
    for iChan in range(num_chans_with_data):
        fig.add_trace(
            go.Scatter(
                x=ephys_time_axis,
                y=np.round(continuous_dat[:, iChan], decimals=2),
                name=f"Channel {iChan}",
                line=dict(width=0.5),
            ),
            row=iChan + 2,
            col=1,
        )

    # add eventplot of spike times to the last subplot, vertically spacing times and coloring by unit
    MU_colors = px.colors.sequential.Rainbow
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
                name=f"Unit {clusters_to_take_from[0][iUnit]}",
                showlegend=False,
            ),
            row=num_chans_with_data + 2,
            col=1,
        )

    # add title and axis labels, make sure x-axis title is only on bottom subplot
    fig.update_layout(
        title=f"<b>Simulated Data from {kinematic_csv_file_name} using {chosen_bodypart_to_load}</b>",
        template=plot_template,
    )
    fig.update_yaxes(title_text="Simulated Force", row=1, col=1)
    fig.update_yaxes(title_text="Simulated Voltage (μV)", row=num_chans_with_data // 2 + 2, col=1)
    fig.update_yaxes(title_text="MUsim Object Spike Times", row=num_chans_with_data + 2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=num_chans_with_data + 2, col=1)

    if show_final_plotly_figure:
        print("Showing final plotly figure...")
        fig.show()
    if save_final_plotly_figure:
        fig.write_html(f"{kinematic_csv_file_name}_using_{chosen_bodypart_to_load}.html")

# add dummy channels to the data to make it num_chans_in_output channels
if num_chans_in_output > num_chans_in_recording:
    continuous_dat = np.hstack(
        (
            continuous_dat,
            np.zeros((len(continuous_dat), num_chans_in_output - num_chans_in_recording)),
        )
    )
elif num_chans_in_output < num_chans_in_recording:
    continuous_dat = continuous_dat[:, :num_chans_in_output]

# now save the continuous.dat array as a binary file
# first, convert to int16
continuous_dat *= 200  # scale for Kilosort
continuous_dat = continuous_dat.astype(np.int16)
print(f"Continuous.dat shape: {continuous_dat.shape}")
print(f"Overall recording length: {len(continuous_dat) / ephys_fs} seconds")
# now save as binary file in int16 format, where elements are 2 bytes, and samples from each channel
# are interleaved, such as: [chan1_sample1, chan2_sample1, chan3_sample1, ...]
# save simulation properties in continuous.dat file name
continuous_dat.tofile(
    f"continuous_{kinematic_csv_file_name}_SNR-{change_SNR_to}_jitter-{shape_jitter_amount}std_files-{len(anipose_sessions_to_load)}.dat"
)
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

finish_time = datetime.datetime.now()
time_elapsed = finish_time - start_time
# use strfdelta to format time elapsed prettily
print(
    (
        "Time elapsed: "
        f"{strfdelta(time_elapsed, '{hours} hours, {minutes} minutes, {seconds} seconds')}"
    )
)
