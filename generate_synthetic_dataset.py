# IMPORT packages
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import signal

from MUsim import MUsim

## first load a 1D kinematic array from a csv file into a numpy array
# format is YYYYMMDD-N, with N being the session number
session_to_load = "20221116-3"  # "20230323-4"  # "20221116-7" # "20230323-4"
# format is bodypart_side_axis, with side being L or R, and axis being x, y, or z
bodypart_to_load = "palm_L_y"
reference_bodypart_to_load = "tailbase_y"
# choose path to get anipose data from
anipose_folder = Path("/snel/share/data/anipose/analysis20230830_godzilla/")
# anipose_folder = Path("/snel/share/data/anipose/analysis20230829_inkblot+kitkat")

kinematic_csv_folder_path = anipose_folder.joinpath("pose-3d")
files = [f for f in kinematic_csv_folder_path.iterdir() if f.is_file() and f.suffix == ".csv"]
kinematic_csv_file_path = [f for f in files if session_to_load in f.name][0]
print(kinematic_csv_file_path)
kinematic_dataframe = pd.read_csv(kinematic_csv_file_path)
chosen_bodypart_array = kinematic_dataframe[bodypart_to_load].to_numpy()

# plot the array with plotly express
# fig = px.line(chosen_bodypart_array)
# # add title and axis labels
# fig.update_layout(
#     title="Raw Kinematics-derived Force Profile",
#     xaxis_title="Time",
#     yaxis_title="Force Approximation",
# )
# fig.show()

# load, 2Hz lowpass, and subtract reference bodypart from chosen bodypart
reference_bodypart_array = kinematic_dataframe[reference_bodypart_to_load].to_numpy()
# lowpass filter the array
kinematics_fs = 125
nyq = 0.5 * kinematics_fs
ref_lowcut = 2
ref_low = ref_lowcut / nyq
ref_order = 2
ref_b, ref_a = signal.butter(ref_order, ref_low, btype="low")
ref_filtered_array = signal.filtfilt(ref_b, ref_a, reference_bodypart_array)
# subtract reference bodypart from chosen bodypart
ref_chosen_bodypart_array = chosen_bodypart_array - ref_filtered_array

# fig = px.line(ref_chosen_bodypart_array)
# # add title and axis labels
# fig.update_layout(
#     title="Raw Kinematics-derived Force Profile, Reference Subtracted",
#     xaxis_title="Time",
#     yaxis_title="Force Approximation",
# )
# fig.show()

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
# fig = px.line(inverted_filtered_array)
# # add title and axis labels
# fig.update_layout(
#     title="Filtered and Inverted Kinematics-derived Force Profile",
#     xaxis_title="Time",
#     yaxis_title="Force Approximation",
# )
# fig.show()

# make all postive
final_force_array = inverted_filtered_array - np.min(inverted_filtered_array)
# plot the final array with plotly express
# fig = px.line(final_force_array)
# # add title and axis labels
# fig.update_layout(
#     title="Final Kinematics-derived Force Profile",
#     xaxis_title="Time",
#     yaxis_title="Force Approximation",
# )
# fig.show()

# interpolate final force array to match ephys sampling rate
ephys_fs = 30000
interp_final_force_array = signal.resample(
    final_force_array, round(len(final_force_array) * (ephys_fs / kinematics_fs))
)
truncated_bodypart_array = interp_final_force_array[: len(interp_final_force_array) // 2]

## initialize MU simulation object, set number of units, and apply the inverted filtered array as the force profile
mu = MUsim()
mu.num_units = 10
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
mu.see("force")  # plot the force profile
mu.see("curves")  # plot the unit response curves
mu.see("spikes")  # plot the spike response

# wait for user to press Enter or escape to continue
input("Press Enter to continue...")
