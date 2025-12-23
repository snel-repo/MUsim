# This script shows how to load this in Python using PyNWB and LINDI
# It assumes you have installed PyNWB and LINDI (pip install pynwb lindi)

import lindi
import pynwb

# Load "20221116-5_godzilla_speed10_incline00"
# # Load https://api.dandiarchive.org/api/assets/1d5d9136-4eb9-42d3-9122-761e0127ef40/download/
# f = lindi.LindiH5pyFile.from_hdf5_file(
#     "https://api.dandiarchive.org/api/assets/1d5d9136-4eb9-42d3-9122-761e0127ef40/download/"
# )

# Load "20221116-8_godzilla_speed10_incline00"
# # Load https://api.dandiarchive.org/api/assets/38893bef-cadc-48b6-a0bb-882febc813bd/download/
# f = lindi.LindiH5pyFile.from_hdf5_file(
#     "https://api.dandiarchive.org/api/assets/38893bef-cadc-48b6-a0bb-882febc813bd/download/"
# )

# Load "20221116-9_godzilla_speed10_incline00"
# Load https://api.dandiarchive.org/api/assets/dd4c0760-3b8a-49f5-83bf-4a3cdb73ea7e/download/
f = lindi.LindiH5pyFile.from_hdf5_file(
    "https://api.dandiarchive.org/api/assets/dd4c0760-3b8a-49f5-83bf-4a3cdb73ea7e/download/"
)

nwb = pynwb.NWBHDF5IO(file=f, mode="r").read()

# nwb.session_description # (str) Rat performed locomotion with the following parameters: speed10 m/min, incline00 degrees.
# nwb.identifier # (str) 95189454-07ef-4a69-87df-0c8c1a2baf86
# nwb.session_start_time # (datetime) 2022-11-16T16:19:28-05:00
# nwb.file_create_date # (datetime) 2025-12-19T17:57:12.022693-05:00
# nwb.timestamps_reference_time # (datetime) 2022-11-16T16:19:28-05:00
# nwb.experimenter # (List[str]) ["O'Connell, Sean", "Wang, Runming"]
# nwb.experiment_description # (str) Synchronized EMG and motion tracking collected from a rat during treadmill locomotion.
# nwb.institution # (str) The Wallace H. Coulter Department of Biomedical Engineering, Georgia Tech and Emory University
# nwb.keywords # (List[str]) ["EMUsort", "motor unit", "MUAP", "MUsim", "SNEL", "Sober", "rat", "EMG", "treadmill", "locomotion", "Anipose"]
# nwb.protocol # (str)
# nwb.lab # (str) SNEL and Sober Lab
# nwb.subject # (Subject)
# nwb.subject.age # (str) P20M
# nwb.subject.age__reference # (str) birth
# nwb.subject.description # (str) Long Evans rat
# nwb.subject.genotype # (str)
# nwb.subject.sex # (str) F
# nwb.subject.species # (str) Rattus norvegicus
# nwb.subject.subject_id # (str) Godzilla
# nwb.subject.weight # (str)
# nwb.subject.date_of_birth # (datetime)

ElectricalSeries = nwb.acquisition[
    "ElectricalSeries"
]  # (ElectricalSeries) Acquisition traces for the ElectricalSeries.
ElectricalSeries.data  # (h5py.Dataset) shape [2576640, 16]; dtype <h
electrodes = ElectricalSeries.electrodes  # (DynamicTableRegion) num. electrodes: 16
# This is a reference into the nwb.electrodes table and can be used in the same way
# For example, electrode_ids = electrodes["id"].data[:] # len(electrode_ids) == 16
# And the other columns can be accessed in the same way
# It's the same table, but a subset of the rows.
ElectricalSeries.starting_time  # 2814.3274666666666 sec
ElectricalSeries.rate  # 30000 Hz

behavior = nwb.processing[
    "behavior"
]  # (ProcessingModule) Contains 3D pose estimation and step timing data derived from synchronized motion capture, DLC, and Anipose during treadmill locomotion.

Anipose = nwb.processing["behavior"]["Anipose"]  # (PoseEstimation)

BehavioralEvents = nwb.processing["behavior"]["BehavioralEvents"]  # (BehavioralEvents)

electrodes = nwb.electrodes  # (DynamicTable)
# electrodes.colnames # (Tuple[str]) ("location", "group", "group_name", "channel_name", "gain_to_physical_unit", "physical_unit", "offset_to_physical_unit")
# electrode_ids = electrodes["id"].data[:] # len(electrode_ids) == 16 (number of electrodes is 16)
# electrodes["location"].data[:] # (np.ndarray) shape [16]; dtype S; Location of the electrode (channel).
# electrodes["group"].data[:] # (np.ndarray) shape [16]; dtype unknown; Reference to the ElectrodeGroup.
# electrodes["group_name"].data[:] # (np.ndarray) shape [16]; dtype S; Name of the ElectrodeGroup.
# electrodes["channel_name"].data[:] # (np.ndarray) shape [16]; dtype S; unique channel reference
# electrodes["gain_to_physical_unit"].data[:]  # (np.ndarray) shape [16]; dtype <d; no description
# electrodes["physical_unit"].data[:] # (np.ndarray) shape [16]; dtype S; no description
# electrodes["offset_to_physical_unit"].data[:] # (np.ndarray) shape [16]; dtype <d; no description

# plot one keypoint and one raw data channel to show alignment for a chosen time slice
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

time_slice = slice(59, 79)
time_slice_idxs = slice(
    time_slice.start * int(ElectricalSeries.rate),
    time_slice.stop * int(ElectricalSeries.rate),
)
emg_time_array = (
    np.arange(time_slice.start, time_slice.stop, 1 / ElectricalSeries.rate)
    + ElectricalSeries.starting_time
)
emg_index_array = np.array(range(time_slice_idxs.start, time_slice_idxs.stop))

emg_chans = [12, 13]

keypoints = [
    "nose",
    "tailbase",
    "wrist_L",
    "wrist_R",
    "palm_L",
    "palm_R",
    "ankle_L",
    "ankle_R",
    "toes_L",
    "toes_R",
]
keypoint = "toes_R"

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=(
        f"Raw EMG Data (Channels {emg_chans})",
        f"Anipose Keypoint ({keypoint})",
    ),
)

for ii, chan in enumerate(emg_chans):
    emg_trace = go.Scatter(
        x=emg_time_array,
        y=ElectricalSeries.data[time_slice_idxs, chan]
        * electrodes["gain_to_physical_unit"].data[0],
        mode="lines",
        name=f"emg{emg_chans[ii]}",
    )

    fig.add_trace(emg_trace, row=1, col=1)
fig.update_layout(
    # autosize=False,
    # width=800,
    # height=400,
    # template="plotly_dark",
    # title=dict(
    #     text="India Daily New Covid Cases",
    #     font=dict(size=24, color="#FFFFFF"),
    #     x=0.5,
    #     y=0.9,
    # ),
    xaxis_title=dict(text="Time (seconds)"),  # font=dict(size=16, color="#FFFFFF")),
    yaxis_title=dict(text="Voltage (uV)"),  # , font=dict(size=16)),
    # plot_bgcolor="rgb(50, 50, 50)",
    # xaxis=dict(tickfont=dict(size=14, color="#FFFFFF")),
    # yaxis=dict(tickfont=dict(size=14, color="#FFFFFF")),
    # legend=dict(x=0.1, y=1.1, orientation="h", font=dict(color="#FFFFFF")),
    # margin=dict(l=10, r=10, t=100, b=50),
)
anipose_time_idxs = np.where(
    (
        np.bitwise_and(
            Anipose.pose_estimation_series[keypoint].timestamps[:]
            >= (time_slice.start + ElectricalSeries.starting_time),
            Anipose.pose_estimation_series[keypoint].timestamps[:]
            < (time_slice.stop + ElectricalSeries.starting_time),
        )
    )
)
anipose_data = Anipose.pose_estimation_series[keypoint].data[anipose_time_idxs]
dims = ["x", "y", "z"]
for jj, anip in enumerate(range(anipose_data.shape[-1])):
    anipose_trace = go.Scatter(
        x=Anipose.pose_estimation_series[keypoint].timestamps[anipose_time_idxs],
        y=anipose_data[:, jj],
        mode="lines",
        name=f"{keypoint}_{dims[jj]}",
    )

    fig.add_trace(anipose_trace, row=2, col=1)

anipose_contacts = BehavioralEvents.time_series["initial_contact"].timestamps[:]
anipose_swings = BehavioralEvents.time_series["initial_swing"].timestamps[:]

anipose_contact_idxs = np.where(
    (
        np.bitwise_and(
            anipose_contacts >= (time_slice.start + ElectricalSeries.starting_time),
            anipose_contacts < (time_slice.stop + ElectricalSeries.starting_time),
        )
    )
)[0]

anipose_swing_idxs = np.where(
    (
        np.bitwise_and(
            anipose_swings >= (time_slice.start + ElectricalSeries.starting_time),
            anipose_swings < (time_slice.stop + ElectricalSeries.starting_time),
        )
    )
)[0]

event_types = [
    anipose_contacts[anipose_contact_idxs],
    anipose_swings[anipose_swing_idxs],
]
colors = ["green", "firebrick"]
# plot vertical lines in row 2
for ee, events in enumerate(event_types):
    for ll, time in enumerate(events):
        fig.add_vline(
            x=time, row=2, col=1, line_width=1, line_dash="dot", line_color=colors[ee]
        )


fig.update_layout(
    xaxis2_title=dict(text="Time (seconds)"),  # font=dict(size=16, color="#FFFFFF")),
    yaxis2_title=dict(text="Postion (mm)"),  # , font=dict(size=16)),
)

fig.show()
