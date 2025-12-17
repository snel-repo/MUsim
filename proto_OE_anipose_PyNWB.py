from datetime import datetime
from pathlib import Path
from pdb import set_trace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ndx_pose import PoseEstimation, PoseEstimationSeries
from neuroconv.datainterfaces import OpenEphysRecordingInterface
from pandas import DataFrame as df
from pandas import read_csv
from pynwb import NWBHDF5IO, NWBFile
from pynwb.behavior import BehavioralEpochs


def load_behavioral_data(file_path):
    df = pd.read_csv(file_path)
    keypoints = 10
    timepoints = len(df)
    behavioral_data = np.zeros((timepoints, keypoints, 3))

    for i in range(1, keypoints + 1):
        behavioral_data[:, i - 1, 0] = df[f"x{i}"]
        behavioral_data[:, i - 1, 1] = df[f"y{i}"]
        behavioral_data[:, i - 1, 2] = df[f"z{i}"]

    return behavioral_data


def load_epoch_data(file_path):
    df = pd.read_csv(file_path)
    return df[["stance_start", "stance_end", "swing_start", "swing_end"]].to_numpy()


# Define paths
path_to_ephys_data = (
    "/home/sean/Downloads/EMUsort_paper_dataset_RAT/ephys/2022-11-16_16-19-28"
)
nwb_file_path = "EMUsort_paper_dataset_RAT.nwb"

# Convert data
stream_names = OpenEphysRecordingInterface.get_stream_names(folder_path=path_to_ephys_data)
interface = OpenEphysRecordingInterface(
    folder_path=path_to_ephys_data, stream_name=stream_names[0]
)
metadata = interface.get_metadata()

# fix time zone
metadata["NWBFile"]["session_start_time"] = metadata["NWBFile"][
    "session_start_time"
].replace(tzinfo=ZoneInfo("America/New_York"))

# add subject metadata
metadata["Subject"] = dict(
    subject_id=f"Godzilla",
    species="Rattus norvegicus",
    description="Long Evans rat",
    sex="F",
    age="P20M",
)
metadata["NWBFile"].update(
    session_id=f"20221116_godzilla_triceps_quadcamera",
    session_description="Electrophysiology data with synchronized motion capture from 4 cameras, triangulated with Anipose",
    experiment_description="Waveforms taken from locomotion of a rat on a treadmill",
    experimenter=["O'Connell, Sean"],
    lab="SNEL and Sober Lab",
    institution="The Wallace H. Coulter Department of Biomedical Engineering, Georgia Tech and Emory University",
    keywords=[
        "EMUsort",
        "motor unit",
        "MUAP",
        "MUsim",
        "SNEL",
        "Sober",
        "rat",
        "EMG",
        "treadmill",
        "locomotion",
        "Anipose",
    ],
)
metadata["Ecephys"]["ElectrodeGroup"][0].update(
    location="Triceps brachii, Lateral head, Left",
    device="Myomatrix, RF-4x8-BVS-8 (previously RF400)",
    description="Simulation based on a bipolar EMG recording with 4 separate threads and 8 channels per thread",
)
metadata["Ecephys"]["Device"][0].update(
    name="Open Ephys Acquisition System",
    description="Open source electrophysiology acquisition system",
)

interface.run_conversion(
    nwbfile_path=nwb_file_path, overwrite=True
)  # , nwb_file=nwb_file_path)

# load anipose CSVs
# base_anipose_path = Path("/snel/share/data/anipose/analysis20230830_godzilla/pose-3d")
# base_anipose_path = Path("/snel/share/data/anipose/analysis20230830_godzilla/pose-3d_processed_by_rat-loco")
base_anipose_path = Path("/home/sean/Downloads/EMUsort_paper_dataset_RAT/anipose/pose-3d_processed_by_rat-loco")
anipose_pose_3d_files = [
    base_anipose_path / "reference_frame_aligned_anipose_df_20221116-5_godzilla_speed10_incline00.csv",
    base_anipose_path / "reference_frame_aligned_anipose_df_20221116-8_godzilla_speed10_incline00.csv",
    base_anipose_path / "reference_frame_aligned_anipose_df_20221116-9_godzilla_speed10_incline00.csv",
]

anipose_dfs = dict()
for i, csv in enumerate(anipose_pose_3d_files):
    anipose_dfs[f'rec{i+1}'] = read_csv(csv)
    
set_trace()
# Read the NWB file
with NWBHDF5IO(nwb_file_path, "r+") as io:
    nwb_file_handle = io.read()
    for iCam in range(4):
        camera = nwb_file_handle.create_device(
            name=f"cam{iCam}",
            description="BlackFly S video camera (BFS-U3-16S2M-CS) for recording behavior",
            manufacturer="Teledyne FLIR LLC",
        )

    data = np.random.rand(100, 2)  # num_frames x (x, y) but can be (x, y, z)
    timestamps = np.linspace(0, 10, num=100)  # a timestamp for every frame
    confidence = np.random.rand(100)  # a confidence value for every frame
    reference_frame = "(0,0,0) corresponds to the left-right treadmill midline for x, the back of the treadmill for y, and the treadmill surface for z, all with ~+/-1mm measurement error."
    confidence_definition = "Anipose score output after calibration and triangulation"
    
    set_trace()

    # # Define behavioral timings
    # behavioral_timings = {
    #     'recording1': (0, 60),
    #     'recording2': (60, 120),
    #     'recording3': (120, 180)
    # }

    # # Process each recording
    # for recording_name, time_range in behavioral_timings.items():
    #     csv_file_path = f'{recording_name}_behavioral_data.csv'
    #     raw_behavioral_data = load_behavioral_data(csv_file_path)

    #     behavioral_timestamps = np.linspace(time_range[0], time_range[1], len(raw_behavioral_data))

    #     # Load stance and swing epochs
    #     epochs = load_epoch_data(f'{recording_name}_epochs.csv')  # CSV for stance and swing times

    #     # Create BehavioralEpochs for stance and swing
    #     stance_epochs = BehavioralEpochs(name=f'{recording_name}_Stance_Epochs',
    #                                     start_times=epochs[:, 0],
    #                                     stop_times=epochs[:, 1])

    #     swing_epochs = BehavioralEpochs(name=f'{recording_name}_Swing_Epochs',
    #                                     start_times=epochs[:, 2],
    #                                     stop_times=epochs[:, 3])

    #     # Add epochs to the NWB file
    #     nwb_file_handle.add_acquisition(stance_epochs)
    #     nwb_file_handle.add_acquisition(swing_epochs)

    # Save updated NWB file
    io.write(nwb_file_handle)

print("NWB file with behavioral epochs for stance and swing created successfully!")
