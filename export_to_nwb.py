import pdb
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from neuroconv.datainterfaces import OpenEphysRecordingInterface

# paths to simulated datasets
# simulated_kinematics = Path("force_array_20240621-163356_shuffled_2_times.npy")
base_path = Path(
    "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test"
)
datasets = [
    "sim_2022-11-17_17-08-07_shape_noise_0.00",
    # "sim_2022-11-17_17-08-07_shape_noise_0.75",
    # "sim_2022-11-17_17-08-07_shape_noise_1.50",
    "sim_2022-11-17_17-08-07_shape_noise_2.00",
    # "sim_2022-11-17_17-08-07_shape_noise_2.25",
    # "sim_2022-11-17_17-08-07_shape_noise_3.00",
    # "sim_2022-11-17_17-08-07_shape_noise_3.75",
    "sim_2022-11-17_17-08-07_shape_noise_4.00",
]
shapeSTDs = [int(0), int(2), int(4)]

d_idx = 2
folder_path = base_path / datasets[d_idx] / "Record Node 101"
print(folder_path)
nwb_save_path = Path(
    "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/nwb_files/LITMUS_Rat"
)
nwb_save_path.mkdir(parents=True, exist_ok=True)

# Change the folder_path to the appropriate location in your system
interface = OpenEphysRecordingInterface(folder_path=folder_path)
# Extract what metadata we can from the source files
metadata = interface.get_metadata()

# session_start_time is required for conversion. If it cannot be inferred
# automatically from the source files you must supply one.
# 2022-11-17_17-08-07
session_start_time = datetime(2022, 11, 17, 17, 8, 7, tzinfo=ZoneInfo("US/Eastern"))
# add subject metadata
metadata["Subject"] = dict(
    subject_id=f"Godzilla_{shapeSTDs[d_idx]}-STD",
    species="Rattus norvegicus",
    description="Long Evans rat",
    sex="F",
    age="P20M",
)

metadata["NWBFile"].update(session_start_time=session_start_time)
metadata["NWBFile"].update(
    session_id=f"LITMUS_Rat_{shapeSTDs[d_idx]}-STD",
    session_description=f"Simulated data with shape noise {shapeSTDs[d_idx]} STD",
    experiment_description="Waveforms taken from locomotion of a rat on a treadmill",
    experimenter=["O'Connell, Sean"],
    lab="SNEL and Sober Lab",
    institution="Wallace H. Coulter Department of Biomedical Engineering, Georgia Tech and Emory University",
    keywords=["LITMUS", "Motor Unit", "MUsim", "SNEL", "EMUsort", "Rat", "EMG"],
)
metadata["Ecephys"]["ElectrodeGroup"][0].update(
    location="Triceps brachii, Lateral head, Left",
    device="Myomatrix, RF400",
    description="Simulation based on a bipolar EMG recording with 4 separate threads and 8 channels per thread",
)
metadata["Ecephys"]["Device"][0].update(
    name="Open Ephys Acquisition System",
    description="Open source electrophysiology acquisition system",
)

# metadata["VectorData"] = dict(description="Simulated electromyography data")

# Choose a path for saving the nwb file and run the conversion
nwbfile_path = nwb_save_path / f"LITMUS_Rat_{shapeSTDs[d_idx]}-STD.nwb"
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True)
