from pathlib import Path
from pdb import set_trace
from zoneinfo import ZoneInfo

import numpy as np
from ndx_pose import PoseEstimation, PoseEstimationSeries
from neuroconv.datainterfaces import OpenEphysRecordingInterface
from pandas import read_csv
from pynwb import NWBHDF5IO
from pynwb.behavior import BehavioralEvents, TimeSeries

# Define paths
path_to_ephys_data = Path(
    "/snel/share/data/emusort/EMUsort_paper_acquired_rat_data/ephys/"
)

path_to_anipose_data = Path(
    "/snel/share/data/emusort/EMUsort_paper_acquired_rat_data/anipose"
)
# Define session IDs
session_ids = [
    "20221116-5_godzilla_speed10_incline00",
    "20221116-8_godzilla_speed10_incline00",
    "20221116-9_godzilla_speed10_incline00",
]

# create one NWB for each recording in the session
for session_id in session_ids:
    nwb_file_path = f"{session_id}_temp.nwb"

    # Convert data
    stream_names = OpenEphysRecordingInterface.get_stream_names(
        folder_path=path_to_ephys_data / session_id / "2022-11-16_16-19-28"
    )
    interface = OpenEphysRecordingInterface(
        folder_path=path_to_ephys_data / session_id, stream_name=stream_names[0]
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
        age="P20M",  # postnatal 20 months
    )
    metadata["NWBFile"].update(
        session_id=f"{session_id}",
        session_description=f"Rat performed locomotion with the following parameters: {session_id.split('_')[-2]} m/min, {session_id.split('_')[-1]} degrees.",
        experiment_description="Synchronized EMG and motion tracking collected from a rat during treadmill locomotion.",
        experimenter=["O'Connell, Sean", "Wang, Runming"],
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
        description="32-channel high-density intramuscular EMG array implanted in the left lateral head of the triceps brachii muscle.",
    )
    metadata["Ecephys"]["Device"][0].update(
        name="Open Ephys Acquisition System",
        description="Open source electrophysiology acquisition system",
    )
    interface.run_conversion(
        nwbfile_path=nwb_file_path, overwrite=True, metadata=metadata
    )

    # Load Anipose data from CSV
    assert (
        path_to_anipose_data
        / "pose-3d_processed_by_rat-loco"
        / f"processed_anipose_df_{session_id}.csv"
    ).is_file(), "Anipose CSV file not found: " + str(
        path_to_anipose_data
        / "pose-3d_processed_by_rat-loco"
        / f"processed_anipose_df_{session_id}.csv"
    )
    anipose_file_path = (
        path_to_anipose_data
        / "pose-3d_processed_by_rat-loco"
        / f"processed_anipose_df_{session_id}.csv"
    )
    anipose_df = read_csv(anipose_file_path)

    # Read the NWB file
    with NWBHDF5IO(nwb_file_path, "r") as io:
        nwb_file_handle = io.read()
        for iCam in range(4):
            camera = nwb_file_handle.create_device(
                name=f"camera{iCam}",
                description=(
                    "BlackFly S video camera (BFS-U3-16S2M-CS) for recording behavior. "
                    "Videos were captured with 2x2 decimation down to 720x540 and sampled with 10-bit resolution at 125Hz. "
                    "Cameras were all triggered by the same TTL pulses to ensure synchronization with https://github.com/nicthib/FLIR-Multicam."
                ),
                manufacturer="Teledyne FLIR LLC",
            )

        # load anipose csv data for each session id
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
        # get all columns of data, which end in _x, _y, or _z, put into np array, then insert into PoseEstimationSeries
        timestamps = (
            anipose_df["timestamps"].to_numpy()
            + nwb_file_handle.acquisition["ElectricalSeries"].starting_time
        )
        dims = ["x", "y", "z"]
        data = dict()
        pose_est_series_list = []
        for keypoint in keypoints:
            keypoint_confidence_array = anipose_df[f"{keypoint}_score"].to_numpy()
            anipose_keypoint_3D_array = np.zeros((len(anipose_df), 3))
            for iDim, dim in enumerate(dims):
                column_name = f"{keypoint}_{dim}"
                anipose_keypoint_3D_array[:, iDim] = anipose_df[column_name].to_numpy()
            pose_est_series = PoseEstimationSeries(
                name=f"{keypoint}",
                description=f"Anipose tracking data for {keypoint} in x, y, z, aligned to treadmill coordinate frame",
                data=anipose_keypoint_3D_array,
                unit="millimeters",
                reference_frame="(0,0,0) corresponds to the left-right treadmill midline for x, the back of the treadmill for y, and the treadmill surface for z.",
                timestamps=timestamps,
                confidence=keypoint_confidence_array,
                confidence_definition="Anipose score output after calibration and triangulation across the four cameras.",
            )
            pose_est_series_list.append(pose_est_series)

        anipose_data = PoseEstimation(
            name="Anipose",
            pose_estimation_series=pose_est_series_list,
            description=(
                "3D pose estimation data obtained using Anipose from synchronized recordings of 4 BlackFly S cameras, "
                "tracking 10 keypoints on the rat's body during treadmill locomotion."
                "Keypoints include: nose, tailbase, wrist_L, wrist_R, palm_L, palm_R, ankle_L, ankle_R, toes_L, and toes_R."
                "The tailbase can be used as the reference point with the least variability to control for body position drift along the treadmill."
                "Measurement error was within +/-1mm for stable keypoints used for calibration (not included in data)"
            ),
            original_videos=[
                f"/snel/share/data/anipose/analysis20230830_godzilla/videos-raw/{session_id}_cam{i}.mp4"
                for i in range(4)
            ],
            labeled_videos=[
                f"/snel/share/data/anipose/analysis20230830_godzilla/videos-labeled/{session_id}_cam{i}_labeled.mp4"
                for i in range(4)
            ],
            dimensions=np.array([[720, 540]], dtype="uint16"),
            devices=[nwb_file_handle.get_device(f"camera{iCam}") for iCam in range(4)],
            scorer="/snel/share/runs/dlc/Godzilla_20230829/dlc-models/iteration-0/GeneralRatDec13-trainset96shuffle1/train/snapshot-86000*",
            source_software="Anipose",
            source_software_version="v1.1.24",
        )
        # create a "behavior" processing module to store the PoseEstimation and Skeletons objects
        behavior_pm = nwb_file_handle.create_processing_module(
            name="behavior",
            description="Contains 3D pose estimation and step timing data derived from synchronized motion capture, DLC, and Anipose during treadmill locomotion.",
        )
        behavior_pm.add(anipose_data)

        # load gait events from csv
        step_timings_file = (
            path_to_anipose_data
            / "step_timings_from_rat-loco"
            / f"clean_step_timings_{session_id}.csv"
        )
        step_timings_df = read_csv(step_timings_file)

        # now add BehavioralEvents for the "initial_contact" and "initial swing" events from csv
        contact_events = TimeSeries(
            name="initial_contact",
            data=step_timings_df["initial_contact_idx"].to_numpy().astype(float),
            unit="frames",
            resolution=1.0,
            timestamps=step_timings_df["initial_contact_timestamps"].to_numpy()
            + nwb_file_handle.acquisition["ElectricalSeries"].starting_time,
            description="Frame index and corresponding timestamps of initial contact events for the left palm keypoint, which was the same side as the implanted triceps EMG array.",
            continuity="instantaneous",
        )
        swing_events = TimeSeries(
            name="initial_swing",
            data=step_timings_df["initial_swing_idx"].to_numpy().astype(float),
            unit="frames",
            resolution=1.0,
            timestamps=step_timings_df["initial_swing_timestamps"].to_numpy()
            + nwb_file_handle.acquisition["ElectricalSeries"].starting_time,
            description="Frame index and corresponding timestamps of initial swing events for the left palm keypoint, which was the same side as the implanted triceps EMG array.",
            continuity="instantaneous",
        )
        stance_duration = TimeSeries(
            name="stance_duration",
            data=step_timings_df["stance_duration"].to_numpy(),
            unit="seconds",
            resolution=0.008,
            timestamps=step_timings_df["initial_contact_timestamps"].to_numpy()
            + nwb_file_handle.acquisition["ElectricalSeries"].starting_time,
            description="Duration of stance phase between initial contact and initial swing for each step, derived from left palm keypoint during treadmill locomotion.",
            continuity="step",
        )
        swing_duration = TimeSeries(
            name="swing_duration",
            data=np.roll(
                step_timings_df["swing_duration"].to_numpy(), -1
            ),  # roll -1 so first value is not nan, and corresponds to first initial swing
            unit="seconds",
            resolution=0.008,
            timestamps=step_timings_df["initial_swing_timestamps"].to_numpy()
            + nwb_file_handle.acquisition["ElectricalSeries"].starting_time,
            description="Duration of swing phase between initial swing and next initial contact for each step, derived from left palm keypoint during treadmill locomotion.",
            continuity="step",
        )

        gait_events = BehavioralEvents(
            time_series={
                "initial_contact": contact_events,
                "initial_swing": swing_events,
                "stance_duration": stance_duration,
                "swing_duration": swing_duration,
            }
        )

        behavior_pm.add(gait_events)

        # Save updated NWB file
        # nwb_file_handle.set_modified()
        final_nwb_file_path = f"{session_id}.nwb"
        # with NWBHDF5IO(nwb_file_path, "w") as io_write:
        #     io_write.write(nwb_file_handle)
        with NWBHDF5IO(final_nwb_file_path, mode="w") as export_io:
            export_io.export(src_io=io, nwbfile=nwb_file_handle)

# now open each final NWB file to inspect
for session_id in session_ids:
    final_nwb_file_path = f"{session_id}.nwb"
    with NWBHDF5IO(final_nwb_file_path, "r") as io:
        nwb_file_handle = io.read()
        print(f"Session ID: {session_id}")
        print(nwb_file_handle)
        set_trace()

print("NWB files created successfully!")
