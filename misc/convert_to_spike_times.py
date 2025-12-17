import datetime
from pathlib import Path

from MUsim import MUsim

# ground_truth_path = "spikes_20240607-143039_godzilla_20221117_10MU_SNR-1-from_data_jitter-0std_method-KS_templates_12-files.npy"
ground_truth_path = "/home/smoconn/git/MUsim/spikes_files/spikes_20240621-135748_godzilla_20221117_10MU_SNR-1-from_data_jitter-2std_method-KS_templates_12-files.npy"
ground_truth_path = Path().joinpath("spikes_files", ground_truth_path)

random_seed_entropy = 218530072159092100005306709809425040261
mu_GT = MUsim(random_seed_entropy)
mu_GT.load_MUs(
    # npy_file_path
    ground_truth_path,
    1 / 30000,
    load_as="trial",
    slice=[0, 1],
    load_type="MUsim",
)
mu_GT.save_spikes(
    # f"synthetic_spikes_from_{session_name}_using_{chosen_bodypart_to_load}.npy"
    f"spikes_files/{ground_truth_path.stem}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    save_as="indexes",
)
