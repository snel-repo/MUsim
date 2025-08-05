import datetime
from pathlib import Path

from MUsim import MUsim

# ground_truth_path = "spikes_files/spikes_20250219-212813_human_20231003_13MU_SNR-1-from_data_jitter-0.2std_method-KS_templates_1-files"
ground_truth_path = "spikes_files/spikes_20250428-183259_godzilla_20221116_10MU_14CH_SNR-1-from_data_jitter-0.2std_method-KS_templates_12-files"

random_seed_entropy = 218530072159092100005306709809425040261
mu_GT = MUsim(random_seed_entropy)
mu_GT.load_MUs(
    # npy_file_path
    ground_truth_path,
    1 / 30000,
    load_as="trial",
    slice=[0, 1],
    load_type="kilosort",
)
mu_GT.save_spikes(
    # f"synthetic_spikes_from_{session_name}_using_{chosen_bodypart_to_load}.npy"
    f"{ground_truth_path}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.npy",
    save_as="boolean",
)
