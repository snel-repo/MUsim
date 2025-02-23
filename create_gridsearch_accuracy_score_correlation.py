from pathlib import Path
from pdb import set_trace

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

# Set the directories
cwd = Path.cwd()
csv_folder = cwd / "plot4"
print(csv_folder)
sort_folder = Path(
    "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_2.25"
)

# Get CSV files which contain the datestring in the filename
# text_to_filter = "combined_performances_20241001-193"
text_to_filter = (
    "20250220-000001"  # "20250207-162455"  # "20250206-202428"  # "20241002-015642"
)
csv_files = list(csv_folder.glob(f"*{text_to_filter}*.csv"))
print(csv_files)

# Combine the CSV files
combo_csv = pd.concat([pd.read_csv(f) for f in csv_files])

# name first column as 'cluster'
combo_csv.rename(columns={"Unnamed: 0": "cluster"}, inplace=True)

# sort the 'noise_level' column in ascending order, then sort the 'accuracy' column in descending order
combo_csv = combo_csv.sort_values(["noise_level", "accuracy"], ascending=[True, False])

# filter for string match to get the scores of each path corresponding to each datestring
dates = combo_csv.datestring.values
matching_sort_folder_path_list = []
matching_sort_folder_score_list = []
type_I_mean_list = []
type_II_mean_list = []
snr_mean_list = []
frv_mean_list = []
emu_mean_list = []
emu_config_dict = dict()
# from each each date string match get: the path and the scores
for iDateStr in dates:
    this_sort_folder = next(sort_folder.glob(f"*{iDateStr}*"))
    matching_sort_folder_path_list.append(this_sort_folder)
    this_sort_folder_score = float(str(this_sort_folder).split("SCORE_")[-1][:5])
    matching_sort_folder_score_list.append(this_sort_folder_score)
    # load the emu_config.yaml
    yaml = YAML(typ="safe")
    data = yaml.load(this_sort_folder / "emu_config.yaml")
    emu_config_dict[iDateStr] = data
    # get individual score component means across units
    type_I_mean_list.append(
        np.array(emu_config_dict[iDateStr]["Results"]["type_I_scores"]).mean()
    )
    type_II_mean_list.append(
        np.array(emu_config_dict[iDateStr]["Results"]["type_II_scores"]).mean()
    )
    try:
        snr_mean_list.append(
            np.array(emu_config_dict[iDateStr]["Results"]["snr_scores"]).mean()
        )
    except:
        snr_mean_list.append(
            np.array(emu_config_dict[iDateStr]["Results"]["norm_snr_scores"]).mean()
        )
    frv_mean_list.append(
        np.array(
            emu_config_dict[iDateStr]["Results"]["firing_rate_validity_scores"]
        ).mean()
    )
    emu_mean_list.append(
        np.array(emu_config_dict[iDateStr]["Results"]["emusort_scores"]).mean()
    )

type_I_mean_arr = np.array(type_I_mean_list)
type_II_mean_arr = np.array(type_II_mean_list)
snr_mean_arr = np.array(snr_mean_list)
frv_mean_arr = np.array(frv_mean_list)
emu_mean_arr = np.array(emu_mean_list)

EMU_idxs = combo_csv["sort_type"] == "EMUsort"
KS4_idxs = combo_csv["sort_type"] == "Kilosort4"

EMUsort_vals = combo_csv[EMU_idxs]
Kilosort4_vals = combo_csv[KS4_idxs]
all_vals = combo_csv["accuracy"].to_numpy()

matching_sort_folder_score_arr = np.array(matching_sort_folder_score_list)

EMUsort_scores = matching_sort_folder_score_arr[EMU_idxs]
Kilosort4_scores = matching_sort_folder_score_arr[KS4_idxs]

# Total
coef = np.polyfit(matching_sort_folder_score_arr, all_vals, 1)
poly1d_fn = np.poly1d(coef)
print(f"folder sort score {matching_sort_folder_score_arr}")
print(f"all vals \n{all_vals}")
R = np.corrcoef(matching_sort_folder_score_arr, all_vals)[0, 1].round(4)
print(f"Overall correlation is: {R}")
R_emu = np.corrcoef(EMUsort_scores, all_vals[EMU_idxs])[0, 1].round(4)
print(f"EMUsort correlation is: {R_emu}")
R_ks4 = np.corrcoef(Kilosort4_scores, all_vals[KS4_idxs])[0, 1].round(4)
print(f"Kilosort4 correlation is: {R_ks4}")
print("\n")
############################################ Overall
# overall scores
print(type_I_mean_arr)
type_I_R = np.corrcoef(type_I_mean_arr, all_vals)[0, 1].round(4)
print(f"Correlation with Type I score is: {type_I_R}")
print(type_II_mean_arr)
type_II_R = np.corrcoef(type_II_mean_arr, all_vals)[0, 1].round(4)
print(f"Correlation with Type II score is: {type_II_R}")
print(snr_mean_arr)
snr_R = np.corrcoef(snr_mean_arr, all_vals)[0, 1].round(4)
print(f"Correlation with SNR score is: {snr_R}")
print(frv_mean_arr)
frv_R = np.corrcoef(
    frv_mean_arr,
    all_vals,
)[
    0, 1
].round(4)
print(f"Correlation with FR violation score is: {frv_R}")
print(emu_mean_arr)
emu_R = np.corrcoef(emu_mean_arr, all_vals)[0, 1].round(4)
print(f"Correlation with EMUsort score is: {emu_R}")
print("\n")
############################################ EMUsort
# Now compute the correlation with each component of the score EMUsort
print(type_I_mean_arr[EMU_idxs])
type_I_R_emu = np.corrcoef(type_I_mean_arr[EMU_idxs], all_vals[EMU_idxs])[0, 1].round(4)
print(f"Correlation with EMU Type I score is: {type_I_R_emu}")

print(type_II_mean_arr[EMU_idxs])
type_II_R_emu = np.corrcoef(type_II_mean_arr[EMU_idxs], all_vals[EMU_idxs])[0, 1].round(
    4
)
print(f"Correlation with EMU Type II score is: {type_II_R_emu}")

print(snr_mean_arr[EMU_idxs])
snr_R_emu = np.corrcoef(snr_mean_arr[EMU_idxs], all_vals[EMU_idxs])[0, 1].round(4)
print(f"Correlation with EMU SNR score is: {snr_R_emu}")

print(frv_mean_arr[EMU_idxs])
frv_R_emu = np.corrcoef(
    frv_mean_arr[EMU_idxs],
    all_vals[EMU_idxs],
)[
    0, 1
].round(4)
print(f"Correlation with EMU FR violation score is: {frv_R_emu}")
print(emu_mean_arr[EMU_idxs])
emu_R_emu = np.corrcoef(emu_mean_arr[EMU_idxs], all_vals[EMU_idxs])[0, 1].round(4)
print(f"Correlation with EMUsort score is: {emu_R_emu}")
print("\n")
############################################ Kilosort4
# Now compute the correlation with each component of the score Kilosort4
print(type_I_mean_arr[KS4_idxs])
type_I_R_ks4 = np.corrcoef(type_I_mean_arr[KS4_idxs], all_vals[KS4_idxs])[0, 1].round(4)
print(f"Correlation with KS4 Type I score is: {type_I_R_ks4}")
print(type_II_mean_arr[KS4_idxs])
type_II_R_ks4 = np.corrcoef(type_II_mean_arr[KS4_idxs], all_vals[KS4_idxs])[0, 1].round(
    4
)
print(f"Correlation with KS4 Type II score is: {type_II_R_ks4}")
print(snr_mean_arr[KS4_idxs])
snr_R_ks4 = np.corrcoef(snr_mean_arr[KS4_idxs], all_vals[KS4_idxs])[0, 1].round(4)
print(f"Correlation with KS4 SNR score is: {snr_R_ks4}")
print(frv_mean_arr[KS4_idxs])
frv_R_ks4 = np.corrcoef(
    frv_mean_arr[KS4_idxs],
    all_vals[KS4_idxs],
)[
    0, 1
].round(4)
print(f"Correlation with KS4 FR violation score is: {frv_R_ks4}")
print(emu_mean_arr[KS4_idxs])
emu_R_ks4 = np.corrcoef(emu_mean_arr[KS4_idxs], all_vals[KS4_idxs])[0, 1].round(4)
print(f"Correlation with EMUsort score is: {emu_R_ks4}")

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=Kilosort4_scores,
        y=Kilosort4_vals["accuracy"],
        line_color="firebrick",
        name="Kilosort4",
        mode="markers",
        marker_size=30,
        marker_opacity=0.9,
    )
)

fig.add_trace(
    go.Scatter(
        x=EMUsort_scores,
        y=EMUsort_vals["accuracy"],
        line_color="#666565",
        name="EMUsort",
        mode="markers",
        marker_size=30,
        marker_opacity=0.9,
    )
)

fig.add_trace(
    go.Scatter(
        x=sorted(matching_sort_folder_score_arr),
        y=poly1d_fn(sorted(matching_sort_folder_score_arr)),
        line_color="black",
        name=f"Pearson r: {R}",
        mode="lines",
        line_dash="dash",
        line_width=10,
        opacity=1,
    )
)

fig.update_layout(
    template="plotly_white",
    font=dict(family="Open Sans", size=24, color="black", weight="bold"),
    title_font=dict(size=36, color="black", family="Open Sans", weight="bold"),
    title="EMUsort Scores Predict Accuracy",  # "Overlapping Spikes Only",
    # title="All Spikes Evaluated",
    # title="Sort Accuracy Distributions (All Spikes Evaluated)<br><sup>25 parameter combinations each</sup>",
    xaxis_title="EMUsort Score",
    yaxis_title="Sort Accuracy",
    # showlegend=False,
    # showgrid=True,
)

# fig.update_yaxes(range=[0.55, 1])
# fig.update_yaxes(range=[x - 0 for x in [0.55, 1]])

# if Path(cwd / "plot7").is_dir() is False:
#     Path(cwd / "plot7").mkdir()

# Save the plot
# fig.write_image(f"plot7/violin_accuracy_EMUsort_vs_Kilosort4_{text_to_filter}.svg")
fig.show()
