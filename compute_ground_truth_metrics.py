# IMPORT packages
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots

from MUsim import MUsim

start_time = datetime.now()  # begin timer for script execution time


# define a function to convert a timedelta object to a pretty string representation
def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def compute_precision(num_matches, num_kilosort_spikes):
    return num_matches / num_kilosort_spikes


def compute_recall(num_matches, num_ground_truth_spikes):
    return num_matches / num_ground_truth_spikes


def compute_accuracy(num_matches, num_kilosort_spikes, num_ground_truth_spikes):
    return num_matches / (num_kilosort_spikes + num_ground_truth_spikes - num_matches)


def compute_spike_matches(ground_truth_spikes, kilosort_spikes):
    # ground_truth_spikes and kilosort_spikes are numpy arrays with shape (num_bins, num_units)
    # rows are time bins, columns are units
    # each entry is a spike count
    # compute the number of spikes in each time bin
    num_ground_truth_spikes = np.sum(ground_truth_spikes, axis=0)
    num_kilosort_spikes = np.sum(kilosort_spikes, axis=0)
    # compute the number of matches in each time bin
    num_matches = np.sum(ground_truth_spikes * kilosort_spikes, axis=0)
    return num_matches, num_kilosort_spikes, num_ground_truth_spikes


# set parameters
time_frame = [0, 1]  # must be between 0 and 1
ephys_fs = 30000  # Hz
bin_width_for_comparison = 1  # ms
nt0 = 61  # 2.033 ms
random_seed_entropy = 75092699954400878964964014863999053929  # int
clusters_to_take_from = [[18, 2, 11, 0, 4, 10, 1, 9]]  # list of lists
num_motor_units = len(clusters_to_take_from[0])
plot_template = "plotly_white"
plot2_xlim = [0, 1]
show_plot1 = False
show_plot2 = True
save_png_plot1 = False
save_png_plot2 = False
save_html_plot1 = False
save_html_plot2 = False

## load ground truth data
ground_truth_path = Path("spikes_20221116_godzilla_SNR-None_jitter-0std_files-1.npy")
ground_truth_spikes = np.load(ground_truth_path)

## load Kilosort data
# paths to the folders containing the Kilosort data
paths_to_KS_session_folders = [
    Path(
        "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221116/"
    ),
]
sorts_from_each_path_to_load = ["20231006_173000"]  # , ["20230923_125645"], ["20230923_125645"]]

# find the folder name which ends in _myo and append to the paths_to_session_folders
paths_to_each_myo_folder = []
for iDir in paths_to_KS_session_folders:
    myo = [f for f in iDir.iterdir() if (f.is_dir() and f.name.endswith("_myo"))]
    assert len(myo) == 1, "There should only be one _myo folder in each session folder"
    paths_to_each_myo_folder.append(myo[0])
# inside each _myo folder, find the folder name which constains sort_from_each_path_to_load string
list_of_paths_to_sorted_folders = []
for iPath in paths_to_each_myo_folder:
    matches = [
        f
        for f in iPath.iterdir()
        if f.is_dir() and any(s in f.name for s in sorts_from_each_path_to_load)
    ]
    assert len(matches) == 1, "There should only be one sort folder match in each _myo folder"
    list_of_paths_to_sorted_folders.append(matches[0])

# use MUsim object to load and rebin ground truth data
mu_GT = MUsim(random_seed_entropy)
mu_GT.sample_rate = 1 / ephys_fs
mu_GT.load_MUs(
    # npy_file_path
    ground_truth_path,
    1 / 30000,
    load_as="trial",
    slice=time_frame,
    load_type="MUsim",
)
mu_GT.rebin_trials(bin_width_for_comparison / 1000)  # rebin to 1 ms bins
mu_GT.save_spikes("./GT_spikes.npy")
ground_truth_spikes = mu_GT.spikes[-1]  # shape is (num_bins, num_units)

# use MUsim object to load and rebin Kilosort data
mu_KS = MUsim(random_seed_entropy)
mu_KS.sample_rate = 1 / ephys_fs
mu_KS.load_MUs(
    # npy_file_path
    list_of_paths_to_sorted_folders[0],
    1 / 30000,
    load_as="trial",
    slice=time_frame,
    load_type="kilosort",
)
mu_KS.rebin_trials(bin_width_for_comparison / 1000)  # rebin to 1 ms bins
mu_KS.save_spikes("./KS_spikes.npy")
kilosort_spikes = mu_KS.spikes[-1]  # shape is (num_bins, num_units)

# subselect clusters from kilosort spikes
# kilosort_spikes = kilosort_spikes[:, clusters_to_take_from[0]]

# ensure kilosort_spikes and ground_truth_spikes have the same shape
# add more kilosort bins to match ground truth (fill with zeros)
if kilosort_spikes.shape[0] < ground_truth_spikes.shape[0]:
    kilosort_spikes = np.vstack(
        (
            kilosort_spikes,
            np.zeros(
                (ground_truth_spikes.shape[0] - kilosort_spikes.shape[0], kilosort_spikes.shape[1])
            ),
        )
    )
assert (
    kilosort_spikes.shape == ground_truth_spikes.shape
), f"Spikes arrays have different shapes: {kilosort_spikes.shape} and {ground_truth_spikes.shape}"

# compute metrics
num_matches, num_kilosort_spikes, num_ground_truth_spikes = compute_spike_matches(
    ground_truth_spikes, kilosort_spikes
)

precision = compute_precision(num_matches, num_kilosort_spikes)
recall = compute_recall(num_matches, num_ground_truth_spikes)
accuracy = compute_accuracy(num_matches, num_kilosort_spikes, num_ground_truth_spikes)

if show_plot1 or save_png_plot1 or save_html_plot1:
    ### plot 1: bar plot of spike counts
    # now create an overlay plot of the two plots above. Do not use subplots, but use two y axes
    # make bar plot of total spike counts use left y axis
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=np.arange(0, num_motor_units),
            y=num_ground_truth_spikes,
            name="Ground Truth",
            marker_color="rgb(55, 83, 109)",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Bar(
            x=np.arange(0, num_motor_units),
            y=num_kilosort_spikes,
            name="Kilosort",
            marker_color="rgb(26, 118, 255)",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, num_motor_units),
            y=precision,
            mode="lines+markers",
            name="Precision",
            line=dict(width=4),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, num_motor_units),
            y=recall,
            mode="lines+markers",
            name="Recall",
            line=dict(width=4),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, num_motor_units),
            y=accuracy,
            mode="lines+markers",
            name="Accuracy",
            line=dict(width=4),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"<b>Comparison of MUsort Performance to Ground Truth, {bin_width_for_comparison} ms Bins</b>",
        xaxis_title="<b>KS Cluster ID</b>",
        legend_title="Ground Truth Metrics",
        template=plot_template,
        yaxis=dict(title="<b>Spike Count</b>"),
        yaxis2=dict(title="<b>Metric Score</b>", range=[0, 1], overlaying="y", side="right"),
    )
    if save_png_plot1:
        # make 1080p image
        fig.write_image(
            f"KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms.png",
            width=1920,
            height=1080,
        )
    if save_html_plot1:
        fig.write_html(
            f"KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot1:
        fig.show()

# create a new array for each type of error (false positive, false negative, true positive)
false_positive_spikes = np.zeros(kilosort_spikes.shape)
false_negative_spikes = np.zeros(kilosort_spikes.shape)
true_positive_spikes = np.zeros(kilosort_spikes.shape)
# loop through each time bin and each unit
for iBin in range(kilosort_spikes.shape[0]):
    for iUnit in range(kilosort_spikes.shape[1]):
        # if kilosort spike and ground truth spike, true positive
        if kilosort_spikes[iBin, iUnit] == 1 and ground_truth_spikes[iBin, iUnit] == 1:
            true_positive_spikes[iBin, iUnit] = 1
        # if kilosort spike but no ground truth spike, false positive
        elif kilosort_spikes[iBin, iUnit] == 1 and ground_truth_spikes[iBin, iUnit] == 0:
            false_positive_spikes[iBin, iUnit] = 1
        # if no kilosort spike but ground truth spike, false negative
        elif kilosort_spikes[iBin, iUnit] == 0 and ground_truth_spikes[iBin, iUnit] == 1:
            false_negative_spikes[iBin, iUnit] = 1
print(
    f"Total number of Kilosort spikes: {np.sum(kilosort_spikes)} ({np.sum(kilosort_spikes)/np.sum(ground_truth_spikes)*100:.2f}%)"
)
for iUnit in range(num_motor_units):
    print(
        f"Total number of Kilosort spikes in unit {iUnit+1}: {np.sum(kilosort_spikes[:,iUnit])} ({np.sum(kilosort_spikes[:,iUnit])/np.sum(ground_truth_spikes[:,iUnit])*100:.2f}%)"
    )
print(
    f"Total number of false positive spikes: {np.sum(false_positive_spikes)} ({np.sum(false_positive_spikes)/np.sum(kilosort_spikes)*100:.2f}%)"
)
print(
    f"Total number of false negative spikes: {np.sum(false_negative_spikes)} ({np.sum(false_negative_spikes)/np.sum(ground_truth_spikes)*100:.2f}%)"
)
print(
    f"Total number of true positive spikes: {np.sum(true_positive_spikes)} ({np.sum(true_positive_spikes)/np.sum(ground_truth_spikes)*100:.2f}%)"
)
if show_plot2 or save_png_plot2 or save_html_plot2:
    ### plot 2: spike trains

    # now plot the spike trains, emphasizing each type of error with a different color
    # include an event plot of the kilosort and ground truth spikes for reference
    # make a subplot for each unit
    subtitles = [f"Unit {iUnit//2+1}" for iUnit in range(2 * num_motor_units) if iUnit % 2 == 0]
    # insert a "" between each list element
    for i in range(len(subtitles)):
        subtitles.insert(2 * i + 1, "")
    fig = subplots.make_subplots(
        rows=2 * num_motor_units,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subtitles,
        specs=[[{"secondary_y": True}] for i in range(2 * num_motor_units)],
    )

    for iUnit in range(num_motor_units):
        # add event plots of the kilosort and ground truth spikes, color units according to rainbow
        # add a vertical offset to Kilosort spikes to separate them from ground truth spikes
        # darken the ground truth spikes to distinguish them from Kilosort spikes
        color_KS = "hsl(" + str(iUnit / float(num_motor_units) * 360) + ",100%,50%)"
        color_GT = "hsl(" + str(iUnit / float(num_motor_units) * 360) + ",100%,25%)"
        fig.add_trace(
            go.Scatter(
                x=np.where(kilosort_spikes[:, iUnit] == 1)[0] * bin_width_for_comparison,
                y=np.ones(np.sum(kilosort_spikes[:, iUnit] == 1)) + 1,
                mode="markers",
                name="Kilosort",
                marker_symbol="line-ns",
                marker=dict(
                    color=color_KS,
                    line_color=color_KS,
                    line_width=1.2,
                    size=10,
                ),
                opacity=1,
            ),
            row=2 * iUnit + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.where(ground_truth_spikes[:, iUnit] == 1)[0] * bin_width_for_comparison,
                y=np.ones(np.sum(ground_truth_spikes[:, iUnit] == 1)),
                mode="markers",
                name="Ground Truth",
                marker_symbol="line-ns",
                marker=dict(
                    color=color_GT,
                    line_color=color_GT,
                    line_width=1.2,
                    size=10,
                ),
            ),
            row=2 * iUnit + 1,
            col=1,
        )
        # add event plots of the false positive, false negative, and true positive spikes
        # add a vertical offset to each error type to separate them from each other
        fig.add_trace(
            go.Scatter(
                x=np.where(false_positive_spikes[:, iUnit] == 1)[0] * bin_width_for_comparison,
                y=np.ones(np.sum(false_positive_spikes[:, iUnit] == 1)),
                mode="markers",
                name="False Positive",
                marker=dict(color="red", size=5, symbol="x"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.where(false_negative_spikes[:, iUnit] == 1)[0] * bin_width_for_comparison,
                y=np.ones(np.sum(false_negative_spikes[:, iUnit] == 1)) + 1,
                mode="markers",
                name="False Negative",
                marker=dict(color="orange", size=5, symbol="x"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=np.where(true_positive_spikes[:, iUnit] == 1)[0] * bin_width_for_comparison,
                y=np.ones(np.sum(true_positive_spikes[:, iUnit] == 1)) + 2,
                mode="markers",
                name="True Positive",
                marker=dict(color="green", size=5, symbol="diamond-tall"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )

    fig.update_layout(
        title=f"<b>Comparison of Kilosort and Ground Truth Spike Trains, {bin_width_for_comparison} ms Bins</b>",
        template=plot_template,
    )

    left_bound = int(round(plot2_xlim[0] * len(kilosort_spikes) * bin_width_for_comparison))
    right_bound = int(round(plot2_xlim[1] * len(kilosort_spikes) * bin_width_for_comparison))
    fig.update_xaxes(
        title_text="<b>Time (ms)</b>",
        row=2 * num_motor_units,
        col=1,
        range=[left_bound, right_bound],
    )

    # remove y axes numbers from all plots
    fig.update_yaxes(showticklabels=False)

    if save_png_plot2:
        # make 1080p image
        fig.write_image(
            f"KS_vs_GT_spike_trains_{bin_width_for_comparison}ms.png", width=1920, height=1080
        )
    if save_html_plot2:
        fig.write_html(
            f"KS_vs_GT_spike_trains_{bin_width_for_comparison}ms.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot2:
        fig.show()

finish_time = datetime.now()
time_elapsed = finish_time - start_time
# use strfdelta to format time elapsed prettily
print(
    (
        "Time elapsed: "
        f"{strfdelta(time_elapsed, '{hours} hours, {minutes} minutes, {seconds} seconds')}"
    )
)
