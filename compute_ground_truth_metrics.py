# IMPORT packages
import subprocess
from datetime import datetime
from pathlib import Path
from pdb import set_trace

import colorlover as cl
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
from pandas import DataFrame as df
from scipy.signal import correlate, correlation_lags

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


def compare_spike_trains(
    ground_truth_path,
    random_seed_entropy,
    bin_width_for_comparison,
    ephys_fs,
    time_frame,
    list_of_paths_to_sorted_folders,
    clusters_to_take_from,
):
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

    # subselect clusters in mu_KS using clusters_to_take_from, now in that specified order
    mu_KS.spikes[-1] = mu_KS.spikes[-1][:, clusters_to_take_from]

    # ensure kilosort_spikes and ground_truth_spikes have the same shape
    # add more kilosort bins to match ground truth
    # (fill missing allocation with zeros due to no spikes near end of recording)
    if mu_KS.spikes[-1].shape[0] < mu_GT.spikes[-1].shape[0]:
        mu_KS.spikes[-1] = np.vstack(
            (
                mu_KS.spikes[-1],
                np.zeros(
                    (
                        mu_GT.spikes[-1].shape[0] - mu_KS.spikes[-1].shape[0],
                        mu_KS.spikes[-1].shape[1],
                    )
                ),
            )
        )
    assert (
        mu_KS.spikes[-1].shape == mu_GT.spikes[-1].shape
    ), f"Spikes arrays have different shapes: {mu_KS.spikes[-1].shape} and {mu_GT.spikes[-1].shape}"

    # compute the correlation between the two spike trains for each unit
    # use the correlation to determine the shift for each unit
    # use the shift to align the spike trains
    # use the aligned spike trains to compute the metrics

    min_delay_ms = -0.5  # ms
    max_delay_ms = 0.5  # ms
    min_delay_samples = int(round(min_delay_ms * ephys_fs / 1000))
    max_delay_samples = int(round(max_delay_ms * ephys_fs / 1000))

    for iUnit in range(mu_KS.spikes[-1].shape[1]):
        correlation = correlate(
            mu_KS.spikes[-1][:, iUnit], mu_GT.spikes[-1][:, iUnit], "same"
        )
        lags = correlation_lags(
            len(mu_KS.spikes[-1][:, iUnit]), len(mu_GT.spikes[-1][:, iUnit]), "same"
        )
        lag_constraint_idxs = np.where(
            np.logical_and(lags >= min_delay_samples, lags <= max_delay_samples)
        )[0]
        # limit the lags to range from min_delay_samples to max_delay_samples
        # lags = lags[(lags >= min_delay_samples) & (lags <= max_delay_samples)]
        # find the lag with the highest correlation
        max_correlation_index = np.argmax(correlation[lag_constraint_idxs])
        # find the lag with the highest correlation in the opposite direction
        min_correlation_index = np.argmin(correlation[lag_constraint_idxs])
        # if the max correlation is positive, shift the kilosort spikes to the right
        # if the max correlation is negative, shift the kilosort spikes to the left
        if (
            correlation[lag_constraint_idxs][max_correlation_index]
            > correlation[lag_constraint_idxs][min_correlation_index]
        ):
            shift = lags[lag_constraint_idxs][max_correlation_index]
        else:
            shift = lags[lag_constraint_idxs][min_correlation_index]
        # shift the kilosort spikes
        mu_KS.spikes[-1][:, iUnit] = np.roll(mu_KS.spikes[-1][:, iUnit], -shift)
        # make sure shift hasn't gone near the edge of the min or max delay
        if shift <= min_delay_samples + 1 or shift >= max_delay_samples - 1:
            print(
                f"WARNING: Shifted Kilosort spikes for unit {clusters_to_take_from[iUnit]} by {shift} samples"
            )

    # rebin the spike trains to the bin width for comparison
    mu_GT.rebin_trials(
        bin_width_for_comparison / 1000
    )  # rebin to bin_width_for_comparison ms bins
    mu_GT.save_spikes("./GT_spikes.npy")
    ground_truth_spikes = mu_GT.spikes[-1]  # shape is (num_bins, num_units)

    mu_KS.rebin_trials(
        bin_width_for_comparison / 1000
    )  # rebin to bin_width_for_comparison ms bins
    mu_KS.save_spikes("./KS_spikes.npy")
    kilosort_spikes = mu_KS.spikes[-1]  # shape is (num_bins, num_units)

    # compute false positive, false negative, and true positive spikes
    # create a new array for each type of error (false positive, false negative, true positive)
    true_positive_spikes = np.zeros(kilosort_spikes.shape, dtype=int)
    false_positive_spikes = np.zeros(kilosort_spikes.shape, dtype=int)
    false_negative_spikes = np.zeros(kilosort_spikes.shape, dtype=int)
    # loop through each time bin and each unit
    for iBin in range(kilosort_spikes.shape[0]):
        for iUnit in range(kilosort_spikes.shape[1]):
            # check cases in order of most likely to least likely for efficiency
            if (
                kilosort_spikes[iBin, iUnit] == 0
                and ground_truth_spikes[iBin, iUnit] == 0
            ):
                continue  # no spike in this bin for either kilosort or ground truth, go to next bin
            # if kilosort spike and ground truth spike, true positive
            elif (
                kilosort_spikes[iBin, iUnit] == 1
                and ground_truth_spikes[iBin, iUnit] == 1
            ):
                true_positive_spikes[iBin, iUnit] = 1
            # if kilosort spike but no ground truth spike, false positive
            elif (
                kilosort_spikes[iBin, iUnit] >= 1
                and ground_truth_spikes[iBin, iUnit] == 0
            ):
                false_positive_spikes[iBin, iUnit] = kilosort_spikes[iBin, iUnit]
            # if no kilosort spike but ground truth spike, false negative
            elif (
                kilosort_spikes[iBin, iUnit] == 0
                and ground_truth_spikes[iBin, iUnit] >= 1
            ):
                false_negative_spikes[iBin, iUnit] = ground_truth_spikes[iBin, iUnit]
            # now, must check for numbers larger than 1, and handle those cases appropriately
            elif (
                kilosort_spikes[iBin, iUnit] > 1
                and ground_truth_spikes[iBin, iUnit] == 1
            ):
                # if kilosort spike and ground truth spike, true positive
                true_positive_spikes[iBin, iUnit] = 1
                # if kilosort spike but no ground truth spike, false positive
                false_positive_spikes[iBin, iUnit] = kilosort_spikes[iBin, iUnit] - 1
            elif (
                kilosort_spikes[iBin, iUnit] == 1
                and ground_truth_spikes[iBin, iUnit] > 1
            ):
                # if kilosort spike and ground truth spike, true positive
                true_positive_spikes[iBin, iUnit] = 1
                # if no kilosort spike but ground truth spike, false negative
                false_negative_spikes[iBin, iUnit] = (
                    ground_truth_spikes[iBin, iUnit] - 1
                )
            elif (
                kilosort_spikes[iBin, iUnit] > 1
                and ground_truth_spikes[iBin, iUnit] > 1
            ):
                # if kilosort spike and ground truth spike, true positive
                true_positive_spikes[iBin, iUnit] = min(
                    kilosort_spikes[iBin, iUnit], ground_truth_spikes[iBin, iUnit]
                )
                if kilosort_spikes[iBin, iUnit] > ground_truth_spikes[iBin, iUnit]:
                    # if kilosort spike but no ground truth spike, false positive
                    false_positive_spikes[iBin, iUnit] = (
                        kilosort_spikes[iBin, iUnit] - ground_truth_spikes[iBin, iUnit]
                    )
                elif kilosort_spikes[iBin, iUnit] < ground_truth_spikes[iBin, iUnit]:
                    # if no kilosort spike but ground truth spike, false negative
                    false_negative_spikes[iBin, iUnit] = (
                        ground_truth_spikes[iBin, iUnit] - kilosort_spikes[iBin, iUnit]
                    )
            else:
                raise ValueError(
                    (
                        f"Unhandled case: kilosort_spikes[{iBin}, {iUnit}] = "
                        f"{kilosort_spikes[iBin, iUnit]}, ground_truth_spikes["
                        f"{iBin}, {iUnit}] = {ground_truth_spikes[iBin, iUnit]}"
                    )
                )

    num_matches = np.sum(true_positive_spikes, axis=0)
    num_kilosort_spikes = np.sum(kilosort_spikes, axis=0)
    num_ground_truth_spikes = np.sum(ground_truth_spikes, axis=0)

    precision = compute_precision(num_matches, num_kilosort_spikes)
    recall = compute_recall(num_matches, num_ground_truth_spikes)
    accuracy = compute_accuracy(
        num_matches, num_kilosort_spikes, num_ground_truth_spikes
    )

    # time = iBin * bin_width_for_comparison / 1000
    # if iUnit == 0 and time > 7.998 and time < 8.002:
    #     print("debugging")
    #     set_trace()

    # prettily print the results for each unit
    # print(f"Bin width for comparison: {bin_width_for_comparison} ms")
    # print(f"Number of matches: {num_matches}")
    # print(f"Number of Kilosort spikes: {num_kilosort_spikes}")
    # print(f"Number of ground truth spikes: {num_ground_truth_spikes}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"Accuracy: {accuracy}")

    return (
        kilosort_spikes,
        ground_truth_spikes,
        accuracy,
        precision,
        recall,
        num_matches,
        num_kilosort_spikes,
        num_ground_truth_spikes,
        true_positive_spikes,
        false_positive_spikes,
        false_negative_spikes,
    )


def plot1(
    num_ground_truth_spikes,
    num_kilosort_spikes,
    precision,
    recall,
    accuracy,
    bin_width_for_comparison,
    clusters_to_take_from,
    sort_from_each_path_to_load,
    plot_template,
    show_plot1,
    save_png_plot1,
    save_svg_plot1,
    save_html_plot1,
    figsize=(1920, 1080),
):
    # get suffix after the KS folder name, which is the repo branch name for that sort
    PPP_branch_name = list_of_paths_to_sorted_folders[0].name.split("_")[-1]
    sort_type = "Kilosort" if PPP_branch_name == "KS" else "MUsort"

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
            name=sort_type,
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
        title=f"<b>Comparison of {sort_type} Performance to Ground Truth, {bin_width_for_comparison} ms Bins</b><br><sup>Sort: {sort_from_each_path_to_load}</sup>",
        xaxis_title="<b>KS Cluster ID</b>",
        legend_title="Ground Truth Metrics",
        template=plot_template,
        yaxis=dict(title="<b>Spike Count</b>"),
        yaxis2=dict(
            title="<b>Metric Score</b>", range=[0, 1], overlaying="y", side="right"
        ),
    )
    # update the x tick label of the bar graph to match the cluster ID
    fig.update_xaxes(
        ticktext=[
            f"Unit {clusters_to_take_from[iUnit]}" for iUnit in range(num_motor_units)
        ],
        tickvals=np.arange(0, num_motor_units),
    )

    if save_png_plot1:
        fig.write_image(
            f"KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot1:
        fig.write_image(
            f"KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot1:
        fig.write_html(
            f"KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot1:
        fig.show()


def plot2(
    kilosort_spikes,
    ground_truth_spikes,
    false_positive_spikes,
    false_negative_spikes,
    true_positive_spikes,
    bin_width_for_comparison,
    clusters_to_take_from,
    sort_from_each_path_to_load,
    plot_template,
    show_plot2,
    save_png_plot2,
    save_svg_plot2,
    save_html_plot2,
    figsize=(1920, 1080),
):
    # make a subplot for each unit
    subtitles = [
        f"Unit {clusters_to_take_from[iUnit]}" for iUnit in range(num_motor_units)
    ]
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
                x=np.where(kilosort_spikes[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=kilosort_spikes[np.where(kilosort_spikes[:, iUnit] >= 1)[0], iUnit]
                + 0.5,
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
                x=np.where(ground_truth_spikes[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=ground_truth_spikes[
                    np.where(ground_truth_spikes[:, iUnit] >= 1)[0], iUnit
                ],
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
                x=np.where(false_positive_spikes[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=np.ones(np.sum(false_positive_spikes[:, iUnit] >= 1)),
                mode="markers",
                name="False Positive",
                marker=dict(color="red", size=5, symbol="x"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.where(false_negative_spikes[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=np.ones(np.sum(false_negative_spikes[:, iUnit] >= 1)) + 1,
                mode="markers",
                name="False Negative",
                marker=dict(color="orange", size=5, symbol="x"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.where(true_positive_spikes[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=np.ones(np.sum(true_positive_spikes[:, iUnit] >= 1)) + 2,
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

    left_bound = int(
        round(plot2_xlim[0] * len(kilosort_spikes) * bin_width_for_comparison / 1000)
    )
    right_bound = int(
        round(plot2_xlim[1] * len(kilosort_spikes) * bin_width_for_comparison / 1000)
    )
    fig.update_xaxes(
        title_text="<b>Time (s)</b>",
        row=2 * num_motor_units,
        col=1,
        range=[left_bound, right_bound],
    )

    # remove y axes numbers from all plots
    fig.update_yaxes(showticklabels=False)
    # get suffix after the KS folder name, which is the repo branch name for that sort
    PPP_branch_name = list_of_paths_to_sorted_folders[0].name.split("_")[-1]

    # append sort name instead of time stamp
    if save_png_plot2:
        fig.write_image(
            f"KS_vs_GT_spike_trains_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot2:
        fig.write_image(
            f"KS_vs_GT_spike_trains_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot2:
        fig.write_html(
            f"KS_vs_GT_spike_trains_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.html",
        )
    if show_plot2:
        fig.show()


def plot3(
    bin_width,
    precision,
    recall,
    accuracy,
    num_motor_units,
    clusters_to_take_from,
    sort_from_each_path_to_load,
    plot_template,
    show_plot3,
    save_png_plot3,
    save_svg_plot3,
    save_html_plot3,
    figsize=(1920, 1080),
):
    # this plot shows the performance of MUsort across different bin widths, with 1 trace per motor unit
    # put a subplot for each metric, but give a different color range for each metric. Make it flexible
    # to the number of motor units, then interpolate the color for each motor unit
    fig = subplots.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        subplot_titles=["Precision", "Recall", "Accuracy"],
    )

    # interpolate within darker half the color map to get as many colors as there are motor units
    precision_color_map = cl.interp(
        cl.scales["9"]["seq"]["Greens"][4:9], num_motor_units
    )
    recall_color_map = cl.interp(cl.scales["9"]["seq"]["Oranges"][4:9], num_motor_units)
    accuracy_color_map = cl.interp(cl.scales["9"]["seq"]["Blues"][4:9], num_motor_units)

    metric_color_maps = [
        precision_color_map,
        recall_color_map,
        accuracy_color_map,
    ]

    metric_values = [precision, recall, accuracy]
    for iMetric in range(len(metric_values)):
        for iUnit in range(num_motor_units):
            fig.add_trace(
                go.Scatter(
                    x=bin_width,
                    y=metric_values[iMetric][:, iUnit],
                    mode="lines+markers",
                    name=f"Unit {clusters_to_take_from[iUnit]}",
                    line=dict(width=4),
                    marker=dict(
                        color=metric_color_maps[iMetric][iUnit],
                        size=10,
                    ),
                ),
                row=iMetric + 1,
                col=1,
            )

    fig.update_layout(
        title="<b>Comparison of MUsort Performance to Ground Truth, Across Bin Widths</b>",
        legend_title="Ground Truth Metrics",
        template=plot_template,
        # yaxis=dict(title="<b>Metric Score</b>", range=[0.85, 1.05]),
    )

    # make sure each row has the same y axis range
    for iRow in range(3):
        fig.update_yaxes(
            title_text="<b>Metric Score</b>",
            row=iRow + 1,
            col=1,
            range=[0.5, 1.01],
        )

    fig.update_yaxes(matches="y")

    fig.update_xaxes(
        title_text="<b>Bin Width (ms)</b>",
        row=3,
        col=1,
        # range=[0, 8],
    )

    # get suffix after the KS folder name, which is the repo branch name for that sort
    PPP_branch_name = list_of_paths_to_sorted_folders[0].name.split("_")[-1]

    if save_png_plot3:
        fig.write_image(
            f"KS_vs_GT_bin_width_comparison_{sort_from_each_path_to_load}_{PPP_branch_name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot3:
        fig.write_image(
            f"KS_vs_GT_bin_width_comparison_{sort_from_each_path_to_load}_{PPP_branch_name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot3:
        fig.write_html(
            f"KS_vs_GT_bin_width_comparison_{sort_from_each_path_to_load}_{PPP_branch_name}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot3:
        fig.show()


if __name__ == "__main__":
    # set parameters
    parallel = True
    use_custom_merge_clusters = False
    time_frame = [0, 1]  # must be between 0 and 1
    ephys_fs = 30000  # Hz
    # range from 0.125 ms to 8 ms in log2 increments
    xstart = np.log2(0.125)
    bin_widths_for_comparison = np.logspace(xstart, -xstart, num=13, base=2)
    # index of which bin width of bin_widths_for_comparison to show in plots
    iShow = 6

    nt0 = 61  # 2.033 ms
    random_seed_entropy = 218530072159092100005306709809425040261  # 75092699954400878964964014863999053929  # int
    sorts_from_each_path_to_load = [
        ## simulated20221116:
        # {
        # "20231011_185107"  # 1 std, 4 jitter
        # "20231011_195053"  # 2 std, 4 jitter
        # "20231011_201450"  # 4 std, 4 jitter
        # "20231011_202716"  # 8 std, 4 jitter
        # } All in braces did not have channel delays reintroduced for continuous.dat
        ## simulated20221117:
        # {
        # "20231027_183121"  # 1 std, 4 jitter, all MUsort options ON
        # "20231031_141254"  # 1 std, 4 jitter, all MUsort options ON, slightly better
        # "20231103_160031096827"  # 1 std, 4 jitter, all MUsort options ON, ?
        # "20231103_175840215876"  # 2 std, 8 jitter, all MUsort options ON, ?
        # "20231103_164647242198"  # 2 std, 4 jitter, all MUsort options ON, custom_merge
        # "20231105_192242190872"  # 2 std, 8 jitter, all MUsort options ON, except multi-threshold
        # "20231101_165306036638"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[1,0.5], spkTh=[-6]
        # "20231101_164409249821"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[1,0.5], spkTh=[-2]
        # "20231101_164937098773"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[5,2], spkTh=[-6]
        # "20231101_164129797219"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[5,2], spkTh=[-2]
        # "20231101_165135058289"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[2,1], spkTh=[-6]
        # "20231102_175449741223"  # 1 std, 4 jitter, vanilla Kilosort, Th=[1,0.5], spkTh=[-6]
        "20231103_184523634126"  # 2 std, 8 jitter, vanilla Kilosort, Th=[1,0.5], spkTh=[-6]
        # "20231103_184518491799"  # 2 std, 8 jitter, vanilla Kilosort, Th=[2,1], spkTh=[-6]
        # } All in braces did not have channel delays reintroduced for continuous.dat
    ]
    clusters_to_take_from = {
        # {
        "20231027_183121": [24, 2, 3, 1, 23, 26, 0, 4, 32, 27],
        "20231031_141254": [26, 4, 3, 1, 24, 28, 0, 2, 34, 29],
        "20231103_160031096827": [21, 4, 3, 1, 14, 17, 0, 2, 20, 19],
        "20231103_175840215876": [13, 5, 2, 1, 11, 14, 0, 4, 20, 17],
        # [12, 4, 3, 1, 8, 10, 0, 2, 11, 9], # <- custom_merge
        "20231103_164647242198": [20, 4, 3, 1, 15, 17, 0, 2, 22, 21],
        "20231101_165306036638": [35, 13, 11, 1, 29, 39, 0, 10, 37, 36],
        "20231101_164409249821": [26, 9, 7, 11, 23, 29, 0, 8, 30, 25],
        "20231101_164937098773": [22, 11, 5, 2, 21, 28, 1, 7, 25, 27],
        "20231101_164129797219": [25, 9, 7, 1, 24, 27, 0, 8, 29, 28],
        "20231101_165135058289": [21, 9, 3, 2, 19, 23, 0, 4, 28, 24],
        "20231102_175449741223": [9, 48, 23, 24, 3, 4, 22, 0, 10, 13],
        "20231103_184523634126": [17, 31, 14, 16, 18, 5, 13, 19, 6, 10],
        "20231103_184518491799": [5, 35, 12, 15, 3, 6, 13, 0, 8, 28],
        # ^ 28 is filler unit because it was not found          ^^^^
        "20231105_192242190872": [26, 5, 2, 1, 18, 19, 0, 3, 22, 21],
        # } All in braces did not have channel delays reintroduced for continuous.dat
    }
    # [ # godzilla 11-16-2022
    # [8, 5, 7, 1, 3, 2, 0, 6]  # 1 std, 4 jitter
    # [7, 4, 5, 3, 2, 1, 0, 6]  # 2 std, 4 jitter
    # [8, 5, 7, 1, 3, 4, 0, 6]  # 4 std, 4 jitter
    # [9, 3, 7, 1, 2, 8, 0, 6]  # 8 std, 4 jitter
    # ]  # [[18, 2, 11, 0, 4, 10, 1, 9]]  # list of lists
    num_motor_units = len(clusters_to_take_from[sorts_from_each_path_to_load[0]])
    plot_template = "plotly_white"
    plot2_xlim = [0, 0.1]
    show_plot1 = False
    show_plot2 = False
    show_plot3 = False
    save_png_plot1 = False
    save_png_plot2 = False
    save_png_plot3 = False
    save_svg_plot1 = True
    save_svg_plot2 = True
    save_svg_plot3 = True
    save_html_plot1 = False
    save_html_plot2 = False
    save_html_plot3 = False
    ## load ground truth data
    ground_truth_path = Path(
        "spikes_20221117_godzilla_SNR-1-from_data_jitter-4std_files-11.npy"
        # "spikes_20221117_godzilla_SNR-1-from_data_jitter-1std_files-5.npy"
        # "spikes_20221116_godzilla_SNR-8-from_data_jitter-4std_files-1.npy"
    )  # spikes_20221116_godzilla_SNR-None_jitter-0std_files-1.npy
    ## load Kilosort data
    # paths to the folders containing the Kilosort data
    paths_to_KS_session_folders = [
        Path(
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221116/"
            "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221117/"
        ),
    ]
    # find the folder name which ends in _myo and append to the paths_to_session_folders
    paths_to_each_myo_folder = []
    for iDir in paths_to_KS_session_folders:
        myo = [f for f in iDir.iterdir() if (f.is_dir() and f.name.endswith("_myo"))]
        assert (
            len(myo) == 1
        ), "There should only be one _myo folder in each session folder"
        paths_to_each_myo_folder.append(myo[0])
    # inside each _myo folder, find the folder name which constains sort_from_each_path_to_load string
    list_of_paths_to_sorted_folders = []
    for iPath in paths_to_each_myo_folder:
        matches = [
            f
            for f in iPath.iterdir()
            if f.is_dir() and any(s in f.name for s in sorts_from_each_path_to_load)
        ]
        assert (
            len(matches) == 1
        ), "There should only be one sort folder match in each _myo folder"
        if use_custom_merge_clusters:
            # append the path to the custom_merge_clusters folder
            list_of_paths_to_sorted_folders.append(
                matches[0].joinpath("custom_merges/final_merge")
            )
        list_of_paths_to_sorted_folders.append(matches[0])

    if parallel:
        import multiprocessing as mp

        with mp.Pool(processes=len(bin_widths_for_comparison)) as pool:
            zip_obj = zip(
                [ground_truth_path] * len(bin_widths_for_comparison),
                [random_seed_entropy] * len(bin_widths_for_comparison),
                bin_widths_for_comparison,
                [ephys_fs] * len(bin_widths_for_comparison),
                [time_frame] * len(bin_widths_for_comparison),
                [list_of_paths_to_sorted_folders] * len(bin_widths_for_comparison),
                [clusters_to_take_from[sorts_from_each_path_to_load[0]]]
                * len(bin_widths_for_comparison),
            )
            results = pool.starmap(
                compare_spike_trains,
                zip_obj,
            )

            # extract results
            kilosort_spikes = [result[0] for result in results]
            ground_truth_spikes = [result[1] for result in results]
            accuracies = [result[2] for result in results]
            precisions = [result[3] for result in results]
            recalls = [result[4] for result in results]
            num_matches = [result[5] for result in results]
            num_kilosort_spikes = [result[6] for result in results]
            num_ground_truth_spikes = [result[7] for result in results]
            true_positive_spikes = [result[8] for result in results]
            false_positive_spikes = [result[9] for result in results]
            false_negative_spikes = [result[10] for result in results]

            # collapse accuracies, precisions, and recalls into a 2D array
            accuracies = np.vstack(accuracies)
            precisions = np.vstack(precisions)
            recalls = np.vstack(recalls)
    else:
        # compare spike trains for each bin width
        kilosort_spikes = []
        ground_truth_spikes = []
        accuracies = []
        precisions = []
        recalls = []
        num_matches = []
        num_kilosort_spikes = []
        num_ground_truth_spikes = []
        true_positive_spikes = []
        false_positive_spikes = []
        false_negative_spikes = []
        for iBinWidth in range(len(bin_widths_for_comparison)):
            (
                kilosort_spikes_temp,
                ground_truth_spikes_temp,
                accuracy_temp,
                precision_temp,
                recall_temp,
                num_matches_temp,
                num_kilosort_spikes_temp,
                num_ground_truth_spikes_temp,
                true_positive_spikes_temp,
                false_positive_spikes_temp,
                false_negative_spikes_temp,
            ) = compare_spike_trains(
                ground_truth_path,
                random_seed_entropy,
                bin_widths_for_comparison[iBinWidth],
                ephys_fs,
                time_frame,
                list_of_paths_to_sorted_folders,
                clusters_to_take_from[sorts_from_each_path_to_load[0]],
            )

            # convert all to numpy arrays before appending
            kilosort_spikes.append(np.array(kilosort_spikes_temp))
            ground_truth_spikes.append(np.array(ground_truth_spikes_temp))
            accuracies.append(np.array(accuracy_temp))
            precisions.append(np.array(precision_temp))
            recalls.append(np.array(recall_temp))
            num_matches.append(np.array(num_matches_temp))
            num_kilosort_spikes.append(np.array(num_kilosort_spikes_temp))
            num_ground_truth_spikes.append(np.array(num_ground_truth_spikes_temp))
            true_positive_spikes.append(np.array(true_positive_spikes_temp))
            false_positive_spikes.append(np.array(false_positive_spikes_temp))
            false_negative_spikes.append(np.array(false_negative_spikes_temp))

        # collapse accuracies, precisions, and recalls into a 2D array
        accuracies = np.vstack(accuracies)
        precisions = np.vstack(precisions)
        recalls = np.vstack(recalls)

    if show_plot1 or save_png_plot1 or save_html_plot1:
        ### plot 1: bar plot of spike counts
        # now create an overlay plot of the two plots above. Do not use subplots, but use two y axes
        # make bar plot of total spike counts use left y axis
        plot1(
            num_ground_truth_spikes[iShow],
            num_kilosort_spikes[iShow],
            precisions[iShow],
            recalls[iShow],
            accuracies[iShow],
            bin_widths_for_comparison[iShow],
            clusters_to_take_from[sorts_from_each_path_to_load[0]],
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot1,
            save_png_plot1,
            save_html_plot1,
            # make figsize 1080p
            figsize=(1920, 1080),
        )

    # print(
    #     f"Total number of Kilosort spikes: {np.sum(kilosort_spikes[iShow])} ({np.sum(kilosort_spikes[iShow])/np.sum(ground_truth_spikes[iShow])*100:.2f}%)"
    # )
    # print(
    #     f"\tTotal number of true positive spikes: {np.sum(true_positive_spikes[iShow])} ({np.sum(true_positive_spikes[iShow])/np.sum(ground_truth_spikes[iShow])*100:.2f}%)"
    # )
    # print(
    #     f"\tTotal number of false positive spikes: {np.sum(false_positive_spikes[iShow])} ({np.sum(false_positive_spikes[iShow])/np.sum(kilosort_spikes[iShow])*100:.2f}%)"
    # )
    # print(
    #     f"\tTotal number of false negative spikes: {np.sum(false_negative_spikes[iShow])} ({np.sum(false_negative_spikes[iShow])/np.sum(ground_truth_spikes[iShow])*100:.2f}%)"
    # )
    # for iUnit in range(num_motor_units):
    #     print(
    #         f"Total number of Kilosort spikes in unit {clusters_to_take_from[sorts_from_each_path_to_load[0]][iUnit]}: {np.sum(kilosort_spikes[iShow][:,iUnit])} ({np.sum(kilosort_spikes[iShow][:,iUnit])/np.sum(ground_truth_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    #     print(
    #         f"\tTotal number of true positive spikes in unit {clusters_to_take_from[sorts_from_each_path_to_load[0]][iUnit]}: {np.sum(true_positive_spikes[iShow][:,iUnit])} ({np.sum(true_positive_spikes[iShow][:,iUnit])/np.sum(ground_truth_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    #     print(
    #         f"\tTotal number of false positive spikes in unit {clusters_to_take_from[sorts_from_each_path_to_load[0]][iUnit]}: {np.sum(false_positive_spikes[iShow][:,iUnit])} ({np.sum(false_positive_spikes[iShow][:,iUnit])/np.sum(kilosort_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    #     print(
    #         f"\tTotal number of false negative spikes in unit {clusters_to_take_from[sorts_from_each_path_to_load[0]][iUnit]}: {np.sum(false_negative_spikes[iShow][:,iUnit])} ({np.sum(false_negative_spikes[iShow][:,iUnit])/np.sum(ground_truth_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    # print metrics for each unit
    print("\n")  # add a newline for readability
    print("Sort: ", sorts_from_each_path_to_load[0])
    unit_df = df()
    unit_df["Unit"] = np.array(
        clusters_to_take_from[sorts_from_each_path_to_load[0]]
    ).astype(int)
    unit_df["True Count"] = num_ground_truth_spikes[iShow]
    unit_df["KS Count"] = num_kilosort_spikes[iShow]
    unit_df["Precision"] = precisions[iShow]
    unit_df["Recall"] = recalls[iShow]
    unit_df["Accuracy"] = accuracies[iShow]
    unit_df["Unit"].astype(int)
    unit_df.set_index("Unit", inplace=True)
    print(unit_df)
    print("\n")  # add a newline for readability
    # print lowest and highest accuracies, limit to 3 decimal places
    print(
        f"Lowest accuracy: {unit_df['Accuracy'].min():.3f}, Unit {unit_df['Accuracy'].idxmin()}"
    )
    print(
        f"Highest accuracy: {unit_df['Accuracy'].max():.3f}, Unit {unit_df['Accuracy'].idxmax()}"
    )

    # print average accuracy plus or minus standard deviation
    print(
        f"Average accuracy: {unit_df['Accuracy'].mean():.3f} +/- {unit_df['Accuracy'].std():.3f}"
    )
    # print average accuracy weighted by number of spikes in each unit
    print(
        f"Weighted average accuracy: {np.average(unit_df['Accuracy'], weights=unit_df['True Count']):.3f}"
    )
    # print average precision plus or minus standard deviation
    print(
        f"Average precision: {unit_df['Precision'].mean():.3f} +/- {unit_df['Precision'].std():.3f}"
    )
    # print average recall plus or minus standard deviation
    print(
        f"Average recall: {unit_df['Recall'].mean():.3f} +/- {unit_df['Recall'].std():.3f}"
    )

    if show_plot2 or save_png_plot2 or save_html_plot2:
        ### plot 2: spike trains

        # now plot the spike trains, emphasizing each type of error with a different color
        # include an event plot of the kilosort and ground truth spikes for reference
        plot2(
            kilosort_spikes[iShow],
            ground_truth_spikes[iShow],
            false_positive_spikes[iShow],
            false_negative_spikes[iShow],
            true_positive_spikes[iShow],
            bin_widths_for_comparison[iShow],
            clusters_to_take_from[sorts_from_each_path_to_load[0]],
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot2,
            save_png_plot2,
            save_html_plot2,
            # make figsize 1080p
            figsize=(1920, 1080),
        )

    if show_plot3 or save_png_plot3 or save_html_plot3:
        ### plot 3: comparison across bin widths, ground truth metrics
        # the bin width for comparison is varied from 0.25 ms to 8 ms, doubling each time
        # the metrics are computed for each bin width
        plot3(
            bin_widths_for_comparison,
            precisions,
            recalls,
            accuracies,
            num_motor_units,
            clusters_to_take_from[sorts_from_each_path_to_load[0]],
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot3,
            save_png_plot3,
            save_html_plot3,
            # make figsize 1080p
            figsize=(1920, 1080),
        )

    finish_time = datetime.now()
    time_elapsed = finish_time - start_time
    # use strfdelta to format time elapsed prettily
    print(
        (
            "Time elapsed: "
            f"{strfdelta(time_elapsed, '{hours} hours, {minutes} minutes, {seconds} seconds')}"
        )
    )
