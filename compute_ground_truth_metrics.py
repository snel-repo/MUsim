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

    min_delay_ms = -1  # ms
    max_delay_ms = 1  # ms
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
    plot2_xlim,
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

    # cut all arrays short by the factor of plot2_xlim
    left_bound = int(round(plot2_xlim[0] * len(kilosort_spikes)))
    right_bound = int(round(plot2_xlim[1] * len(kilosort_spikes)))
    kilosort_spikes = kilosort_spikes[left_bound:right_bound, :]
    ground_truth_spikes = ground_truth_spikes[left_bound:right_bound, :]
    false_positive_spikes = false_positive_spikes[left_bound:right_bound, :]
    false_negative_spikes = false_negative_spikes[left_bound:right_bound, :]
    true_positive_spikes = true_positive_spikes[left_bound:right_bound, :]

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
    automatically_assign_cluster_mapping = True
    time_frame = [0, 1]  # must be between 0 and 1
    ephys_fs = 30000  # Hz
    # range from 0.125 ms to 8 ms in log2 increments
    xstart = np.log2(0.125)
    bin_widths_for_comparison = np.logspace(xstart, -xstart, num=13, base=2)
    # index of which bin width of bin_widths_for_comparison to show in plots
    iShow = 6

    nt0 = 121  # number of time bins in the template, in ms it is 3.367
    random_seed_entropy = 218530072159092100005306709809425040261  # 75092699954400878964964014863999053929  # int
    plot_template = "plotly_white"
    plot2_xlim = [0, 0.1]
    show_plot1 = False
    show_plot2 = False
    show_plot3 = False
    save_png_plot1 = True
    save_png_plot2 = True
    save_png_plot3 = False
    save_svg_plot1 = False
    save_svg_plot2 = False
    save_svg_plot3 = False
    save_html_plot1 = False
    save_html_plot2 = True
    save_html_plot3 = False

    ## paths with simulated data
    path_to_sim_dat = Path(
        "continuous_20221117_godzilla_SNR-400-constant_jitter-0std_files-11.dat"
    )  # continuous_20221117_godzilla_SNR-None-constant_jitter-0std_files-11.dat

    ## load ground truth data
    ground_truth_path = Path(
        "spikes_20221117_godzilla_SNR-400-constant_jitter-0std_files-11.npy"
        # "spikes_20221117_godzilla_SNR-1-from_data_jitter-4std_files-11.npy"
        # "spikes_20221117_godzilla_SNR-1-from_data_jitter-1std_files-5.npy"
        # "spikes_20221116_godzilla_SNR-8-from_data_jitter-4std_files-1.npy"
    )  # spikes_20221116_godzilla_SNR-None_jitter-0std_files-1.npy
    ## load Kilosort data
    # paths to the folders containing the Kilosort data
    paths_to_KS_session_folders = [
        Path(
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221116/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221117/"
            "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/triple/simulated20231219/"
        ),
    ]
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
        # "20231105_192242190872"  # 2 std, 8 jitter, all MUsort options ON, except multi-threshold $$$ BEST EMUsort $$$
        # "20231101_165306036638"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[1,0.5], spkTh=[-6]
        # "20231101_164409249821"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[1,0.5], spkTh=[-2]
        # "20231101_164937098773"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[5,2], spkTh=[-6]
        # "20231101_164129797219"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[5,2], spkTh=[-2]
        # "20231101_165135058289"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[2,1], spkTh=[-6]
        # "20231102_175449741223"  # 1 std, 4 jitter, vanilla Kilosort, Th=[1,0.5], spkTh=[-6]
        # "20231103_184523634126"  # 2 std, 8 jitter, vanilla Kilosort, Th=[1,0.5], spkTh=[-6] $$$ BEST Kilosort3 $$$
        # "20231103_184518491799"  # 2 std, 8 jitter, vanilla Kilosort, Th=[2,1], spkTh=[-6]
        # } All in braces did not have channel delays reintroduced for continuous.dat
        #### Below are with new 16 channel, triple rat dataset.
        # simulated20231219:
        "20231220_180513756759"  # SNR-400-constant_jitter-0std_files-11, vanilla Kilosort, Th=[10,4], spkTh=[-6]
        # "20231220_172352030313"  # SNR-400-constant_jitter-0std_files-11, EMUsort, Th=[5,2], spkTh=[-3,-6]
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
        #### Below are with new 16 channel, triple rat dataset.
        "20231220_180513756759": [
            39,
            99,
            103,
            89,
            0,
            30,
            2,
            29,
            110,
            79,
            6,
            69,
            119,
            112,
            3,
            88,
            130,
            56,
            28,
            129,
            117,
            58,
            13,
            96,
            15,
        ],  # 39 and also below 129 are filler units because they were not found
        "20231220_172352030313": [
            5,
            53,
            82,
            83,
            7,
            51,
            21,
            2,
            24,
            55,
            54,
            20,
            22,
            40,
            52,
            56,
            59,
            71,
            58,
            48,
            66,
            37,
            44,
            19,
            33,
        ],  # 33 is filler unit because it was not found
    }
    # [ # godzilla 11-16-2022
    # [8, 5, 7, 1, 3, 2, 0, 6]  # 1 std, 4 jitter
    # [7, 4, 5, 3, 2, 1, 0, 6]  # 2 std, 4 jitter
    # [8, 5, 7, 1, 3, 4, 0, 6]  # 4 std, 4 jitter
    # [9, 3, 7, 1, 2, 8, 0, 6]  # 8 std, 4 jitter
    # ]  # [[18, 2, 11, 0, 4, 10, 1, 9]]  # list of lists

    clusters_in_sort_to_use = clusters_to_take_from[sorts_from_each_path_to_load[0]]
    num_motor_units = len(clusters_in_sort_to_use)
    true_spike_counts_for_each_cluster = np.load(str(ground_truth_path)).sum(axis=0)

    # find the folder name which ends in _myo and append to the paths_to_session_folders
    paths_to_each_myo_folder = []
    for iDir in paths_to_KS_session_folders:
        myo = [f for f in iDir.iterdir() if (f.is_dir() and f.name.endswith("_myo"))]
        assert (
            len(myo) == 1
        ), f"There should only be one _myo folder in each session folder, but there were {len(myo)} in {iDir}"
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
        ), f"There should only be one sort folder match in each _myo folder, but there were {len(matches)} in {iPath}"
        if use_custom_merge_clusters:
            # append the path to the custom_merge_clusters folder
            list_of_paths_to_sorted_folders.append(
                matches[0].joinpath("custom_merges/final_merge")
            )
        list_of_paths_to_sorted_folders.append(matches[0])

    if automatically_assign_cluster_mapping:
        # automatically assign cluster mapping by extracting the waves at the spike times for all
        # clusters, getting the median waveform for each cluster using both groundtruth and the sort
        # by using the respective spike times, then computing the correlation between each cluster's
        # median wave and the median waves of the ground truth clusters, pairing the clusters with
        # the highest correlation match, and then using those pairs to reorder the clusters
        # in 'clusters_to_take_from' to match the ground truth clusters
        # also need to be sure to check all lags between the ground truth and the sort median waves
        # to make sure that the correlation is not being computed between two waves that are
        # misaligned in time, which would result in an errantly and artificially low correlation
        spike_times_list = [
            np.load(str(path_to_sorted_folder.joinpath("spike_times.npy"))).flatten()
            for path_to_sorted_folder in list_of_paths_to_sorted_folders
        ]

        spike_clusters_list = [
            np.load(str(path_to_sorted_folder.joinpath("spike_clusters.npy"))).flatten()
            for path_to_sorted_folder in list_of_paths_to_sorted_folders
        ]

        # clusters_in_sort_to_use = np.unique(
        #     spike_clusters_list[0]
        # )  # take all clusters in the first sort

        # get the spike times for each cluster
        spike_times = spike_times_list[0]
        spike_clusters = spike_clusters_list[0]

        # drop any clusters with <300 spikes
        # clusters_in_sort_to_use = clusters_in_sort_to_use[
        #     np.array(
        #         [
        #             np.sum(spike_clusters == iCluster) >= 300
        #             for iCluster in clusters_in_sort_to_use
        #         ]
        #     ).astype(int)
        # ]

        # get the spike times for each cluster
        spike_times_for_each_cluster = [
            spike_times[spike_clusters == iCluster]
            for iCluster in clusters_in_sort_to_use
        ]

        # load and reshape into numchans x whatever (2d array) the data.bin file
        sim_ephys_data = np.memmap(
            str(path_to_sim_dat), dtype="int16", mode="r"
        ).reshape(
            -1, 24
        )  ### WARNING HARDCODED 24 CHANNELS ### !!!

        # only take the first 16 channels, last 8 are dummy channels
        sim_ephys_data = sim_ephys_data[:, :16]

        # get the spike snippets for each cluster
        spike_snippets_for_each_cluster = [
            np.array(
                [
                    sim_ephys_data[
                        int(iSpike_time - nt0 // 2) : int(iSpike_time + nt0 // 2 + 1),
                        :,
                    ]
                    for iSpike_time in iCluster_spike_times
                ]
            )
            for iCluster_spike_times in spike_times_for_each_cluster
        ]  # dimensions are (num_spikes, nt0, num_chans_in_recording)
        # get the median waveform shape for each cluster, but standardize the waveforms
        # before computing the median
        standardized_spike_snippets_for_each_cluster = [
            (iCluster_snippets - np.mean(iCluster_snippets)) / np.std(iCluster_snippets)
            for iCluster_snippets in spike_snippets_for_each_cluster
        ]
        median_spike_snippets_for_each_cluster = [
            np.median(iCluster_snippets, axis=0)
            for iCluster_snippets in standardized_spike_snippets_for_each_cluster
        ]

        # now do the same for the ground truth spikes. Load the ground truth spike times
        # which are 1's and 0's, where 1's indicate a spike and 0's indicate no spike
        # each column is a different unit, and row is a different time point in the recording
        # now extract the waves at the spike times for all clusters, get the GT median waveform
        # get the spike times for each cluster with np.where, but make
        ground_truth_spike_times = np.load(str(ground_truth_path))
        GT_spike_times_for_each_cluster = [
            np.where(ground_truth_spike_times[:, iCluster] == 1)[0]
            for iCluster in range(ground_truth_spike_times.shape[1])
        ]
        # get the spike snippets for each cluster in the ground truth
        spike_snippets_for_each_cluster_ground_truth = []
        median_spike_snippets_for_each_cluster_ground_truth = []
        for iCluster in range(len(GT_spike_times_for_each_cluster)):
            spike_snippets_for_each_cluster_ground_truth.append([])
            # get the spike snippets for each cluster
            for iSpike_time in GT_spike_times_for_each_cluster[iCluster]:
                if iSpike_time - nt0 // 2 >= 0 and iSpike_time + nt0 // 2 + 1 <= len(
                    sim_ephys_data
                ):
                    spike_snippets_for_each_cluster_ground_truth[iCluster].append(
                        sim_ephys_data[
                            int(iSpike_time - nt0 // 2) : int(
                                iSpike_time + nt0 // 2 + 1
                            ),
                            :,
                        ]
                    )
            spike_snippets_for_each_cluster_ground_truth[iCluster] = np.array(
                spike_snippets_for_each_cluster_ground_truth[iCluster]
            )

            # get the median waveform shape for each cluster, but standardize the waveforms
            # before computing the median
            standardized_spike_snippets_for_each_cluster_ground_truth = (
                spike_snippets_for_each_cluster_ground_truth[iCluster]
                - np.mean(spike_snippets_for_each_cluster_ground_truth[iCluster])
            ) / np.std(spike_snippets_for_each_cluster_ground_truth[iCluster])

            median_spike_snippets_for_each_cluster_ground_truth.append(
                np.median(
                    standardized_spike_snippets_for_each_cluster_ground_truth, axis=0
                )
            )
        # now compute the correlation between each cluster's median wave
        # and the median waves of the ground truth clusters, pairing the clusters with the highest
        # correlation match and then using those pairs to reorder the clusters in
        # 'clusters_to_take_from' to match the ground truth clusters
        # once a high correlation is found, remove that cluster from the future correlation calculations
        # so that it cannot be paired with another cluster
        # need to compute correlation for all lags and take highest correlation to ensure success
        # even if the ground truth and the sort are misaligned in time
        # use scipy.signal.correlate
        GT_clusters_iter = list(
            range(len(median_spike_snippets_for_each_cluster_ground_truth))
        )
        cluster_mapping = dict()
        new_cluster_ordering = np.nan * np.ones(
            len(median_spike_snippets_for_each_cluster_ground_truth)
        )
        sort_clust_IDs_weighted_corr_score_dict = dict()
        for iCluster in range(len(median_spike_snippets_for_each_cluster)):
            if len(GT_clusters_iter) == 0:
                break
            # initialize a list of correlations for each cluster
            correlations = []
            for iCluster_ground_truth in GT_clusters_iter:
                # initialize a list of correlations for each lag
                correlations_for_each_lag = []
                for iLag in range(-nt0 // 2, nt0 // 2 + 1):
                    # compute the correlation between the two median waves
                    correlations_for_each_lag.append(
                        np.corrcoef(
                            np.roll(
                                median_spike_snippets_for_each_cluster[
                                    iCluster
                                ].T.flatten(),
                                iLag,
                                axis=0,
                            ),
                            median_spike_snippets_for_each_cluster_ground_truth[
                                iCluster_ground_truth
                            ].T.flatten(),
                        )[0, 1]
                    )
                # append the highest correlation for this cluster
                correlations.append(np.max(correlations_for_each_lag))
                # scale correlations by the accuracy of the spike count compared to the true spike count
                # this is to prevent a cluster with a very low spike count from being matched to a cluster
                # with a very high spike count, which might have a high correlation but is not a good match
                # because the spike counts are so different
                # equation is: exp( abs( true spike count - sort spike count ) / (2 * true spike count^2) )
                # which will be 1 if the spike counts are the same, and will be 0 if the spike counts are
                # very different, with a sigma equal to the true spike count to prevent a cluster with a
                # dramatically different spike count from being matched
                spike_count_match_score = np.exp(
                    -(
                        (
                            np.abs(
                                true_spike_counts_for_each_cluster[
                                    iCluster_ground_truth
                                ]
                                - spike_times_for_each_cluster[iCluster].shape[0]
                            )
                        )
                        ** 2
                    )
                    / (
                        2
                        * (true_spike_counts_for_each_cluster[iCluster_ground_truth])
                        ** 2
                    )
                )
                # # if len(GT_clusters_iter) == len(
                # #     median_spike_snippets_for_each_cluster_ground_truth
                # # ):  # print only for the first sort cluster
                # #     print(
                # #         f"With a true count of {true_spike_counts_for_each_cluster[iCluster_ground_truth]} and a sort count of {spike_times_for_each_cluster[iCluster].shape[0]}, the spike count match score is {spike_count_match_score}"
                # #     )
                # #     print(
                # #         f"Correlations for cluster {clusters_in_sort_to_use[iCluster]} are {correlations}"
                # #     )

                # now scale the correlation by the spike count match index
                correlations[-1] = correlations[-1] * spike_count_match_score

            # now find the cluster with the highest correlation
            GT_cluster_match_idx = np.argmax(correlations)
            # track the highest correlation for each sort cluster in a dictionary
            sort_clust_IDs_weighted_corr_score_dict[
                clusters_in_sort_to_use[iCluster]
            ] = correlations[GT_cluster_match_idx]
            # now assign the cluster mapping, where the key is the sort cluster
            # and the value is the matching ground truth cluster
            cluster_mapping[clusters_in_sort_to_use[iCluster]] = GT_clusters_iter[
                GT_cluster_match_idx
            ]
            # now remove the ground truth cluster from the list of clusters to match
            GT_clusters_iter.pop(GT_cluster_match_idx)
        # now loop through the clusters in 'clusters_to_take_from' and use as key to cluster_mapping
        # to get the ground truth cluster that it maps to, and then place the sort cluster ID into
        # the new_cluster_ordering array at the index of the ground truth cluster ID
        for iCluster in range(len(clusters_in_sort_to_use)):
            try:
                new_cluster_ordering[
                    cluster_mapping[clusters_in_sort_to_use[iCluster]]
                ] = clusters_in_sort_to_use[iCluster]
            except KeyError:
                continue

        # now replace the clusters_to_take_from with the new_cluster_ordering
        clusters_in_sort_to_use = new_cluster_ordering.astype(int)
        num_motor_units = len(clusters_in_sort_to_use)

    # set_trace()
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
                [clusters_in_sort_to_use] * len(bin_widths_for_comparison),
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
                clusters_in_sort_to_use,
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

    if show_plot1 or save_png_plot1 or save_html_plot1 or save_svg_plot1:
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
            clusters_in_sort_to_use,
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot1,
            save_png_plot1,
            save_svg_plot1,
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
    #         f"Total number of Kilosort spikes in unit {clusters_in_sort_to_use[iUnit]}: {np.sum(kilosort_spikes[iShow][:,iUnit])} ({np.sum(kilosort_spikes[iShow][:,iUnit])/np.sum(ground_truth_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    #     print(
    #         f"\tTotal number of true positive spikes in unit {clusters_in_sort_to_use[iUnit]}: {np.sum(true_positive_spikes[iShow][:,iUnit])} ({np.sum(true_positive_spikes[iShow][:,iUnit])/np.sum(ground_truth_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    #     print(
    #         f"\tTotal number of false positive spikes in unit {clusters_in_sort_to_use[iUnit]}: {np.sum(false_positive_spikes[iShow][:,iUnit])} ({np.sum(false_positive_spikes[iShow][:,iUnit])/np.sum(kilosort_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    #     print(
    #         f"\tTotal number of false negative spikes in unit {clusters_in_sort_to_use[iUnit]}: {np.sum(false_negative_spikes[iShow][:,iUnit])} ({np.sum(false_negative_spikes[iShow][:,iUnit])/np.sum(ground_truth_spikes[iShow][:,iUnit])*100:.2f}%)"
    #     )
    # print metrics for each unit
    print("\n")  # add a newline for readability
    print("Sort: ", sorts_from_each_path_to_load[0])
    unit_df = df()
    unit_df["Unit"] = np.array(clusters_in_sort_to_use).astype(int)
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

    if show_plot2 or save_png_plot2 or save_html_plot2 or save_svg_plot2:
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
            clusters_in_sort_to_use,
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot2,
            plot2_xlim,
            save_png_plot2,
            save_svg_plot2,
            save_html_plot2,
            # make figsize 1080p
            figsize=(1920, 1080),
        )

    if show_plot3 or save_png_plot3 or save_html_plot3 or save_svg_plot3:
        ### plot 3: comparison across bin widths, ground truth metrics
        # the bin width for comparison is varied from 0.25 ms to 8 ms, doubling each time
        # the metrics are computed for each bin width
        plot3(
            bin_widths_for_comparison,
            precisions,
            recalls,
            accuracies,
            num_motor_units,
            clusters_in_sort_to_use,
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot3,
            save_png_plot3,
            save_svg_plot3,
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
