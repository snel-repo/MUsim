# IMPORT packages
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
    GT_clusters_to_use,
    random_seed_entropy,
    bin_width_for_comparison,
    ephys_fs,
    time_frame,
    list_of_paths_to_sorted_folders,
    clusters_in_sort_to_use,
    spike_isolation_radius_ms=None,
):
    def remove_isolated_spikes(MUsim_obj, radius):
        """Function removes isolated spikes from a MUsim object (from the most recent
           MUsim_obj.spikes entry)

        Args:
            MUsim_obj (MUsim): Object created from MUsim class, containing at least one MUsim_obj.spikes entry
            radius (int, float): Numeric value specifying the number of points to check on either side of the
                                 spike time

        Returns:
            MUsim: MUsim class object without isolated spikes
        """
        isolated_spike_times = np.asarray(
            MUsim_obj.spikes[-1].sum(axis=1) == 1
        ).nonzero()[0]
        for iTime in isolated_spike_times:
            try:
                # if a spike is isolated within the radius, then delete that spike from consideration
                # make sure to account for when the radius goes past the beginning and end of array
                if (
                    MUsim_obj.spikes[-1][
                        iTime - radius : iTime + radius,
                        :,
                    ].sum()
                    == 1
                ):  # if the sum of this slice is 1, there's an isolated spike here
                    MUsim_obj.spikes[-1][
                        iTime - radius : iTime + radius,
                        :,
                    ] = 0
                    # print(f"1st block: zeroed at {iTime}")
            except IndexError:
                if iTime - radius < 0:
                    if (
                        MUsim_obj.spikes[-1][0 : iTime + radius, :].sum() == 1
                    ):  # if the sum of this slice is 1, there's an isolated spike here
                        MUsim_obj.spikes[-1][0 : iTime + radius, :] = 0
                        print(
                            f"Handled IndexError encountered at beginning: zeroed isolated spikes at {iTime}"
                        )
                elif iTime + radius > MUsim_obj.spikes[-1].shape[0]:
                    if (
                        MUsim_obj.spikes[-1][iTime - radius : -1, :].sum() == 1
                    ):  # if the sum of this slice is 1, there's an isolated spike here
                        MUsim_obj.spikes[-1][iTime - radius : -1, :] = 0
                        print(
                            f"Handled IndexError encountered at end: zeroed isolated spikes at {iTime}"
                        )
                else:
                    print(
                        "WARNING: Uncaught case in isolated spike removal, stopping for debugging"
                    )
                    set_trace()
        return MUsim_obj

    # use MUsim object to load and rebin ground truth data
    mu_GT = MUsim(random_seed_entropy)
    mu_GT.sample_rate = 1 / ephys_fs
    mu_GT.load_MUs(
        # npy_file_path
        ground_truth_path,
        1 / ephys_fs,
        load_as="trial",
        slice=time_frame,
        load_type="MUsim",
    )
    mu_GT.spikes[-1] = mu_GT.spikes[-1][:, GT_clusters_to_use]

    # use MUsim object to load and rebin Kilosort data
    mu_KS = MUsim(random_seed_entropy)
    mu_KS.sample_rate = 1 / ephys_fs
    mu_KS.load_MUs(
        # npy_file_path
        list_of_paths_to_sorted_folders[0],
        1 / ephys_fs,
        load_as="trial",
        slice=time_frame,
        load_type="kilosort",
    )

    # subselect clusters in mu_KS using clusters_in_sort_to_use, now in that specified order
    mu_KS.spikes[-1] = mu_KS.spikes[-1][:, clusters_in_sort_to_use]

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

    min_delay_ms = -2  # ms
    max_delay_ms = 2  # ms
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
                f"WARNING: Shifted Kilosort spikes for unit {clusters_in_sort_to_use[iUnit]} by {shift} samples"
            )

    if spike_isolation_radius_ms is not None:
        assert type(spike_isolation_radius_ms) in [
            int,
            float,
        ], "Type of spike_isolation_radius_ms must be int or float"
        assert spike_isolation_radius_ms > 0, "spike_isolation_radius_ms must be >0"
        # delete any spikes from mu_KS and mu_GT which are not within spike_isolation_radius_ms of
        # spikes from neighboring MUs
        spike_isolation_radius_pts = int(spike_isolation_radius_ms / 1000 * ephys_fs)
        print(
            f"removing isolated spikes with a radius of {spike_isolation_radius_pts} points"
        )
        mu_GT = remove_isolated_spikes(mu_GT, spike_isolation_radius_pts)
        mu_KS = remove_isolated_spikes(mu_KS, spike_isolation_radius_pts)

    # rebin the spike trains to the bin width for comparison
    mu_GT.rebin_trials(
        bin_width_for_comparison / 1000
    )  # rebin to bin_width_for_comparison ms bins

    mu_KS.rebin_trials(
        bin_width_for_comparison / 1000
    )  # rebin to bin_width_for_comparison ms bins

    mu_GT.save_spikes("./GT_spikes.npy")
    mu_KS.save_spikes("./KS_spikes.npy")

    ground_truth_spikes = mu_GT.spikes[-1]  # shape is (num_bins, num_units)
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
    clusters_in_sort_to_use,
    GT_clusters_to_use,
    sort_from_each_path_to_load,
    plot_template,
    plot1_bar_type,
    plot1_ylim,
    show_plot1a,
    save_png_plot1a,
    save_svg_plot1a,
    save_html_plot1a,
    show_plot1b,
    save_png_plot1b,
    save_svg_plot1b,
    save_html_plot1b,
    figsize=(1920, 1080),
):
    # get suffix after the KS folder name, which is the repo branch name for that sort
    PPP_branch_name = list_of_paths_to_sorted_folders[0].name.split("_")[-1]
    sort_type = "Kilosort" if PPP_branch_name == "KS" else "EMUsort"

    if show_plot1a or save_png_plot1a or save_svg_plot1a or save_html_plot1a:
        fig1a = go.Figure()
        fig1a.add_trace(
            go.Scatter(
                x=np.arange(0, num_motor_units),
                y=precision,
                mode="lines+markers",
                name="Precision",
                line=dict(width=4, color="green"),
                # yaxis="y2",
            )
        )
        fig1a.add_trace(
            go.Scatter(
                x=np.arange(0, num_motor_units),
                y=recall,
                mode="lines+markers",
                name="Recall",
                line=dict(width=4, color="crimson"),
                # yaxis="y2",
            )
        )
        fig1a.add_trace(
            go.Scatter(
                x=np.arange(0, num_motor_units),
                y=accuracy,
                mode="lines+markers",
                name="Accuracy",
                line=dict(width=4, color="orange"),
                # yaxis="y2",
            )
        )

        # make the title shifted higher up,
        # make text much larger
        fig1a.update_layout(
            title={
                "text": f"<b>Comparison of {sort_type} Performance to Ground Truth, {bin_width_for_comparison} ms Bins</b><br><sup>Sort: {sort_from_each_path_to_load}</sup>",
                # "y": 0.95,
            },
            xaxis_title="<b>GT Cluster ID,<br>True Count</b>",
            # legend_title="Ground Truth Metrics",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template=plot_template,
            yaxis=dict(
                title="<b>Metric Score</b>",
                title_standoff=1,
                range=[0, 1],
                # overlaying="y2",
            ),
            # yaxis2=dict(
            #     title=bar_yaxis_title,
            #     title_standoff=1,
            #     # anchor="free",
            #     # autoshift=True,
            #     # shift=-30,
            #     # side="right",
            # ),
        )
        # update the x tick label of the bar graph to match the cluster ID
        fig1a.update_xaxes(
            ticktext=[
                f"Unit {GT_clusters_to_use[iUnit]},<br>{str(round(num_ground_truth_spikes[iUnit]/1000,1))}k"
                for iUnit in range(num_motor_units)
            ],
            tickvals=np.arange(0, num_motor_units),
            # tickfont=dict(size=14, family="Arial"),
        )

    if show_plot1b or save_png_plot1b or save_svg_plot1b or save_html_plot1b:
        # make text larger
        fig1b = go.Figure(
            layout=go.Layout(
                yaxis=dict(
                    # title_font=dict(size=14, family="Arial"),
                    title_standoff=10,
                ),
                # title_font=dict(size=18),
            )
        )

        if plot1_bar_type == "totals":
            fig1b.add_trace(
                go.Bar(
                    x=np.arange(0, num_motor_units),
                    y=num_ground_truth_spikes,
                    name="Ground Truth",
                    marker_color="rgb(55, 83, 109)",
                    opacity=0.5,
                )
            )
            fig1b.add_trace(
                go.Bar(
                    x=np.arange(0, num_motor_units),
                    y=num_kilosort_spikes,
                    name=sort_type,
                    marker_color="rgb(26, 118, 255)",
                    opacity=0.5,
                )
            )
            bar_yaxis_title = "<b>Spike Count</b>"
        elif plot1_bar_type == "percent":
            fig1b.add_trace(
                go.Bar(
                    x=np.arange(0, num_motor_units),
                    y=100 * num_kilosort_spikes / num_ground_truth_spikes,
                    name="% True Spike Count",
                    # showlegend=False,
                    marker_color="cornflowerblue",
                    opacity=1,
                )
            )
        else:
            raise ValueError(
                f"plot1_bar_type must be 'totals' or 'percent', not {plot1_bar_type}"
            )
        bar_yaxis_title = "<b>% True Spike Count</b>"
        fig1b.add_hline(
            y=100,
            line_width=3,
            line_dash="dash",
            line_color="black",
            # yref="y2",
            name="100% Spike Count",
        )
        # make all the text way larger
        fig1b.update_layout(
            title={
                "text": f"<b>True Spike Count Captured for Each Cluster Using {sort_type}, {bin_width_for_comparison} ms Bins</b><br><sup>Sort: {sort_from_each_path_to_load}</sup>",
                # "y": 0.95,
            },
            xaxis_title="<b>GT Cluster ID,<br>True Count</b>",
            # legend_title="Ground Truth Metrics",
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template=plot_template,
            # yaxis=dict(
            #     title="<b>Metric Score</b>",
            #     title_standoff=1,
            #     range=[0, 1],
            #     # overlaying="y2",
            # ),
            yaxis=dict(
                title=bar_yaxis_title,
                # title_standoff=1,
                # anchor="free",
                # autoshift=True,
                # shift=-30,
                # side="right",
            ),
            # make the title text larger
            # title_font=dict(size=18),
        )
        # set_trace()

        # update the x tick label of the bar graph to match the cluster ID
        fig1b.update_xaxes(
            ticktext=[
                f"Unit {GT_clusters_to_use[iUnit]},<br>{str(round(num_ground_truth_spikes[iUnit]/1000,1))}k"
                for iUnit in range(num_motor_units)
            ],
            tickvals=np.arange(0, num_motor_units),
            # tickfont=dict(size=14, family="Arial"),
        )
        fig1b.update_layout(yaxis_range=plot1_ylim)
        # make y axis title smaller
        # fig1b.update_yaxes(title_font=dict(size=14, family="Arial"))

        # move the y axis title closer to the y axis
        fig1b.update_yaxes(title_standoff=10)

        # make subplot titles bigger
        # fig.update_annotations(font=dict(size=18))

    if save_png_plot1a:
        fig1a.write_image(
            f"Fig1a_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot1a:
        fig1a.write_image(
            f"Fig1a_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot1a:
        fig1a.write_html(
            f"Fig1a_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot1a:
        fig1a.show()

    if save_png_plot1b:
        fig1b.write_image(
            f"Fig1b_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot1b:
        fig1b.write_image(
            f"Fig1b_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot1b:
        fig1b.write_html(
            f"Fig1b_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{PPP_branch_name}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot1b:
        fig1b.show()


def plot2(
    kilosort_spikes,
    ground_truth_spikes,
    false_positive_spikes,
    false_negative_spikes,
    true_positive_spikes,
    bin_width_for_comparison,
    clusters_in_sort_to_use,
    GT_clusters_to_use,
    sort_from_each_path_to_load,
    plot_template,
    show_plot2,
    plot2_xlim,
    save_png_plot2,
    save_svg_plot2,
    save_html_plot2,
    figsize=(1920, 1080),
):
    # get suffix after the KS folder name, which is the repo branch name for that sort
    PPP_branch_name = list_of_paths_to_sorted_folders[0].name.split("_")[-1]
    sort_type = "Kilosort" if PPP_branch_name == "KS" else "EMUsort"
    # make a subplot for each unit
    subtitles = [
        f"Unit {GT_clusters_to_use[iUnit]}" for iUnit in range(num_motor_units)
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

    # cut all arrays short by the factor of plot2_xlim, all arrays are zeros and ones
    left_bound = int(round(plot2_xlim[0] * len(kilosort_spikes)))
    right_bound = int(round(plot2_xlim[1] * len(kilosort_spikes)))
    kilosort_spikes_cut = kilosort_spikes[left_bound:right_bound, :]
    ground_truth_spikes_cut = ground_truth_spikes[left_bound:right_bound, :]
    false_positive_spikes_cut = false_positive_spikes[left_bound:right_bound, :]
    false_negative_spikes_cut = false_negative_spikes[left_bound:right_bound, :]
    true_positive_spikes_cut = true_positive_spikes[left_bound:right_bound, :]

    for iUnit in range(num_motor_units):
        # add event plots of the kilosort and ground truth spikes, color units according to rainbow
        # add a vertical offset to Kilosort spikes to separate them from ground truth spikes
        # darken the ground truth spikes to distinguish them from Kilosort spikes
        color_KS = "hsl(" + str(iUnit / float(num_motor_units) * 360) + ",100%,50%)"
        color_GT = "hsl(" + str(iUnit / float(num_motor_units) * 360) + ",100%,25%)"
        fig.add_trace(
            go.Scatter(
                x=np.where(kilosort_spikes_cut[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=kilosort_spikes_cut[
                    np.where(kilosort_spikes_cut[:, iUnit] >= 1)[0], iUnit
                ]
                + 0.5,
                mode="markers",
                name=f"Sorted, Unit {clusters_in_sort_to_use[iUnit]}",
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
                x=np.where(ground_truth_spikes_cut[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=ground_truth_spikes_cut[
                    np.where(ground_truth_spikes_cut[:, iUnit] >= 1)[0], iUnit
                ],
                mode="markers",
                name=f"Ground Truth, Unit {GT_clusters_to_use[iUnit]}",
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
                x=np.where(false_positive_spikes_cut[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=np.ones(np.sum(false_positive_spikes_cut[:, iUnit] >= 1)),
                mode="markers",
                name="False Positive",
                marker=dict(color="red", size=5, symbol="x"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.where(false_negative_spikes_cut[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=np.ones(np.sum(false_negative_spikes_cut[:, iUnit] >= 1)) + 1,
                mode="markers",
                name="False Negative",
                marker=dict(color="orange", size=5, symbol="x"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.where(true_positive_spikes_cut[:, iUnit] >= 1)[0]
                * bin_width_for_comparison
                / 1000,
                y=np.ones(np.sum(true_positive_spikes_cut[:, iUnit] >= 1)) + 2,
                mode="markers",
                name="True Positive",
                marker=dict(color="green", size=5, symbol="diamond-tall"),
            ),
            row=2 * iUnit + 2,
            col=1,
        )

    fig.update_layout(
        title=f"<b>Comparison of {sort_type} and Ground Truth Spike Trains, {bin_width_for_comparison} ms Bins</b>",
        template=plot_template,
    )

    fig.update_xaxes(
        title_text="<b>Time (s)</b>",
        row=2 * num_motor_units,
        col=1,
        range=[left_bound / 1000, right_bound / 1000] * int(bin_width_for_comparison),
    )
    # print(
    #     f"set xaxis range to {[left_bound / 1000, right_bound / 1000] * int(bin_width_for_comparison)}"
    # )
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
    clusters_in_sort_to_use,
    GT_clusters_to_use,
    sort_from_each_path_to_load,
    plot_template,
    show_plot3,
    save_png_plot3,
    save_svg_plot3,
    save_html_plot3,
    figsize=(1920, 1080),
):
    # get suffix after the KS folder name, which is the repo branch name for that sort
    PPP_branch_name = list_of_paths_to_sorted_folders[0].name.split("_")[-1]
    sort_type = "Kilosort" if PPP_branch_name == "KS" else "EMUsort"
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
                    name=f"Unit {clusters_in_sort_to_use[iUnit]}",
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
        title=f"<b>Comparison of {sort_type} Performance to Ground Truth, Across Bin Widths</b>",
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
            range=[0, 1.01],
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


def plot4(
    bin_width,
    precision,
    recall,
    accuracy,
    nt0,
    num_motor_units,
    clusters_in_sort_to_use,
    GT_clusters_to_use,
    sort_from_each_path_to_load,
    plot_template,
    show_plot3,
    save_png_plot3,
    save_svg_plot3,
    save_html_plot3,
    figsize=(1920, 1080),
):
    # now add a fourth plot using the same structure as the above
    # this plot will show multiple multi-channel voltage examples of when overlaps occured
    # coloring the location of each individual spike with the median wave for that unit
    # each row in the plot will be a channel, and each column is an example overlap
    # spike_isolation_radius_ms cannot be None or 0

    num_examples = 10  # number of spike examples
    chans_to_use = list(range(6))
    num_chans = len(chans_to_use)  # number of channels
    time_points_of_waveform = 2 * nt0 + 1
    sub_title_list = [str(iStr) for iStr in GT_clusters_to_use]
    fig = subplots.make_subplots(
        rows=num_chans,
        cols=num_examples,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.0,
        horizontal_spacing=0.01,
        subplot_titles=sub_title_list,
    )

    # be sure to use presentation mode to see the full grid
    fig.layout.template = "plotly_white"

    light_colors = [
        "rgb(95,135,255)",
        "rgb(208, 64, 64)",
        "rgb(64, 169, 64)",
        "rgb(255, 170, 30)",
        "rgb(183, 80, 234)",
        "rgb(30, 130, 30)",
        "rgb(255, 158, 158)",
        "rgb(146, 107, 67)",
        "rgb(30,255,255)",
        "rgb(177, 142, 249)",
        "rgb(149, 166, 183)",
        "rgb(190, 112, 75)",
    ]
    dark_colors = [
        "royalblue",
        "firebrick",
        "forestgreen",
        "darkorange",
        "darkorchid",
        "darkgreen",
        "lightcoral",
        "rgb(116, 77, 37)",
        "cyan",
        "mediumpurple",
        "lightslategray",
        "seinna",
    ]

    #     light_colors = [
    #     "lightblue",
    #     "lightcoral",
    #     "lawngreen",
    #     "cyan",
    #     "mediumpurple",
    #     "orange",
    #     "magenta",
    #     "sandybrown"

    # ]
    # dark_colors = [
    #     "darkblue",
    #     "firebrick",
    #     "forestgreen",
    #     "darkcyan",
    #     "purple",
    #     "darkorange",
    #     "darkmagenta",
    #     "seinna",
    #     "darkorchid",
    #     "darkgreen",
    # plot the raw data at 10 randomly chosen GT spike times across 10 channels, then plot any
    # nearby GT or sorted unit median multi-channel waveform with a slight y-axis offset above and
    # ground truth with a y-axis offset below the raw voltage. For each GT spike time chosen, find
    # any spike times within +/- 2 ms (+/- 60 pts), and those will also be plotted. For each
    # cluster and spike time identified, plot the median multi-channel waveform for that cluster
    # at the identified time. Therefore, start by choosing 10 GT spike times, and create a 4D
    # numpy array of np.nans with shape of:
    #   (num_examples, num_motor_units, num_chans, time_points_of_waveform)
    # when these are plotted over/under the raw voltage time (121 points long), all the 61 point
    # waveforms will need to be offset in the x-axis of the subplot by the difference in spike
    # times from the first randomly chosen GT spike time, which will be used as the reference
    # and placed at point 61, make sure to get 1 reference from each of the real GT clusters
    # tricky part will be to loop through all clusters to get the spike times within the time range
    # of the reference and track which other units happened nearby the reference spike
    # once all 10 reference examples are collected, and all other corresponding spike times
    # plot each example as a column, and different rows are the different channels of data

    # initialize snippet container arrays
    raw_data_at_each_ground_truth_time = np.nan * np.ones(
        (num_examples, num_chans, time_points_of_waveform)
    )
    wave_placements_for_each_cluster_ground_truth = np.nan * np.ones(
        (num_examples, num_motor_units, num_chans, time_points_of_waveform)
    )
    wave_placements_for_each_cluster_from_sorter = np.nan * np.ones(
        (num_examples, num_motor_units, num_chans, time_points_of_waveform)
    )
    # get one random spike time from each GT cluster, ensuring the time has a margin of time_points_of_waveform
    reference_time_for_each_GT_cluster = [
        np.random.choice(
            iTimes[
                np.bitwise_and(
                    iTimes > time_points_of_waveform + 1,
                    iTimes < len(sim_ephys_data) - time_points_of_waveform - 1,
                )
            ]
        ).astype(int)
        for iTimes in GT_spike_times_for_each_cluster
    ]

    # extract the time_points_of_waveform-wide spike snippets for each cluster using ground truth spike times
    for iExample, this_spike_time in enumerate(reference_time_for_each_GT_cluster):
        raw_data_at_each_ground_truth_time[iExample, :, :] = sim_ephys_data[
            int(this_spike_time - time_points_of_waveform // 2) : int(
                this_spike_time + time_points_of_waveform // 2 + 1
            ),
            chans_to_use,
        ].T

    # now loop through the GT times and fill the spike_placements arrays with any nearby spikes
    # slice down to only nt0//2, then shift indexing by difference between spike time and GT reference time
    for iExample, this_spike_time in enumerate(reference_time_for_each_GT_cluster):
        for iCluster in GT_clusters_to_use:
            try:
                closest_spike_time_idx_for_this_cluster_sorted = np.asarray(
                    np.abs(
                        spike_times_for_each_cluster[iCluster].astype(float)
                        - this_spike_time
                    )
                    < ephys_fs / 1000
                ).nonzero()[0][0]
                closest_spike_time_idx_for_this_cluster_GT = np.asarray(
                    np.abs(
                        GT_spike_times_for_each_cluster[iCluster].astype(float)
                        - this_spike_time
                    )
                    < ephys_fs / 1000  # 1 ms
                ).nonzero()[0][0]
            except IndexError:
                continue

            # WARNING: check this indexing
            closest_spike_time_for_this_cluster_sorted = spike_times_for_each_cluster[
                iCluster
            ][closest_spike_time_idx_for_this_cluster_sorted]
            closest_spike_time_for_this_cluster_GT = GT_spike_times_for_each_cluster[
                iCluster
            ][closest_spike_time_idx_for_this_cluster_GT]

            spike_time_difference_sorted = (
                closest_spike_time_for_this_cluster_sorted - this_spike_time
            ).astype(int)
            spike_time_difference_GT = (
                closest_spike_time_for_this_cluster_GT - this_spike_time
            ).astype(int)

            assert (
                non_standard_median_spike_snippets_for_each_cluster_GT[0].shape[0]
                == non_standard_median_spike_snippets_for_each_cluster[0].shape[0]
            )

            center_idx_of_wave_matches = (
                non_standard_median_spike_snippets_for_each_cluster[0].shape[0] // 2
            )

            wave_placements_for_each_cluster_from_sorter[
                iExample,
                iCluster,
                :,
                time_points_of_waveform // 4
                + spike_time_difference_sorted : 3 * (time_points_of_waveform // 4)
                + spike_time_difference_sorted,
            ] = non_standard_median_spike_snippets_for_each_cluster[iCluster][
                center_idx_of_wave_matches
                - (time_points_of_waveform // 4) : center_idx_of_wave_matches
                + (time_points_of_waveform // 4),
                chans_to_use,
            ].T  # make (chans, time)

            wave_placements_for_each_cluster_ground_truth[
                iExample,
                iCluster,
                :,
                time_points_of_waveform // 4
                + spike_time_difference_GT : 3 * (time_points_of_waveform // 4)
                + spike_time_difference_GT,
            ] = non_standard_median_spike_snippets_for_each_cluster_GT[iCluster][
                center_idx_of_wave_matches
                - (time_points_of_waveform // 4) : center_idx_of_wave_matches
                + (time_points_of_waveform // 4),
                chans_to_use,
            ].T  # make (chans, time)

    for iCluster in GT_clusters_to_use:
        for iChan, chan in enumerate(chans_to_use):
            # get the mean waveform for this unit on this channel
            raw_GT_waveform = raw_data_at_each_ground_truth_time[iCluster, iChan, :]
            # add the trace of the raw GT waveform
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(raw_GT_waveform))
                    * 1000
                    / ephys_fs,  # get time in ms
                    y=raw_GT_waveform,
                    line=dict(color="black"),
                    line_width=1,
                    showlegend=True,
                ),
                row=iChan + 1,
                col=iCluster + 1,
            )

    # plot the sorter result with slight vertical offset
    for iCluster in GT_clusters_to_use:
        for iChan, chan in enumerate(chans_to_use):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(
                        len(
                            wave_placements_for_each_cluster_from_sorter[
                                iExample, iCluster, iChan, :
                            ]
                        )
                    )
                    * 1000
                    / ephys_fs,
                    y=(
                        wave_placements_for_each_cluster_from_sorter[
                            iExample, iCluster, iChan, :
                        ]
                        + 500  # vertical offset of 500 uV
                    ),
                    line=dict(color=light_colors[iCluster]),
                    line_width=0.5,
                    showlegend=True,
                    name=f"Sorted Unit {iCluster}, Chan {chan}",
                ),
                row=iChan + 1,
                col=iCluster + 1,
            )

    # plot the ground truth on the same level as raw data
    for iCluster in GT_clusters_to_use:
        for iChan, chan in enumerate(chans_to_use):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(
                        len(
                            wave_placements_for_each_cluster_ground_truth[
                                iExample, iCluster, iChan, :
                            ]
                        )
                    )
                    * 1000
                    / ephys_fs,
                    y=(
                        wave_placements_for_each_cluster_ground_truth[
                            iExample, iCluster, iChan, :
                        ]
                    ),
                    line=dict(color=dark_colors[iCluster]),
                    line_width=0.5,
                    showlegend=True,
                    name=f"GT Unit {iCluster}, Chan {chan}",
                ),
                row=iChan + 1,
                col=iCluster + 1,
            )

    fig.update_xaxes(visible=False)
    # fig.update_yaxes(visible=False)
    # make the grid look nice, disable tick labels and y-axis line
    # make all the x and y axes the same
    fig.update_xaxes(showticklabels=False, matches="x", showline=False, showgrid=False)
    fig.update_yaxes(
        showticklabels=False,
        matches="y",
        showline=False,
        showgrid=False,
        zerolinecolor="grey",
    )

    # enable tick label only for column 1 and row 5
    fig.update_xaxes(showticklabels=True, col=1, row=1)
    fig.update_yaxes(showticklabels=True, col=1, row=1)

    # make font size of tick labels smaller and bold
    fig.update_yaxes(
        # tickfont=dict(size=14, family="Arial", color="white"),
        title="Voltage (uV)",
        col=1,
        row=1,
    )

    # make y axis title smaller
    # fig.update_yaxes(
    #     title_font=dict(size=14, family="Arial", color="white"), col=1, row=1
    # )

    # move the y axis title closer to the y axis
    fig.update_yaxes(title_standoff=0, col=1, row=1)

    # make subplot titles bigger
    # fig.update_annotations(font=dict(size=18))

    # make background black
    # fig.update_layout(paper_bgcolor="black", plot_bgcolor="black")

    # make the grid big enough to see
    fig.update_layout(
        # height=500,
        # width=800,
        title_text="<b>Comparison of Ground Truth to Sorter Results</b>",
        # font=dict(size=20, family="Arial", color="white"),
    )
    fig.show()


if __name__ == "__main__":
    # set parameters
    parallel = True
    use_custom_merge_clusters = False
    automatically_assign_cluster_mapping = True
    method_for_automatic_cluster_mapping = "waves"  # can be "waves", "times", or "trains"  what the correlation is computed on to map clusters
    time_frame = [0, 1]  # must be between 0 and 1
    ephys_fs = 30000  # Hz
    xstart = np.log2(
        0.125
    )  # choose bin widths as a range from 0.125 ms to 8 ms in log2 increments
    bin_widths_for_comparison = np.logspace(xstart, -xstart, num=13, base=2)
    bin_widths_for_comparison = [1]
    spike_isolation_radius_ms = 1  # radius of isolation of a spike for it to be removed from consideration. set to positive float, integer, or set None to disable
    iShow = 0  # index of which bin width of bin_widths_for_comparison to show in plots

    nt0 = 121  # number of time bins in the template, in ms it is 3.367
    random_seed_entropy = 218530072159092100005306709809425040261  # 75092699954400878964964014863999053929  # int
    plot_template = "plotly_white"  # ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
    plot1_bar_type = "percent"  # totals / percent
    plot1_ylim = [0, 135]
    plot2_xlim = [0, 1]
    show_plot1a = True
    show_plot1b = True
    show_plot2 = False
    show_plot3 = False
    show_plot4 = False
    save_png_plot1a = False
    save_png_plot1b = False
    save_png_plot2 = False
    save_png_plot3 = False
    save_png_plot4 = False
    save_svg_plot1a = False
    save_svg_plot1b = False
    save_svg_plot2 = False
    save_svg_plot3 = False
    save_svg_plot4 = False
    save_html_plot1a = False
    save_html_plot1b = False
    save_html_plot2 = False
    save_html_plot3 = False
    save_html_plot4 = False

    ## TBD: NEED TO ADD FLAG FOR DATASET CHOICE, to flip all related variables
    ## paths with simulated data
    path_to_sim_dat = Path(
        # "continuous_20221117_godzilla_SNR-400-constant_jitter-0std_files-11.dat"  # triple rat
        "continuous_20221117_godzilla_SNR-None-constant_jitter-0std_files-11.dat"  # godzilla only
        # "continuous_20221117_godzilla_SNR-1-from_data_jitter-4std_files-11.dat"
    )
    ## load ground truth data
    ground_truth_path = Path(
        # "spikes_20221117_godzilla_SNR-400-constant_jitter-0std_files-11.npy"  # triple rat
        "spikes_20221117_godzilla_SNR-1-from_data_jitter-4std_files-11.npy"  # godzilla only
        # "spikes_20221117_godzilla_SNR-1-from_data_jitter-1std_files-5.npy"
        # "spikes_20221116_godzilla_SNR-8-from_data_jitter-4std_files-1.npy"
    )  # spikes_20221116_godzilla_SNR-None_jitter-0std_files-1.npy

    # set which ground truth clusters to compare with (a range from 0 to num_motor_units)
    GT_clusters_to_use = list(range(0, 10))
    num_motor_units = len(GT_clusters_to_use)

    ## load Kilosort data
    # paths to the folders containing the Kilosort data
    paths_to_KS_session_folders = [
        Path(
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221116/"
            "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221117/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/triple/simulated20231219/"
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
        "20231103_184523634126"  # 2 std, 8 jitter, vanilla Kilosort, Th=[1,0.5], spkTh=[-6] $$$ BEST Kilosort3 $$$
        # "20231103_184518491799"  # 2 std, 8 jitter, vanilla Kilosort, Th=[2,1], spkTh=[-6]
        # } All in braces did not have channel delays reintroduced for continuous.dat
        #### Below are with new 16 channel, triple rat dataset.
        # simulated20231219:
        # "20231220_180513756759"  # SNR-400-constant_jitter-0std_files-11, vanilla Kilosort, Th=[10,4], spkTh=[-6]
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

    if parallel:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

    clusters_in_sort_to_use = clusters_to_take_from[sorts_from_each_path_to_load[0]]
    true_spike_counts_for_each_cluster = np.load(str(ground_truth_path)).sum(axis=0)
    # find the folder name which ends in _myo and append to the paths_to_session_folders
    paths_to_each_myo_folder = []
    for iDir in paths_to_KS_session_folders:
        myo = [f for f in iDir.iterdir() if (f.is_dir() and f.name.endswith("_myo"))]
        assert (
            len(myo) == 1
        ), f"There should be one _myo folder in each session folder, but there were {len(myo)} in {iDir}"
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
        ), f"There should be one sort folder match in each _myo folder, but there were {len(matches)} in {iPath}"
        if use_custom_merge_clusters:
            # append the path to the custom_merge_clusters folder
            list_of_paths_to_sorted_folders.append(
                matches[0].joinpath("custom_merges/final_merge")
            )
        else:
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

        # now do the same for the ground truth spikes. Load the ground truth spike times
        # which are 1's and 0's, where 1's indicate a spike and 0's indicate no spike
        # each column is a different unit, and row is a different time point in the recording
        # use np.where to get the spike times for each cluster
        ground_truth_spike_times = np.load(str(ground_truth_path))
        GT_spike_times_for_each_cluster = [
            np.where(ground_truth_spike_times[:, iCluster] == 1)[0]
            for iCluster in GT_clusters_to_use
        ]

        # now use either the spike times or the waveforms to map the clusters according to best
        # correlation score across all pairs of clusters and time lags
        if method_for_automatic_cluster_mapping == "waves":
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
                            int(iSpike_time - nt0 // 2) : int(
                                iSpike_time + nt0 // 2 + 1
                            ),
                            :,
                        ]
                        for iSpike_time in iCluster_spike_times
                    ]
                )
                for iCluster_spike_times in spike_times_for_each_cluster
            ]  # dimensions are (num_spikes, nt0, num_chans_in_recording)

            # store non-standardized waveforms
            non_standard_median_spike_snippets_for_each_cluster = [
                np.median(iCluster_snippets, axis=0)
                for iCluster_snippets in spike_snippets_for_each_cluster
            ]

            # get the median waveform shape for each cluster, but standardize the waveforms
            # before computing the median
            standardized_spike_snippets_for_each_cluster = [
                (iCluster_snippets - np.mean(iCluster_snippets))
                / np.std(iCluster_snippets)
                for iCluster_snippets in spike_snippets_for_each_cluster
            ]
            median_spike_snippets_for_each_cluster = [
                np.median(iCluster_snippets, axis=0)
                for iCluster_snippets in standardized_spike_snippets_for_each_cluster
            ]
            # now extract the waves at the spike times for all clusters, get the GT median waveform

            # get the spike snippets for each cluster in the ground truth
            spike_snippets_for_each_cluster_ground_truth = []
            GT_median_spike_snippets_for_each_cluster = []
            non_standard_median_spike_snippets_for_each_cluster_GT = []
            for iCluster in range(len(GT_spike_times_for_each_cluster)):
                spike_snippets_for_each_cluster_ground_truth.append([])
                # get the spike snippets for each cluster
                for iSpike_time in GT_spike_times_for_each_cluster[iCluster]:
                    if (
                        iSpike_time - nt0 // 2 >= 0
                        and iSpike_time + nt0 // 2 + 1 <= len(sim_ephys_data)
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

                # store non-standardized waveforms
                non_standard_median_spike_snippets_for_each_cluster_GT.append(
                    np.median(
                        spike_snippets_for_each_cluster_ground_truth[iCluster], axis=0
                    )
                )

                # get the median waveform shape for each cluster, but standardize the waveforms
                # before computing the median
                standardized_spike_snippets_for_each_cluster_ground_truth = (
                    spike_snippets_for_each_cluster_ground_truth[iCluster]
                    - np.mean(spike_snippets_for_each_cluster_ground_truth[iCluster])
                ) / np.std(spike_snippets_for_each_cluster_ground_truth[iCluster])

                GT_median_spike_snippets_for_each_cluster.append(
                    np.median(
                        standardized_spike_snippets_for_each_cluster_ground_truth,
                        axis=0,
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
                range(len(GT_median_spike_snippets_for_each_cluster))
            )
            KS_clusters_iter = list(range(len(median_spike_snippets_for_each_cluster)))
            # cluster_mapping = dict()
            # new_cluster_ordering = np.nan * np.ones(
            #     len(GT_median_spike_snippets_for_each_cluster)
            # )
            # sort_clust_IDs_weighted_corr_score_dict = dict()
            # initialize a np.array of correlations for each cluster combination
            correlations = np.zeros((len(GT_clusters_iter), len(KS_clusters_iter)))
            for jCluster_GT in GT_clusters_iter:
                # this will only break out of the loop if KS clusters are fewer than GT clusters
                if len(KS_clusters_iter) == 0:
                    break
                for iCluster_KS in KS_clusters_iter:
                    # initialize a list of correlations for each lag
                    correlations_for_each_lag = []
                    for iLag in range(-nt0 // 2, nt0 // 2 + 1):
                        # compute the correlation between the two median waves
                        correlations_for_each_lag.append(
                            np.corrcoef(
                                np.roll(
                                    median_spike_snippets_for_each_cluster[
                                        iCluster_KS
                                    ].T.flatten(),
                                    iLag,
                                    axis=0,
                                ),
                                GT_median_spike_snippets_for_each_cluster[
                                    jCluster_GT
                                ].T.flatten(),
                            )[0, 1]
                        )
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
                                    true_spike_counts_for_each_cluster[jCluster_GT]
                                    - spike_times_for_each_cluster[iCluster_KS].shape[0]
                                )
                            )
                            ** 2
                        )
                        / (2 * (true_spike_counts_for_each_cluster[jCluster_GT]) ** 2)
                    )
                    # append the highest correlation for these cluster combinations and lags
                    correlations[jCluster_GT, iCluster_KS] = np.max(
                        correlations_for_each_lag
                    )

                    # now scale the correlation by the spike count match index
                    # correlations[jCluster_GT, iCluster_KS] = (
                    #     correlations[jCluster_GT, iCluster_KS] * spike_count_match_score
                    # )
                print(
                    f"Done computing waveform shape correlations for GT cluster {jCluster_GT} of {len(GT_clusters_to_use)-1}"
                )
        elif method_for_automatic_cluster_mapping == "trains":
            # assign cluster mapping by extracting the spike times for all clusters, and then
            # computing the pairwise cross-correlation between each cluster's spike times
            # shift the spike times by all possible lags up to +/- 2 ms, and then take the
            # highest correlation as the match
            # it should produce a 'correlations' variable that is the same format as the "waves"
            # method above, but with correlations computed on the spike times instead of the waves
            # this approach will not use the median waveforms at all, and will only use the spike times
            # to compute the correlation

            # initialize a np.array of correlations for each cluster combination
            correlations = np.zeros(
                (len(GT_clusters_to_use), len(clusters_in_sort_to_use))
            )

            # initialize a numpy array of spike events for each KS cluster (times x clusters)
            KS_spike_events_for_each_cluster = np.zeros(
                (ground_truth_spike_times.shape[0], len(clusters_in_sort_to_use)),
                dtype=np.uint8,
            )
            for iCluster_KS in range(len(clusters_in_sort_to_use)):
                KS_spike_events_for_each_cluster[
                    spike_times_for_each_cluster[iCluster_KS], iCluster_KS
                ] = 1
            # initialize a numpy array of spike events for each GT cluster (times x clusters)
            GT_spike_events_for_each_cluster = np.zeros(
                (ground_truth_spike_times.shape[0], len(GT_clusters_to_use)),
                dtype=np.uint8,
            )
            for jCluster_GT in range(len(GT_clusters_to_use)):
                GT_spike_events_for_each_cluster[
                    GT_spike_times_for_each_cluster[jCluster_GT], jCluster_GT
                ] = 1

            # subsample the spike events to the time frame of interest
            KS_spike_events_for_each_cluster = KS_spike_events_for_each_cluster[
                int(time_frame[0] * ground_truth_spike_times.shape[0]) : int(
                    time_frame[1] * ground_truth_spike_times.shape[0]
                ),
            ]
            GT_spike_events_for_each_cluster = GT_spike_events_for_each_cluster[
                int(time_frame[0] * ground_truth_spike_times.shape[0]) : int(
                    time_frame[1] * ground_truth_spike_times.shape[0]
                ),
            ]

            # subsample further by taking blocks of block_size seconds, every block_spacing seconds
            block_size = 5  # seconds
            block_spacing = 25  # seconds
            block_offset = int(block_size * ephys_fs)
            block_spacing_idx = int(block_spacing * ephys_fs)
            KS_spike_events_for_each_cluster_sub = np.zeros(
                (
                    KS_spike_events_for_each_cluster.shape[0]
                    // round(block_spacing / block_size)
                    + 1,
                    len(clusters_in_sort_to_use),
                ),
                dtype=np.uint8,
            )
            GT_spike_events_for_each_cluster_sub = np.zeros(
                (
                    GT_spike_events_for_each_cluster.shape[0]
                    // round(block_spacing / block_size)
                    + 1,
                    len(GT_clusters_to_use),
                ),
                dtype=np.uint8,
            )
            for ii, iBlock in enumerate(
                range(0, KS_spike_events_for_each_cluster.shape[0], block_spacing_idx)
            ):  # loop through, assigning the blocks from the original spike events into the subsampled spike events
                # we are not summing, we are just slicing the original spike events into the subsampled spike events
                # skip last block
                if (
                    ii + 1
                ) * block_offset >= KS_spike_events_for_each_cluster_sub.shape[0]:
                    break
                KS_spike_events_for_each_cluster_sub[
                    ii * block_offset : (1 + ii) * block_offset
                ] = KS_spike_events_for_each_cluster[iBlock : iBlock + block_offset]
                GT_spike_events_for_each_cluster_sub[
                    ii * block_offset : (1 + ii) * block_offset
                ] = GT_spike_events_for_each_cluster[iBlock : iBlock + block_offset]
            KS_spike_events_for_each_cluster = KS_spike_events_for_each_cluster_sub
            GT_spike_events_for_each_cluster = GT_spike_events_for_each_cluster_sub

            # now compute a proxy of correlation by matrix multiplication
            # use np.roll to roll the entire KS matrix by all possible lags
            # but do not roll the GT matrix, so that the correlation is computed
            min_delay_ms = -1  # ms
            max_delay_ms = 1  # ms
            min_delay_samples = int(round(min_delay_ms * ephys_fs / 1000))
            max_delay_samples = int(round(max_delay_ms * ephys_fs / 1000))

            # # compute the pairwise correlation between the two median waves
            # for jCluster_GT in range(len(GT_clusters_to_use)):
            #     for iCluster_KS in range(len(clusters_in_sort_to_use)):
            #         # initialize a list of correlations for each lag
            #         correlations_for_each_lag = []
            #         for iLag in range(min_delay_samples, max_delay_samples + 1):
            #             correlations_for_each_lag.append(
            #                 np.corrcoef(
            #                     np.roll(
            #                         KS_spike_events_for_each_cluster[:, iCluster_KS],
            #                         iLag,
            #                         axis=0,
            #                     ),
            #                     GT_spike_events_for_each_cluster[:, jCluster_GT],
            #                 )[0, 1]
            #             )
            #         # append the highest correlation for these cluster combinations and lags
            #         correlations[jCluster_GT, iCluster_KS] = np.max(
            #             correlations_for_each_lag
            #         )
            #     print(
            #         f"Done computing spike time correlations for GT cluster {jCluster_GT} of {len(GT_clusters_to_use)-1}"
            #     )

            # make the above nested for loop into a parallel for loop
            def compute_train_correlations_for_each_GT_cluster(jCluster_GT):
                for iCluster_KS in range(len(clusters_in_sort_to_use)):
                    # initialize a list of correlations for each lag
                    correlations_for_each_lag = []
                    for iLag in range(min_delay_samples, max_delay_samples + 1):
                        correlations_for_each_lag.append(
                            np.corrcoef(
                                np.roll(
                                    KS_spike_events_for_each_cluster[:, iCluster_KS],
                                    iLag,
                                    axis=0,
                                ),
                                GT_spike_events_for_each_cluster[:, jCluster_GT],
                            )[0, 1]
                        )
                print(
                    f"Done computing spike train correlations for GT cluster {jCluster_GT} of {len(GT_clusters_to_use)-1}"
                )
                return (
                    np.max(correlations_for_each_lag),
                    jCluster_GT,
                )  # return the highest correlation and the GT cluster index

            if parallel:
                with ProcessPoolExecutor(
                    max_workers=min(mp.cpu_count() // 2, num_motor_units)
                ) as executor:
                    futures = [
                        executor.submit(
                            compute_train_correlations_for_each_GT_cluster, jCluster_GT
                        )
                        for jCluster_GT in range(len(GT_clusters_to_use))
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        correlations[result[1], :] = result[0]
            else:
                for jCluster_GT in range(len(GT_clusters_to_use)):
                    correlations[
                        jCluster_GT, :
                    ] = compute_train_correlations_for_each_GT_cluster(jCluster_GT)[0]
        elif method_for_automatic_cluster_mapping == "times":
            # this method will work by looping through the spike times (not the arrays of 1's/0's)
            # it will loop through each GT cluster, and each KS cluster, and for the times of each
            # GT cluster, it will take the difference between that and all the times of the KS cluster
            # it will histogram the differences to 1ms bins, and then take the highest bin as the
            # correlation score for that GT cluster and KS cluster
            # output will be in the same format as the above other methods, but with correlations
            # computed on the spike times instead of the waves or entire spike trains
            # each GT cluster will be computed in parallel

            # initialize a np.array of correlations for each cluster combination
            correlations = np.zeros(
                (len(GT_clusters_to_use), len(clusters_in_sort_to_use))
            )

            def compute_spike_time_correlations_for_each_KS_cluster(
                jCluster_GT, iCluster_KS
            ):
                # initialize a list of correlations for each lag
                correlations_for_each_lag = []
                # loop through the spike times for the GT cluster
                for iSpike_time_GT in GT_spike_times_for_each_cluster[jCluster_GT]:
                    # take the difference between this spike time and all the spike times for the KS cluster
                    # and then histogram the differences to +/- 1ms bins, then sum the histogram for
                    # then take the highest bin as the correlation score for that GT cluster and KS cluster
                    correlations_for_each_lag.append(
                        spike_times_for_each_cluster[iCluster_KS].astype(int)
                        - iSpike_time_GT
                    )

                hist = np.histogram(
                    correlations_for_each_lag,
                    bins=2
                    * np.arange(
                        -int(ephys_fs / 1000),
                        int(ephys_fs / 1000) + 1,
                        int(ephys_fs / 500),
                    ),
                    # for bins, collect all spikes within +/- 2ms,
                    # but restrict to one or the other, to disallow erroneous correlations due to jitter
                )

                return (
                    np.max(hist[0] / len(GT_spike_times_for_each_cluster[jCluster_GT])),
                    iCluster_KS,
                )

            # define a function for parallel execution
            def compute_spike_time_correlations_for_each_GT_cluster(jCluster_GT):
                results_container = []
                if parallel:
                    with ProcessPoolExecutor(
                        max_workers=min(
                            mp.cpu_count() // num_motor_units,
                            len(clusters_in_sort_to_use),
                        )
                    ) as executor:
                        futures = [
                            executor.submit(
                                compute_spike_time_correlations_for_each_KS_cluster,
                                jCluster_GT,
                                iCluster_KS,
                            )
                            for iCluster_KS in range(len(clusters_in_sort_to_use))
                        ]
                        for future in as_completed(futures):
                            result = future.result()
                            results_container.append(result)
                            print(
                                f"Done computing spike time correlations for GT cluster {jCluster_GT}"
                                f" and KS cluster {result[1]} with correlation score {np.round(result[0],4)}"
                            )
                    # get out each result from the container in separate lists
                else:
                    for iCluster_KS in range(len(clusters_in_sort_to_use)):
                        results_container.append(
                            compute_spike_time_correlations_for_each_KS_cluster(
                                jCluster_GT, iCluster_KS
                            )
                        )

                corr_list = [iResult[0] for iResult in results_container]
                KS_cluster_list = [iResult[1] for iResult in results_container]

                # sort corr_list by order of KS_cluster_list
                corr_list = [corr_list[i] for i in np.argsort(KS_cluster_list)]

                return (corr_list, jCluster_GT)

            if parallel:
                with ProcessPoolExecutor(
                    max_workers=min(mp.cpu_count() // num_motor_units, num_motor_units)
                ) as executor:
                    futures = [
                        executor.submit(
                            compute_spike_time_correlations_for_each_GT_cluster,
                            jCluster_GT,
                        )
                        for jCluster_GT in range(len(GT_clusters_to_use))
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        correlations[result[1], :] = result[0]
            else:
                for jCluster_GT in range(len(GT_clusters_to_use)):
                    correlations[
                        jCluster_GT, :
                    ] = compute_spike_time_correlations_for_each_GT_cluster(
                        jCluster_GT
                    )[
                        0
                    ]

        else:
            raise Exception(
                f"method_for_automatic_cluster_mapping must be either 'waves', 'times', or 'trains', but was {method_for_automatic_cluster_mapping}"
            )

        # now find the cluster with the highest correlation
        sorted_cluster_pair_corr_idx = np.unravel_index(
            np.argsort(correlations.ravel()), correlations.shape
        )
        sorted_GT_cluster_match_idxs = np.flip(sorted_cluster_pair_corr_idx[0])
        sorted_KS_cluster_match_idxs = np.flip(sorted_cluster_pair_corr_idx[1])

        GT_mapped_idxs = np.nan * np.ones((len(sorted_GT_cluster_match_idxs), 2))
        GT_mapped_idxs[:, 0] = sorted_GT_cluster_match_idxs
        GT_mapped_idxs[:, 1] = sorted_KS_cluster_match_idxs
        GT_mapped_idxs = GT_mapped_idxs.astype(int)

        # now extract only non-repeated rows from the first column
        # (so there's only one match per GT cluster, in order)
        [uniq_GT, uniq_GT_idx, uniq_GT_inv_idx] = np.unique(
            GT_mapped_idxs[:, 0], return_index=True, return_inverse=True
        )
        [uniq_KS, uniq_KS_idx, uniq_KS_inv_idx] = np.unique(
            GT_mapped_idxs[:, 1], return_index=True, return_inverse=True
        )

        # find the ordering of which GT clusters have the highest score
        def get_unique_N(iterable, N):
            """Yields (in order) the first N unique elements of iterable.
            Might yield less if data too short."""
            seen = set()
            for e in iterable:
                if e in seen:
                    continue
                seen.add(e)
                yield e
                if len(seen) == N:
                    return

        sorted_by_corr_uniq_GT = get_unique_N(uniq_GT_inv_idx, len(uniq_GT))

        best_uniq_pair_idxs = np.nan * np.ones_like(uniq_GT_idx)
        claimed_KS_idxs = np.nan * np.ones_like(uniq_GT_idx)
        for iGT_clust_idx in sorted_by_corr_uniq_GT:
            best_corr_clust_match_idxs = np.where(uniq_GT_inv_idx == iGT_clust_idx)
            for iBest_corr_match in best_corr_clust_match_idxs[0]:
                if GT_mapped_idxs[iBest_corr_match][1] not in claimed_KS_idxs:
                    best_uniq_pair_idxs[iGT_clust_idx] = iBest_corr_match
                    claimed_KS_idxs[iGT_clust_idx] = GT_mapped_idxs[iBest_corr_match][1]
                    break
            else:
                raise Exception(
                    f"No unique cluster match was found for ground truth cluster {iGT_clust_idx}!"
                )

        best_uniq_pair_idxs = best_uniq_pair_idxs.astype(int)
        GT_mapped_idxs = GT_mapped_idxs[best_uniq_pair_idxs]
        GT_mapped_corrs = [
            correlations[idx_pair[0], idx_pair[1]] for idx_pair in GT_mapped_idxs
        ]
        # sorted_by_GT_idxs = np.argsort(GT_mapped_idxs[:, 0])
        # GT_mapped_idxs = GT_mapped_idxs[sorted_by_GT_idxs]
        # GT_cluster_match_idx = best_cluster_pair_corr_idx // len(
        #     GT_median_spike_snippets_for_each_cluster
        # )
        # KS_cluster_match_idx = best_cluster_pair_corr_idx % len(
        #     GT_median_spike_snippets_for_each_cluster
        # )
        # track the highest correlation for each sort cluster in a dictionary
        # sort_clust_IDs_weighted_corr_score_dict[
        #     KS_clusters_iter[KS_cluster_match_idx]
        # ] = correlations[KS_cluster_match_idx]
        # now assign the cluster mapping, where the key is the sort cluster
        # and the value is the matching ground truth cluster
        # cluster_mapping[KS_clusters_iter[KS_cluster_match_idx]] = GT_cluster_match_idx
        # now remove the best matching GT and KS clusters from the lists to search from
        # GT_clusters_iter.pop(GT_cluster_match_idx)
        # KS_clusters_iter.pop(KS_cluster_match_idx)
        # now loop through the clusters in 'clusters_to_take_from' and use as key to cluster_mapping
        # to get the ground truth cluster that it maps to, and then place the sort cluster ID into
        # the new_cluster_ordering array at the index of the ground truth cluster ID
        # for iCluster in range(len(clusters_in_sort_to_use)):
        #     try:
        #         new_cluster_ordering[
        #             cluster_mapping[clusters_in_sort_to_use[iCluster]]
        #         ] = clusters_in_sort_to_use[iCluster]
        #     except KeyError:
        #         continue

        # now replace the clusters_to_take_from with the new_cluster_ordering
        # clusters_in_sort_to_use = new_cluster_ordering.astype(int)
        # cluster_mapped_indexes = np.fromiter(cluster_mapping.keys(), int)
        # clusters_in_sort_to_use = [
        #     clusters_in_sort_to_use[idx] for idx in cluster_mapped_indexes
        # ]
        clusters_in_sort_to_use = [
            clusters_in_sort_to_use[idx] for idx in GT_mapped_idxs[:, 1]
        ]
        # num_motor_units = len(clusters_in_sort_to_use)

        corr_df = df(
            GT_clusters_to_use,  # GT_mapped_idxs[:, 0],
            columns=["GT Unit"],
            index=clusters_in_sort_to_use,
        )
        corr_df.index.name = "KS Unit"
        corr_df["Corr. Score"] = np.array(GT_mapped_corrs)
        print(corr_df)

    if parallel:
        with mp.Pool(processes=len(bin_widths_for_comparison)) as pool:
            zip_obj = zip(
                [ground_truth_path] * len(bin_widths_for_comparison),
                [GT_clusters_to_use] * len(bin_widths_for_comparison),
                [random_seed_entropy] * len(bin_widths_for_comparison),
                bin_widths_for_comparison,
                [ephys_fs] * len(bin_widths_for_comparison),
                [time_frame] * len(bin_widths_for_comparison),
                [list_of_paths_to_sorted_folders] * len(bin_widths_for_comparison),
                [clusters_in_sort_to_use] * len(bin_widths_for_comparison),
                [spike_isolation_radius_ms] * len(bin_widths_for_comparison),
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
                GT_clusters_to_use,
                random_seed_entropy,
                bin_widths_for_comparison[iBinWidth],
                ephys_fs,
                time_frame,
                list_of_paths_to_sorted_folders,
                clusters_in_sort_to_use,
                spike_isolation_radius_ms,
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

    if (
        show_plot1a
        or save_png_plot1a
        or save_html_plot1a
        or save_svg_plot1a
        or show_plot1b
        or save_png_plot1b
        or save_html_plot1b
        or save_svg_plot1b
    ):
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
            GT_clusters_to_use,
            sorts_from_each_path_to_load[0],
            plot_template,
            plot1_bar_type,
            plot1_ylim,
            show_plot1a,
            save_png_plot1a,
            save_svg_plot1a,
            save_html_plot1a,
            show_plot1b,
            save_png_plot1b,
            save_svg_plot1b,
            save_html_plot1b,
            # make figsize 1080p
            figsize=(1280, 1440),
        )

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
            GT_clusters_to_use,
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
            GT_clusters_to_use,
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot3,
            save_png_plot3,
            save_svg_plot3,
            save_html_plot3,
            # make figsize 1080p
            figsize=(1920, 1080),
        )

    if show_plot4 or save_png_plot4 or save_html_plot4 or save_svg_plot4:
        ### plot 4: examples of overlaps throughout sort to validate results
        plot4(
            bin_widths_for_comparison,
            precisions,
            recalls,
            accuracies,
            nt0,
            num_motor_units,
            clusters_in_sort_to_use,
            GT_clusters_to_use,
            sorts_from_each_path_to_load[0],
            plot_template,
            show_plot4,
            save_png_plot4,
            save_svg_plot4,
            save_html_plot4,
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
