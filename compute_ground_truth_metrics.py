# IMPORT packages
import gc
from datetime import datetime
from pathlib import Path
from pdb import set_trace

import colorlover as cl
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
from mat73 import loadmat
from pandas import DataFrame as df
from scipy.signal import correlate, correlation_lags

from MUsim import MUsim

# import tracemalloc
# from collections import Counter
# import linecache
# import os


start_time = datetime.now()  # begin timer for script execution time


# def display_top(snapshot, key_type="lineno", limit=3):
#     snapshot = snapshot.filter_traces(
#         (
#             tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#             tracemalloc.Filter(False, "<unknown>"),
#         )
#     )
#     top_stats = snapshot.statistics(key_type)

#     print("Top %s lines" % limit)
#     for index, stat in enumerate(top_stats[:limit], 1):
#         frame = stat.traceback[0]
#         # replace "/path/to/module/file.py" with "module/file.py"
#         filename = os.sep.join(frame.filename.split(os.sep)[-2:])
#         print(
#             "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
#         )
#         line = linecache.getline(frame.filename, frame.lineno).strip()
#         if line:
#             print("    %s" % line)

#     other = top_stats[limit:]
#     if other:
#         size = sum(stat.size for stat in other)
#         print("%s other: %.1f KiB" % (len(other), size / 1024))
#     total = sum(stat.size for stat in top_stats)
#     print("Total allocated size: %.1f KiB" % (total / 1024))


# define a function to convert a timedelta object to a pretty string representation
def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


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


def compute_precision(num_matches, num_kilosort_spikes):
    result = num_matches / num_kilosort_spikes
    return result


def compute_recall(num_matches, num_ground_truth_spikes, jCluster_GT=None):
    if jCluster_GT is None:
        result = num_matches / num_ground_truth_spikes
    else:
        result = num_matches / num_ground_truth_spikes[jCluster_GT]
    return result


def compute_accuracy(
    num_matches, num_kilosort_spikes, num_ground_truth_spikes, jCluster_GT=None
):
    if jCluster_GT is None:
        result = num_matches / (
            num_kilosort_spikes + num_ground_truth_spikes - num_matches
        )
    else:
        result = num_matches / (
            num_kilosort_spikes + num_ground_truth_spikes[jCluster_GT] - num_matches
        )
    return result


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
    isolated_spike_times = np.asarray(MUsim_obj.spikes[-1].sum(axis=1) == 1).nonzero()[
        0
    ]
    spike_counter = 0
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
                # set_trace()
                MUsim_obj.spikes[-1][
                    iTime - radius : iTime + radius,
                    :,
                ] = 0
                spike_counter += 1
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
        MUsim_obj.removed_spike_count = spike_counter
    return MUsim_obj


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

    # use MUsim object to load and rebin ground truth data
    mu_GT = MUsim(random_seed_entropy)
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
        mu_GT = remove_isolated_spikes(mu_GT, spike_isolation_radius_pts)
        mu_KS = remove_isolated_spikes(mu_KS, spike_isolation_radius_pts)
        print(
            f"removed {mu_GT.removed_spike_count} isolated spikes from GT spikes with radius {spike_isolation_radius_pts} pts"
        )
        print(
            f"removed {mu_KS.removed_spike_count} isolated spikes from KS spikes with radius {spike_isolation_radius_pts} pts"
        )

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


def find_best_cluster_matches(
    correlations,
    precisions,
    recalls,
    accuracies,
    num_matches,
    num_kilosort_spikes,
    num_ground_truth_spikes,
    true_positive_spikes,
    false_positive_spikes,
    false_negative_spikes,
    kilosort_spikes,
    ground_truth_spikes,
    method_for_automatic_cluster_mapping,
    KS_clusters_to_consider=None,
):
    # loop through each correlations list element, which house scores for all GT clusters in each sort
    # GT_mapped_corrs_list = []
    # precisions_list = []
    # recalls_list = []
    # accuracies_list = []
    # num_matches_list = []
    # num_kilosort_spikes_list = []
    # num_ground_truth_spikes_list = []
    # true_positive_spikes_list = []
    # false_positive_spikes_list = []
    # false_negative_spikes_list = []
    # kilosort_spikes_list = []
    # ground_truth_spikes_list = []
    if KS_clusters_to_consider is None:
        clusters_in_sort_to_use_list = []
    else:
        clusters_in_sort_to_use_list = KS_clusters_to_consider

    for iSort in range(len(correlations)):
        if KS_clusters_to_consider is None:
            # now find the cluster with the highest correlation
            sorted_cluster_pair_corr_idx = np.unravel_index(
                np.argsort(correlations[iSort].ravel()), correlations[iSort].shape
            )

            # make sure to slice off coordinate pairs with a nan result
            num_nan_pairs = np.isnan(np.sort(correlations[iSort].ravel())).sum()
            sorted_GT_cluster_match_idxs = np.flip(sorted_cluster_pair_corr_idx[0])[
                num_nan_pairs:
            ]
            sorted_KS_cluster_match_idxs = np.flip(sorted_cluster_pair_corr_idx[1])[
                num_nan_pairs:
            ]

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
            sorted_by_corr_uniq_GT = get_unique_N(uniq_GT_inv_idx, len(uniq_GT))
            best_uniq_pair_idxs = np.nan * np.ones_like(uniq_GT_idx)
            claimed_KS_idxs = np.nan * np.ones_like(uniq_GT_idx)
            for iGT_clust_idx in sorted_by_corr_uniq_GT:
                best_corr_clust_match_idxs = np.where(uniq_GT_inv_idx == iGT_clust_idx)
                for iBest_corr_match in best_corr_clust_match_idxs[0]:
                    if GT_mapped_idxs[iBest_corr_match][1] not in claimed_KS_idxs:
                        best_uniq_pair_idxs[iGT_clust_idx] = iBest_corr_match
                        claimed_KS_idxs[iGT_clust_idx] = GT_mapped_idxs[
                            iBest_corr_match
                        ][1]
                        break
                else:
                    print(
                        f"WARNING: GT cluster {iGT_clust_idx} was not matched to any KS cluster"
                    )
                    set_trace()

            best_uniq_pair_idxs = best_uniq_pair_idxs.astype(int)
            GT_mapped_idxs = GT_mapped_idxs[best_uniq_pair_idxs]
            GT_mapped_corrs = [
                correlations[iSort][idx_pair[0], idx_pair[1]]
                for idx_pair in GT_mapped_idxs
            ]

            if method_for_automatic_cluster_mapping == "accuracies":
                GT_mapped_precisions = [
                    precisions[iSort][idx_pair[0], idx_pair[1]]
                    for idx_pair in GT_mapped_idxs
                ]
                GT_mapped_recalls = [
                    recalls[iSort][idx_pair[0], idx_pair[1]]
                    for idx_pair in GT_mapped_idxs
                ]
                GT_mapped_accuracies = [
                    accuracies[iSort][idx_pair[0], idx_pair[1]]
                    for idx_pair in GT_mapped_idxs
                ]
                if (
                    num_matches[iSort].shape[1] == 1
                    and num_kilosort_spikes[iSort].shape[1] == 1
                ):
                    GT_mapped_num_matches = num_matches[iSort].flatten().tolist()
                    GT_mapped_num_KS_spikes = (
                        num_kilosort_spikes[iSort].flatten().tolist()
                    )
                else:
                    GT_mapped_num_matches = [
                        num_matches[iSort][idx_pair[0], idx_pair[1]]
                        for idx_pair in GT_mapped_idxs
                    ]
                    GT_mapped_num_KS_spikes = [
                        num_kilosort_spikes[iSort][idx_pair[0], idx_pair[1]]
                        for idx_pair in GT_mapped_idxs
                    ]

                # shape of metrics is (num_GT_clusters, num_KS_clusters), but KS clusters were
                # pared down to only the ones that matched to a GT cluster, so the shape of the
                # metrics will effectively be (num_GT_clusters, num_GT_clusters)
                accuracies[iSort] = np.array(GT_mapped_accuracies)
                precisions[iSort] = np.array(GT_mapped_precisions)
                recalls[iSort] = np.array(GT_mapped_recalls)
                num_matches[iSort] = np.array(GT_mapped_num_matches)
                num_kilosort_spikes[iSort] = np.array(GT_mapped_num_KS_spikes).astype(
                    int
                )
                # they are copies, so just take the first one
                num_ground_truth_spikes[iSort] = num_ground_truth_spikes[iSort][0]
                ground_truth_spikes[iSort] = np.array(ground_truth_spikes[iSort])[0]

                if (
                    true_positive_spikes[iSort].shape[2] == 1
                    and false_positive_spikes[iSort].shape[2] == 1
                    and false_negative_spikes[iSort].shape[2] == 1
                    and kilosort_spikes[iSort].shape[2] == 1
                ):
                    true_positive_spikes[iSort] = (
                        true_positive_spikes[iSort].squeeze().T
                    )
                    false_positive_spikes[iSort] = (
                        false_positive_spikes[iSort].squeeze().T
                    )
                    false_negative_spikes[iSort] = (
                        false_negative_spikes[iSort].squeeze().T
                    )
                    kilosort_spikes[iSort] = kilosort_spikes[iSort].squeeze().T
                else:
                    true_positive_spikes[iSort] = true_positive_spikes[iSort][
                        GT_mapped_idxs[:, 0], :, GT_mapped_idxs[:, 1]
                    ].T
                    false_positive_spikes[iSort] = false_positive_spikes[iSort][
                        GT_mapped_idxs[:, 0], :, GT_mapped_idxs[:, 1]
                    ].T
                    false_negative_spikes[iSort] = false_negative_spikes[iSort][
                        GT_mapped_idxs[:, 0], :, GT_mapped_idxs[:, 1]
                    ].T
                    kilosort_spikes[iSort] = kilosort_spikes[iSort][
                        GT_mapped_idxs[:, 0], :, GT_mapped_idxs[:, 1]
                    ].T

                    # true_positive_spikes[iSort] = true_positive_spikes[iSort][
                    #     :, GT_mapped_idxs[:, 1]
                    # ]
                    # false_positive_spikes[iSort] = false_positive_spikes[iSort][
                    #     :, GT_mapped_idxs[:, 1]
                    # ]
                    # false_negative_spikes[iSort] = false_negative_spikes[iSort][
                    #     :, GT_mapped_idxs[:, 1]
                    # ]
                    # kilosort_spikes[iSort] = kilosort_spikes[iSort][:, GT_mapped_idxs[:, 1]]
                clusters_in_sort_to_use = GT_mapped_idxs[:, 1]

                # append each stacked array to each corresponding list
                # accuracies_list.append(accuracies)
                # precisions_list.append(precisions)
                # recalls_list.append(recalls)
                # num_matches_list.append(num_matches)
                # num_kilosort_spikes_list.append(num_kilosort_spikes)
                # num_ground_truth_spikes_list.append(num_ground_truth_spikes)
                # true_positive_spikes_list.append(true_positive_spikes)
                # false_positive_spikes_list.append(false_positive_spikes)
                # false_negative_spikes_list.append(false_negative_spikes)
                # kilosort_spikes_list.append(kilosort_spikes)
                # ground_truth_spikes_list.append(ground_truth_spikes)
                clusters_in_sort_to_use_list.append(clusters_in_sort_to_use)

                metrics_df = df(
                    GT_clusters_to_use,  # GT_mapped_idxs[:, 0],
                    columns=["GT Unit"],
                    index=clusters_in_sort_to_use,
                )
                metrics_df.index.name = "KS Unit"
                metrics_df["Precision"] = np.array(GT_mapped_precisions)
                metrics_df["Recall"] = np.array(GT_mapped_recalls)
                metrics_df["Accuracy"] = np.array(GT_mapped_accuracies)
                print(metrics_df)
            else:
                clusters_in_sort_to_use = [
                    clusters_in_sort_to_use[idx] for idx in GT_mapped_idxs[:, 1]
                ]
            score_df = df(
                GT_clusters_to_use,  # GT_mapped_idxs[:, 0],
                columns=["GT Unit"],
                index=clusters_in_sort_to_use,
            )
            score_df.index.name = "KS Unit"
            score_df["Match Score"] = np.array(GT_mapped_corrs)
            print(score_df)
        else:
            # get first elements across all GT clusters, since they are all identical
            # because KS clusters were already specified with KS_clusters_to_consider
            accuracies[iSort] = np.array(
                [
                    accuracies[iSort][ii, iKS]
                    for ii, iKS in enumerate(KS_clusters_to_consider[iSort])
                ]
            ).flatten()
            precisions[iSort] = np.array(
                [
                    precisions[iSort][ii, iKS]
                    for ii, iKS in enumerate(KS_clusters_to_consider[iSort])
                ]
            ).flatten()
            recalls[iSort] = np.array(
                [
                    recalls[iSort][ii, iKS]
                    for ii, iKS in enumerate(KS_clusters_to_consider[iSort])
                ]
            ).flatten()
            num_matches[iSort] = np.array(num_matches[iSort]).flatten()
            num_kilosort_spikes[iSort] = np.array(num_kilosort_spikes[iSort]).flatten()

            num_ground_truth_spikes[iSort] = num_ground_truth_spikes[iSort][0]
            ground_truth_spikes[iSort] = np.array(ground_truth_spikes[iSort])[0]

            # ensure proper shaping of the *_spikes variables
            if (
                true_positive_spikes[iSort].shape[2] == 1
                and false_positive_spikes[iSort].shape[2] == 1
                and false_negative_spikes[iSort].shape[2] == 1
                and kilosort_spikes[iSort].shape[2] == 1
            ):
                true_positive_spikes[iSort] = true_positive_spikes[iSort].squeeze().T
                false_positive_spikes[iSort] = false_positive_spikes[iSort].squeeze().T
                false_negative_spikes[iSort] = false_negative_spikes[iSort].squeeze().T
                kilosort_spikes[iSort] = kilosort_spikes[iSort].squeeze().T
        # print metrics for each unit
        print("\n")  # add a newline for readability
        print("Sort: ", sorts_from_each_path_to_load[iSort])

        unit_df = df()
        unit_df["Unit"] = np.array(clusters_in_sort_to_use_list[iSort]).astype(int)
        unit_df["True Count"] = num_ground_truth_spikes[iSort][GT_clusters_to_use]
        unit_df["KS Count"] = num_kilosort_spikes[iSort]
        unit_df["Precision"] = precisions[iSort]
        unit_df["Recall"] = recalls[iSort]
        unit_df["Accuracy"] = accuracies[iSort]
        # unit_df["Unit"].astype(int)
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

        print("\n")  # add a newline for readability
    return (
        precisions,
        recalls,
        accuracies,
        num_matches,
        num_kilosort_spikes,
        num_ground_truth_spikes,
        true_positive_spikes,
        false_positive_spikes,
        false_negative_spikes,
        kilosort_spikes,
        ground_truth_spikes,
        clusters_in_sort_to_use_list,  # one list per sort, of KS cluster IDs paired to GT clusters
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
    sorts_from_each_path_to_load,
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
    # search all the paths to sorted folders for the KS or EMUsort string, in the order of matches with sorts_from_each_path_to_load
    sort_types = [None] * len(sorts_from_each_path_to_load)
    for iS in range(len(sorts_from_each_path_to_load)):
        for iP in range(len(list_of_paths_to_sorted_folders[0])):
            if (
                sorts_from_each_path_to_load[iS]
                in list_of_paths_to_sorted_folders[0][iP].name
            ):
                sort_types[iS] = (
                    "Kilosort"
                    if list_of_paths_to_sorted_folders[0][iP].name.split("_")[-1]
                    == "KS"
                    else "EMUsort"
                )
                break
    # make sure all sort_types were found
    assert None not in sort_types, "Not all sort_types were found"

    for iSort in range(len(sorts_from_each_path_to_load)):
        # get suffix after the KS folder name, which is the repo branch name for that sort
        # PPP_branch_name = list_of_paths_to_sorted_folders[0][iSort].name.split("_")[-1]
        # sort_type = "Kilosort" if "KS" in PPP_branch_name else "EMUsort"

        if show_plot1a or save_png_plot1a or save_svg_plot1a or save_html_plot1a:
            fig1a = go.Figure()
            fig1a.add_trace(
                go.Scatter(
                    x=np.arange(0, num_motor_units),
                    y=precision[iSort],
                    mode="lines+markers",
                    name="Precision",
                    line=dict(width=4, color="green"),
                    # yaxis="y2",
                )
            )
            fig1a.add_trace(
                go.Scatter(
                    x=np.arange(0, num_motor_units),
                    y=recall[iSort],
                    mode="lines+markers",
                    name="Recall",
                    line=dict(width=4, color="crimson"),
                    # yaxis="y2",
                )
            )
            fig1a.add_trace(
                go.Scatter(
                    x=np.arange(0, num_motor_units),
                    y=accuracy[iSort],
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
                    "text": f"<b>Comparison of {sort_types[iSort]} Performance to Ground Truth, {bin_width_for_comparison[0]} ms Bins</b><br><sup>Sort: {sorts_from_each_path_to_load[iSort]}</sup>",
                    # "y": 0.95,
                },
                xaxis_title="<b>GT Cluster ID,<br>True Count</b>",
                # legend_title="Ground Truth Metrics",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                ),
                template=plot_template,
                yaxis=dict(
                    title="<b>Metric Score</b>",
                    title_standoff=1,
                    range=[0, 1.1],
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
                    f"Unit {GT_clusters_to_use[iUnit]},<br>{str(round(num_ground_truth_spikes[iSort][iUnit]/1000,1))}k"
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
                        y=num_ground_truth_spikes[iSort],
                        name="Ground Truth",
                        marker_color="rgb(55, 83, 109)",
                        opacity=0.5,
                    )
                )
                fig1b.add_trace(
                    go.Bar(
                        x=np.arange(0, num_motor_units),
                        y=num_kilosort_spikes[iSort],
                        name=sort_types,
                        marker_color="rgb(26, 118, 255)",
                        opacity=0.5,
                    )
                )
                bar_yaxis_title = "<b>Spike Count</b>"
            elif plot1_bar_type == "percent":
                fig1b.add_trace(
                    go.Bar(
                        x=np.arange(0, num_motor_units),
                        y=100
                        * num_kilosort_spikes[iSort]
                        / num_ground_truth_spikes[iSort],
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
                    "text": f"<b>True Spike Count Captured for Each Cluster Using {sort_types}, {bin_width_for_comparison} ms Bins</b><br><sup>Sort: {sorts_from_each_path_to_load[iSort]}</sup>",
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
                f"plot1/plot1a_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sorts_from_each_path_to_load[iSort]}_{sort_types}.png",
                width=figsize[0],
                height=figsize[1],
            )
        if save_svg_plot1a:
            fig1a.write_image(
                f"plot1/plot1a_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sorts_from_each_path_to_load[iSort]}_{sort_types}.svg",
                width=figsize[0],
                height=figsize[1],
            )
        if save_html_plot1a:
            fig1a.write_html(
                f"plot1/plot1a_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sorts_from_each_path_to_load[iSort]}_{sort_types}.html",
                include_plotlyjs="cdn",
                full_html=False,
            )
        if show_plot1a:
            fig1a.show()

        if save_png_plot1b:
            fig1b.write_image(
                f"plot1/plot1b_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sorts_from_each_path_to_load[iSort]}_{sort_types}.png",
                width=figsize[0],
                height=figsize[1],
            )
        if save_svg_plot1b:
            fig1b.write_image(
                f"plot1/plot1b_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sorts_from_each_path_to_load[iSort]}_{sort_types}.svg",
                width=figsize[0],
                height=figsize[1],
            )
        if save_html_plot1b:
            fig1b.write_html(
                f"plot1/plot1b_KS_vs_GT_performance_metrics_{bin_width_for_comparison}ms_{sorts_from_each_path_to_load[iSort]}_{sort_types}.html",
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
    sort_index=0,
):
    # search all the paths to sorted folders for the KS or EMUsort string, in the order of matches with sorts_from_each_path_to_load
    sort_types = [None] * len(sorts_from_each_path_to_load)
    for iSort in range(len(sorts_from_each_path_to_load)):
        for iPath in range(len(list_of_paths_to_sorted_folders[0])):
            if (
                sorts_from_each_path_to_load[iSort]
                in list_of_paths_to_sorted_folders[0][iPath].name
            ):
                suffix = list_of_paths_to_sorted_folders[0][iPath].name.split("_")[-1]
            elif (
                sorts_from_each_path_to_load[iSort]
                in list_of_paths_to_sorted_folders[0][iPath].parent.name
            ):
                suffix = list_of_paths_to_sorted_folders[0][iPath].parent.name.split(
                    "_"
                )[-1]
            else:
                continue  # keep searching
            if suffix == "KS":
                sort_types[iSort] = "Kilosort3"
            elif suffix == "KS4":
                sort_types[iSort] = "Kilosort4"
            else:
                sort_types[iSort] = "EMUsort"
            break  # found a match with the datestring, label and break out of the loop
    # make sure all sort_types were found
    assert None not in sort_types, "Not all sort_types were found"
    sort_type = sort_types[sort_index]
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
    left_bound = int(round(plot2_xlim[0] * kilosort_spikes.shape[0]))
    right_bound = int(round(plot2_xlim[1] * kilosort_spikes.shape[0]))
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
        range=np.array([left_bound / 1000, right_bound / 1000])
        * bin_width_for_comparison,
    )
    # remove y axes numbers from all plots
    fig.update_yaxes(showticklabels=False)

    # append sort name instead of time stamp
    if save_png_plot2:
        fig.write_image(
            f"plot2/plot2_KS_vs_GT_spike_trains_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{sort_type}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot2:
        fig.write_image(
            f"plot2/plot2_KS_vs_GT_spike_trains_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{sort_type}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot2:
        fig.write_html(
            f"plot2/plot2_KS_vs_GT_spike_trains_{bin_width_for_comparison}ms_{sort_from_each_path_to_load}_{sort_type}.html",
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
    PPP_branch_name = list_of_paths_to_sorted_folders[0][iSort].name.split("_")[-1]
    sort_type = "Kilosort" if "KS" in PPP_branch_name else "EMUsort"
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
            range=[0, 1.1],
        )

    fig.update_yaxes(matches="y")

    fig.update_xaxes(
        title_text="<b>Bin Width (ms)</b>",
        row=3,
        col=1,
        # range=[0, 8],
    )

    # get suffix after the KS folder name, which is the repo branch name for that sort
    PPP_branch_name = list_of_paths_to_sorted_folders[0][iSort].name.split("_")[-1]

    if save_png_plot3:
        fig.write_image(
            f"plot3/plot3_KS_vs_GT_bin_width_comparison_{sort_from_each_path_to_load}_{PPP_branch_name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot3:
        fig.write_image(
            f"plot3/plot3_KS_vs_GT_bin_width_comparison_{sort_from_each_path_to_load}_{PPP_branch_name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot3:
        fig.write_html(
            f"plot3/plot3_KS_vs_GT_bin_width_comparison_{sort_from_each_path_to_load}_{PPP_branch_name}.html",
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
    num_motor_units,
    clusters_in_sort_to_use,
    GT_clusters_to_use,
    sorts_from_each_path_to_load,  # list of strings
    plot_template,
    show_plot4,
    save_png_plot4,
    save_svg_plot4,
    save_html_plot4,
    figsize=(1920, 1080),
):
    # this function will plot the precision, recall, and accuracy for each sort. The precision, recall,
    # and accuracy are lists, each from a different sort, and the mean and standard deviation of
    # each metric will be plotted for each motor unit in each sort. The x axis will be the motor
    # units, and the y axis will be the metric score. The mean will be plotted as a line, and the
    # standard deviation will be plotted as a +/- error bars around the line. The line and +/- error bars
    # will be colored according to the sort. There is a subplot for each metric and the different sorts
    # will overlie eachother in the same subplot. It is like plot3, but with the x-axis being the motor
    # units, and replacing multiple lines with a single line and +/- error bars for each metric.
    # The plot will be optionally be saved as a png, svg, and html file, and optionally shown in a
    # browser window.
    # pass
    # get suffix after the KS folder name, which is the repo branch name for that sort
    # PPP_branch_names = [
    #     list_of_paths_to_sorted_folders[0][iSort].name.split("_")[-1]
    #     for iSort in range(len(list_of_paths_to_sorted_folders[0]))
    # ]
    # sort_types = [
    #     "Kilosort" if PPP_branch_names[iSort] == "KS" else "EMUsort"
    #     for iSort in range(len(list_of_paths_to_sorted_folders[0]))
    # ]

    # search all the paths to sorted folders for the KS or EMUsort string, in the order of matches with sorts_from_each_path_to_load
    sort_types = [None] * len(sorts_from_each_path_to_load)
    noise_levels = [None] * len(sorts_from_each_path_to_load)
    for iSort in range(len(sorts_from_each_path_to_load)):
        for iPath in range(len(list_of_paths_to_sorted_folders[0])):
            if (
                sorts_from_each_path_to_load[iSort]
                in list_of_paths_to_sorted_folders[0][iPath].name
            ):
                suffix = list_of_paths_to_sorted_folders[0][iPath].name.split("_")[-1]
            elif (
                sorts_from_each_path_to_load[iSort]
                in list_of_paths_to_sorted_folders[0][iPath].parent.name
            ):
                suffix = list_of_paths_to_sorted_folders[0][iPath].parent.name.split(
                    "_"
                )[-1]
            else:
                continue  # keep searching
            if suffix == "KS":
                sort_types[iSort] = "Kilosort3"
            elif suffix == "KS4":
                sort_types[iSort] = "Kilosort4"
            else:
                sort_types[iSort] = "EMUsort"
            # for each match, extract the noise level from the path, which will be revealed by splitting the parent path by underscores and taking the last index
            # will be a number of format x.xx, such as 3.00, and convert to string
            try:
                noise_levels[iSort] = [
                    np.round(float(noise_lvl), 2)
                    for noise_lvl in list_of_paths_to_sorted_folders[0][
                        iPath
                    ].parent.parent.name.split("_")
                    if "." in noise_lvl
                ][0]
            except:
                noise_levels[iSort] = [
                    np.round(float(noise_lvl), 2)
                    for noise_lvl in list_of_paths_to_sorted_folders[0][
                        iPath
                    ].parent.name.split("_")
                    if "." in noise_lvl
                ][0]

            break  # found a match with the datestring, label and break out of the loop
    # make sure all sort_types were found
    assert None not in sort_types, "Not all sort_types were found"
    # make sure all noise levels were the same and not None
    assert (None not in noise_levels) and (
        len(set(noise_levels)) == 1
    ), "Noise levels did not match across folders"
    if show_plot4 or save_png_plot4 or save_svg_plot4 or save_html_plot4:
        # make a subplot for each metric
        fig = subplots.make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["Precision", "Recall", "Accuracy"],
        )

        # interpolate within darker half the color map to get as many colors as there are motor units
        N_colors = 10
        precision_color_map = cl.interp(cl.scales["9"]["seq"]["Greens"][2:9], N_colors)
        recall_color_map = cl.interp(cl.scales["9"]["seq"]["Oranges"][2:9], N_colors)
        accuracy_color_map = cl.interp(cl.scales["9"]["seq"]["Blues"][2:9], N_colors)

        metric_color_maps = [
            precision_color_map,
            recall_color_map,
            accuracy_color_map,
        ]

        # collapse list of each metric into a 3D numpy array, then take the mean and standard deviation
        # of each metric across the newly created axis, then plot the mean and standard deviation for each
        # motor unit in each metric
        precision = np.array(precision)
        recall = np.array(recall)
        accuracy = np.array(accuracy)

        # place each metric into a pandas dataframe, sorted by the accuracy. Include a column what type of sort it is
        # then sort the dataframe by the accuracy, grouped by the type of sort
        # also include the sort datestring as a df column
        metrics_df = df(
            {
                "precision": precision.mean(axis=1),
                "recall": recall.mean(axis=1),
                "accuracy": accuracy.mean(axis=1),
                "sort_type": sort_types,
                "datestring": sorts_from_each_path_to_load,
                "noise_level": noise_levels,
            }
        )
        metrics_df = metrics_df.sort_values(
            by=["accuracy", "sort_type"], ascending=[False, True]
        )
        print(metrics_df)

        metric_values = [precision, recall, accuracy]
        # color by what type of sort it is, make it darker if it is Kilosort
        for iMetric in range(len(metric_values)):

            for sort_type in set(sort_types):
                if sort_type == "EMUsort":
                    # take averages across the sort axis (axis 0), but only for the elements that match
                    # the sort type
                    metric_means_EMU = np.mean(
                        metric_values[iMetric][
                            np.where([int(i == "EMUsort") for i in sort_types])
                        ],
                        axis=0,
                    )
                    metric_stds_EMU = np.std(
                        metric_values[iMetric][
                            np.where([int(i == "EMUsort") for i in sort_types])
                        ],
                        axis=0,
                    )
                    # # add the standard deviation as shaded region around the mean
                    # # make opacity of the fill 0.6
                    # fig.add_trace(
                    #     go.Scatter(
                    #         x=list(range(num_motor_units))
                    #         + list(range(num_motor_units)[::-1]),
                    #         y=np.concatenate(
                    #             [
                    #                 metric_means_EMU + metric_stds_EMU,
                    #                 metric_means_EMU[::-1] - metric_stds_EMU[::-1],
                    #             ]
                    #         ),
                    #         fill="toself",
                    #         mode="lines",
                    #         fillcolor=(metric_color_maps[iMetric][0]),
                    #         line=dict(width=0),
                    #         showlegend=False,
                    #         opacity=0.6,
                    #     ),
                    #     row=iMetric + 1,
                    #     col=1,
                    # )

                    # add the mean as a line
                    # and add the standard deviation as a +/- error bars around the mean
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(num_motor_units)),
                            y=metric_means_EMU,
                            mode="markers+lines",
                            name=sort_type,
                            marker=dict(
                                color=(
                                    metric_color_maps[iMetric][-1]
                                ),  # make EMUsort darkest color
                                size=8,
                            ),
                            line=dict(
                                width=6,
                                color=(metric_color_maps[iMetric][-1]),
                            ),
                            # add the standard deviation as a +/- error bars around the mean
                            error_y=dict(
                                type="data",
                                array=metric_stds_EMU,
                                visible=True,
                                color=(metric_color_maps[iMetric][-1]),
                                thickness=4,
                                width=4,
                            ),
                            opacity=1,
                        ),
                        row=iMetric + 1,
                        col=1,
                    )
                elif "Kilosort" in sort_type:
                    if sort_type == "Kilosort3":
                        color_idx = 0
                    elif sort_type == "Kilosort4":  # make Kilosort4 slightly darker
                        color_idx = len(accuracy_color_map) // 2

                    # add the standard deviation as a +/- error bars around the mean
                    # make opacity of the fill 0.6
                    metric_means_KS = np.mean(
                        metric_values[iMetric][
                            np.where([int(i == sort_type) for i in sort_types])
                        ],
                        axis=0,
                    )
                    metric_stds_KS = np.std(
                        metric_values[iMetric][
                            np.where([int(i == sort_type) for i in sort_types])
                        ],
                        axis=0,
                    )
                    # fig.add_trace(
                    #     go.Scatter(
                    #         x=list(range(num_motor_units))
                    #         + list(range(num_motor_units)[::-1]),
                    #         y=np.concatenate(
                    #             [
                    #                 metric_means_KS + metric_stds_KS,
                    #                 metric_means_KS[::-1] - metric_stds_KS[::-1],
                    #             ]
                    #         ),
                    #         fill="toself",
                    #         mode="lines",
                    #         fillcolor=(metric_color_maps[iMetric][-2]),
                    #         line=dict(width=0),
                    #         showlegend=False,
                    #         opacity=0.6,
                    #     ),
                    #     row=iMetric + 1,
                    #     col=1,
                    # )
                    # add the mean as a line
                    # and add the standard deviation as a +/- error bars around the mean
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(num_motor_units)),
                            y=metric_means_KS,
                            mode="markers+lines",
                            name=sort_type,
                            marker=dict(
                                # color="black",
                                color=(metric_color_maps[iMetric][color_idx]),
                                size=8,
                            ),
                            line=dict(
                                width=6,
                                color=(metric_color_maps[iMetric][color_idx]),
                            ),
                            # add the standard deviation as a +/- error bars around the mean
                            error_y=dict(
                                type="data",
                                array=metric_stds_KS,
                                visible=True,
                                color=(metric_color_maps[iMetric][color_idx]),
                                thickness=4,
                                width=4,
                            ),
                            opacity=1,
                        ),
                        row=iMetric + 1,
                        col=1,
                    )
        unique_sort_types = list(set(sort_types))
        num_each_sort_type = [sort_types.count(i) for i in unique_sort_types]
        fig.update_layout(
            title=f"<b>Performance of {', '.join(unique_sort_types)} Across All {', '.join([str(i) for i in num_each_sort_type])} Sorts, Noise Level: {noise_levels[0]}</b>",
            legend_title="Means +/- 1 Standard Deviation",
            template=plot_template,
            yaxis=dict(title="<b>Metric Score</b>", range=[0, 1.1]),
        )

        # make sure each row has the same y axis range
        for iRow in range(3):
            fig.update_yaxes(
                title_text="<b>Metric Score</b>",
                row=iRow + 1,
                col=1,
                range=[0, 1.1],
            )

        fig.update_yaxes(matches="y")

        fig.update_xaxes(
            title_text="<b>GT Cluster ID</b>",
            row=3,
            col=1,
        )

        if save_png_plot4:
            fig.write_image(  # add datestr
                f"plot4/plot4_KS_vs_GT_performance_comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{','.join(unique_sort_types)}_{','.join([str(i) for i in num_each_sort_type])}.png",
                width=figsize[0],
                height=figsize[1],
            )
        if save_svg_plot4:
            fig.write_image(
                f"plot4/plot4_KS_vs_GT_performance_comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{','.join(unique_sort_types)}_{','.join([str(i) for i in num_each_sort_type])}.svg",
                width=figsize[0],
                height=figsize[1],
            )
        if save_html_plot4:
            fig.write_html(
                f"plot4/plot4_KS_vs_GT_performance_comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{','.join(unique_sort_types)}_{','.join([str(i) for i in num_each_sort_type])}.html",
                include_plotlyjs="cdn",
                full_html=False,
            )
        if save_plot4_df_as_pickle:
            metrics_df.to_pickle(
                f"plot4/plot4_KS_vs_GT_performance_comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{','.join(unique_sort_types)}_{','.join([str(i) for i in num_each_sort_type])}.pkl"
            )
        if save_plot4_df_as_csv:
            metrics_df.to_csv(
                f"plot4/plot4_KS_vs_GT_performance_comparison_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{','.join(unique_sort_types)}_{','.join([str(i) for i in num_each_sort_type])}.csv"
            )
        if show_plot4:
            try:
                fig.show()
            except Exception as e:
                print(e)  # don't crash if the plot can't be shown, but print the error

        # print the average and std of the accuracy for the dataframes, filtering for each unique sort type
        print("\n")
        print(
            metrics_df.groupby("sort_type")
            .agg({"accuracy": ["mean", "std"]})
            .sort_values(by=("accuracy", "mean"), ascending=False)
        )


def plot5(
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
    parallel = True  # can set to True, False, and "duo", which executes 2 processes at a time, with a clever ordering of pairs to keep memory consumption down
    use_custom_merge_clusters = False
    from_spike_interface = True
    automatically_assign_cluster_mapping = True
    method_for_automatic_cluster_mapping = "accuracies"  # can be "accuracies", "waves", "times", or "trains"  what the correlation is computed on to map clusters
    simulation_method = "MUsim"  # can be "MUsim" or "konstantin"
    time_frame = [0, 1]  # must be between 0 and 1
    ephys_fs = 30000  # Hz
    xstart = np.log2(
        0.125
    )  # choose bin widths as a range from 0.125 ms to 8 ms in log2 increments
    bin_widths_for_comparison = np.logspace(xstart, -xstart, num=13, base=2)
    bin_widths_for_comparison = [0.1]
    spike_isolation_radius_ms = None  # radius of isolation of a spike for it to be removed from consideration. set to positive float, integer, or set None to disable
    iShow = 0  # index of which bin width of bin_widths_for_comparison to show in plots

    nt0 = 121  # number of time bins in the template, in ms it is 3.367, only used if method_for_automatic_cluster_mapping is "waves"
    random_seed_entropy = 218530072159092100005306709809425040261  # 75092699954400878964964014863999053929  # int
    plot_template = "plotly_white"  # ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
    plot1_bar_type = "percent"  # totals / percent
    plot1_ylim = [0, 135]
    plot2_xlim = [0, 0.006]
    show_plot1a = False
    show_plot1b = False
    show_plot2 = True
    show_plot3 = False
    show_plot4 = True
    show_plot5 = False
    save_png_plot1a = False
    save_png_plot1b = False
    save_png_plot2 = False
    save_png_plot3 = False
    save_png_plot4 = False  #
    save_png_plot5 = False
    save_svg_plot1a = False
    save_svg_plot1b = False
    save_svg_plot2 = False
    save_svg_plot3 = False
    save_svg_plot4 = False  #
    save_svg_plot5 = False
    save_html_plot1a = False
    save_html_plot1b = False
    save_html_plot2 = False
    save_html_plot3 = False
    save_html_plot4 = False
    save_html_plot5 = False
    save_plot4_df_as_pickle = False  #
    save_plot4_df_as_csv = False  #

    ## TBD: NEED TO ADD FLAG FOR DATASET CHOICE, to flip all related variables
    ## paths with simulated data
    path_to_sim_dat = Path(
        # "continuous_20221117_godzilla_SNR-100-constant_jitter-0std_files-14_20240206-160607.dat"  # monkey
        # "continuous_20221117_godzilla_SNR-400-constant_jitter-0std_files-11.dat"  # triple rat
        # "continuous_20221117_godzilla_SNR-None-constant_jitter-0std_files-11.dat"  # godzilla only, old
        # "continuous_20240217-185655_godzilla_20221117_10MU_SNR-None-constant_jitter-0std_method-median_waves_12-files.dat"  # godzilla only, None
        # "continuous_20221117_godzilla_SNR-1-from_data_jitter-4std_files-11.dat"
    )
    ## load ground truth data
    ground_truth_path = Path(
        # "spikes_20221117_godzilla_SNR-100-constant_jitter-0std_files-14_20240206-160539.npy"  # monkey
        # "spikes_20221117_godzilla_SNR-400-constant_jitter-0std_files-11.npy"  # triple rat
        # "spikes_20221117_godzilla_SNR-1-from_data_jitter-4std_files-11.npy"  # godzilla only, old
        # "spikes_20240217-185626_godzilla_20221117_10MU_SNR-None-constant_jitter-0std_method-median_waves_12-files.npy"  # godzilla only, None
        # "spikes_20240217-185448_godzilla_20221117_10MU_SNR-100-constant_jitter-0std_method-median_waves_12-files.npy"  # godzilla only, 100
        # "spikes_20240217-185509_godzilla_20221117_10MU_SNR-200-constant_jitter-0std_method-median_waves_12-files.npy"  # godzilla only, 200
        # "spikes_20240217-185512_godzilla_20221117_10MU_SNR-300-constant_jitter-0std_method-median_waves_12-files.npy"  # godzilla only, 300
        # "spikes_20240217-185528_godzilla_20221117_10MU_SNR-400-constant_jitter-0std_method-median_waves_12-files.npy"  # godzilla only, 400
        ## >= 20240220, godzilla
        #
        # "spikes_20240221-185920_godzilla_20221117_10MU_SNR-None-constant_jitter-0std_method-median_waves_12-files.npy"
        # "spikes_20240220-213114_godzilla_20221117_10MU_SNR-100-constant_jitter-0std_method-median_waves_12-files.npy"
        # "spikes_20240220-213122_godzilla_20221117_10MU_SNR-400-constant_jitter-0std_method-median_waves_12-files.npy"
        # "spikes_20240221-132651_godzilla_20221117_10MU_SNR-700-constant_jitter-0std_method-median_waves_12-files.npy"
        # "spikes_20240220-213138_godzilla_20221117_10MU_SNR-1000-constant_jitter-0std_method-median_waves_12-files.npy"
        ## >= 20240301, godzilla shape noise
        # "spikes_20240229-200231_godzilla_20221117_10MU_SNR-1-from_data_jitter-0std_method-KS_templates_12-files.npy"
        ##
        # "spikes_20240217-221958_monkey_20221202_6MU_SNR-None-constant_jitter-0std_method-median_waves_1-files.npy" # monkey, None
        # "spikes_20240217-221838_monkey_20221202_6MU_SNR-100-constant_jitter-0std_method-median_waves_1-files.npy" # monkey, 100
        # "spikes_20240217-221902_monkey_20221202_6MU_SNR-200-constant_jitter-0std_method-median_waves_1-files.npy" # monkey, 200
        # "spikes_20240217-221916_monkey_20221202_6MU_SNR-300-constant_jitter-0std_method-median_waves_1-files.npy" # monkey, 300
        # "spikes_20240217-221932_monkey_20221202_6MU_SNR-400-constant_jitter-0std_method-median_waves_1-files.npy" # monkey, 400
        # "spikes_20221117_godzilla_SNR-1-from_data_jitter-1std_files-5.npy"
        # "spikes_20221116_godzilla_SNR-8-from_data_jitter-4std_files-1.npy"
        ## for konstantin simulated data, use the path to the simulation folder
        # "/home/smoconn/git/iemg_simulator/simulation_output/vector_100%MVC_600sec_17p_array_18_MUs_2/"
        # "/home/smoconn/git/iemg_simulator/simulation_output/vector_100%MVC_600sec_17p_array_18_MUs_4/"
        # "/home/smoconn/git/iemg_simulator/simulation_output/vector_100%MVC_600sec_17p_array_9_MUs_1/"
        ## >= 20240607, godzilla
        "spikes_20240607-143039_godzilla_20221117_10MU_SNR-1-from_data_jitter-0std_method-KS_templates_12-files.npy"
        ## >= 20240731, monkey
        # "spikes_20240726-204919_monkey_20221202_6MU_SNR-1-from_data_jitter-0std_method-KS_templates_1-files.npy"
    )  # spikes_20221116_godzilla_SNR-None_jitter-0std_files-1.npy
    ground_truth_path = Path().joinpath("spikes_files", ground_truth_path)
    if ".npy" not in ground_truth_path.name:
        assert (
            simulation_method == "konstantin"
        ), "simulation_method must be 'konstantin' if ground_truth_path is not an .npy file"
    # set which ground truth clusters to compare with (a range from 0 to num_motor_units)
    num_motor_units = 6 if "monkey" in ground_truth_path.name else 10
    GT_clusters_to_use = list(range(0, num_motor_units))

    ## load Kilosort data
    # paths to the folders containing the Kilosort data
    paths_to_KS_session_folders = [
        Path(
            # "/snel/share/data/rodent-ephys/open-ephys/monkey/sean-pipeline/simulated20240206"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/triple/simulated20231219/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221116/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/simulated20221117/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/konstantin/simulated20240307/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_0.00/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_0.75/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_1.50/"
            "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_2.25/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_3.00/"
            # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_3.75/"
            # "/snel/share/data/rodent-ephys/open-ephys/monkey/paper_evals/sim_2022-12-02_10-14-45_shape_noise_0.00"
            # "/snel/share/data/rodent-ephys/open-ephys/monkey/paper_evals/sim_2022-12-02_10-14-45_shape_noise_0.75"
            # "/snel/share/data/rodent-ephys/open-ephys/monkey/paper_evals/sim_2022-12-02_10-14-45_shape_noise_1.50"
            # "/snel/share/data/rodent-ephys/open-ephys/monkey/paper_evals/sim_2022-12-02_10-14-45_shape_noise_2.25"
            # "/snel/share/data/rodent-ephys/open-ephys/monkey/paper_evals/sim_2022-12-02_10-14-45_shape_noise_3.00"
            # "/snel/share/data/rodent-ephys/open-ephys/monkey/paper_evals/sim_2022-12-02_10-14-45_shape_noise_3.75"
        ),
    ]
    sorts_from_each_path_to_load = [
        #### below are godzilla only dataset
        ## simulated20221116:
        # {
        # "20231011_185107"  # 1 std, 4 jitter
        # "20231011_195053"  # 2 std, 4 jitter
        # "20231011_201450"  # 4 std, 4 jitter
        # "20231011_202716"  # 8 std, 4 jitter
        # } All in braces did not have channel delays reintroduced for continuous.dat
        ### simulated20221117:
        ## old godzilla only dataset
        # {
        # "20231027_183121"  # 1 std, 4 jitter, all MUsort options ON
        # "20231031_141254"  # 1 std, 4 jitter, all MUsort options ON, slightly better
        # "20231103_160031096827"  # 1 std, 4 jitter, all MUsort options ON, ?
        # "20231103_175840215876",  # 2 std, 8 jitter, all MUsort options ON, ?
        # "20231103_164647242198",  # 2 std, 4 jitter, all MUsort options ON, custom_merge
        # "20231105_192242190872",  # 2 std, 8 jitter, all MUsort options ON, except multi-threshold $$$ BEST EMUsort $$$
        # "20231101_165306036638"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[1,0.5], spkTh=[-6]
        # "20231101_164409249821"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[1,0.5], spkTh=[-2]
        # "20231101_164937098773"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[5,2], spkTh=[-6]
        # "20231101_164129797219"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[5,2], spkTh=[-2]
        # "20231101_165135058289"  # 1 std, 4 jitter, optimal template selection routines OFF, Th=[2,1], spkTh=[-6]
        # "20231102_175449741223",  # 1 std, 4 jitter, vanilla Kilosort, Th=[1,0.5], spkTh=[-6]
        # "20231103_184523634126",  # 2 std, 8 jitter, vanilla Kilosort, Th=[1,0.5], spkTh=[-6] $$$ BEST Kilosort3 $$$
        # "20231103_184518491799",  # 2 std, 8 jitter, vanilla Kilosort, Th=[2,1], spkTh=[-6]
        # } All in braces did not have channel delays reintroduced for continuous.dat
        ## new godzilla only dataset
        ## EMUsort with comparable grid search, 100 noise, with sgolay filter to align templates (makes performance worse)
        # "20240213_141533069348",  # rec-1,2,4,5,6,7_16-good-of-27-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240213_141547394314",  # rec-1,2,4,5,6,7_23-good-of-40-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240213_141554453333",  # rec-1,2,4,5,6,7_17-good-of-22-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240213_141753946101",  # rec-1,2,4,5,6,7_19-good-of-41-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240213_141814833755",  # rec-1,2,4,5,6,7_25-good-of-45-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240213_141947433895",  # rec-1,2,4,5,6,7_42-good-of-69-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240213_142408090765",  # rec-1,2,4,5,6,7_30-good-of-50-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240213_142552661935",  # rec-1,2,4,5,6,7_23-good-of-39-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240213_142638222266",  # rec-1,2,4,5,6,7_26-good-of-51-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240213_142639968742",  # rec-1,2,4,5,6,7_22-good-of-36-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240213_142832895594",  # rec-1,2,4,5,6,7_17-good-of-36-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240213_142848927359",  # rec-1,2,4,5,6,7_31-good-of-52-total_Th,[2,1],spkTh,[-3]_EMUsort
        # ## Kilosort with comparable grid search, 100 noise
        # "20240213_161545004994",  # rec-1,2,4,5,6,7_19-good-of-29-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240213_161559454281",  # rec-1,2,4,5,6,7_30-good-of-45-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240213_161630322120",  # rec-1,2,4,5,6,7_31-good-of-48-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240213_161634687392",  # rec-1,2,4,5,6,7_28-good-of-48-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240213_161641182492",  # rec-1,2,4,5,6,7_39-good-of-68-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240213_161655262229",  # rec-1,2,4,5,6,7_21-good-of-37-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240213_161812382797",  # rec-1,2,4,5,6,7_24-good-of-42-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240213_161846771684",  # rec-1,2,4,5,6,7_33-good-of-65-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240213_161855353121",  # rec-1,2,4,5,6,7_19-good-of-38-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240213_161914859946",  # rec-1,2,4,5,6,7_25-good-of-45-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240213_161951960686",  # rec-1,2,4,5,6,7_12-good-of-23-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240213_162009642022",  # rec-1,2,4,5,6,7_30-good-of-61-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240216_162422867565",  # rec-1,2,4,5,6,7_27-good-of-37-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240216_162446136368",  # rec-1,2,4,5,6,7_23-good-of-42-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240216_162452424118",  # rec-1,2,4,5,6,7_32-good-of-58-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240216_162502422922",  # rec-1,2,4,5,6,7_28-good-of-50-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240216_162508881201",  # rec-1,2,4,5,6,7_22-good-of-34-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240216_162517261656",  # rec-1,2,4,5,6,7_31-good-of-54-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240216_162611369721",  # rec-1,2,4,5,6,7_19-good-of-36-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240216_162658944644",  # rec-1,2,4,5,6,7_25-good-of-64-total_Th,[7,3],spkTh,-9_vanilla_KS
        # ## EMUsort with extended grid search, 100 noise, with sgolay filter to align templates (makes performance worse)
        # "20240214_141128826351",  # rec-1,2,4,5,6,7_17-good-of-26-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240214_141143952219",  # rec-1,2,4,5,6,7_37-good-of-47-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240214_141226394409",  # rec-1,2,4,5,6,7_30-good-of-48-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240214_141343991827",  # rec-1,2,4,5,6,7_23-good-of-34-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240214_141436664193",  # rec-1,2,4,5,6,7_30-good-of-49-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240214_141629655492",  # rec-1,2,4,5,6,7_38-good-of-55-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240214_141714642733",  # rec-1,2,4,5,6,7_19-good-of-35-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240214_141958119630",  # rec-1,2,4,5,6,7_30-good-of-50-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240214_142042252641",  # rec-1,2,4,5,6,7_41-good-of-66-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240214_142253322841",  # rec-1,2,4,5,6,7_19-good-of-39-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240214_142427876250",  # rec-1,2,4,5,6,7_17-good-of-28-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240214_142530301845",  # rec-1,2,4,5,6,7_12-good-of-16-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240214_142540092161",  # rec-1,2,4,5,6,7_39-good-of-58-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240214_142719084690",  # rec-1,2,4,5,6,7_25-good-of-40-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240214_142844937405",  # rec-1,2,4,5,6,7_24-good-of-48-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240214_143003233619",  # rec-1,2,4,5,6,7_25-good-of-47-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240214_143154577938",  # rec-1,2,4,5,6,7_28-good-of-40-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240214_143322507361",  # rec-1,2,4,5,6,7_20-good-of-35-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240214_143401799840",  # rec-1,2,4,5,6,7_22-good-of-32-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240214_143429236321",  # rec-1,2,4,5,6,7_15-good-of-20-total_Th,[2,1],spkTh,[-9]_EMUsort
        # # ## EMUsort with extended grid search, 100 noise, without sgolay filter
        # "20240216_153034858233",  # rec-1,2,4,5,6,7_20-good-of-29-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240216_153046455460",  # rec-1,2,4,5,6,7_20-good-of-32-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240216_153101516752",  # rec-1,2,4,5,6,7_19-good-of-35-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240216_153221873441",  # rec-1,2,4,5,6,7_15-good-of-25-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240216_153347156477",  # rec-1,2,4,5,6,7_29-good-of-57-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240216_153358841926",  # rec-1,2,4,5,6,7_19-good-of-24-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240216_153421051618",  # rec-1,2,4,5,6,7_34-good-of-62-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240216_153710315338",  # rec-1,2,4,5,6,7_36-good-of-49-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240216_153743993841",  # rec-1,2,4,5,6,7_31-good-of-62-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240216_153748662905",  # rec-1,2,4,5,6,7_16-good-of-30-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240216_153759267127",  # rec-1,2,4,5,6,7_14-good-of-28-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240216_154300637843",  # rec-1,2,4,5,6,7_16-good-of-30-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240216_154320725543",  # rec-1,2,4,5,6,7_28-good-of-41-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240216_154412428553",  # rec-1,2,4,5,6,7_14-good-of-19-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240216_154501178414",  # rec-1,2,4,5,6,7_16-good-of-36-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240216_154702028222",  # rec-1,2,4,5,6,7_25-good-of-53-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240216_155025099310",  # rec-1,2,4,5,6,7_11-good-of-22-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240216_160837259935",  # rec-1,2,4,5,6,7_19-good-of-29-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240216_160909021342",  # rec-1,2,4,5,6,7_21-good-of-39-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240216_160946385912",  # rec-1,2,4,5,6,7_36-good-of-59-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # ## Kilosort with comparable grid search, 200 noise
        # "20240214_153111841845",  # rec-1,2,4,5,6,7_25-good-of-42-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240214_153112520369",  # rec-1,2,4,5,6,7_18-good-of-27-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240214_153144013764",  # rec-1,2,4,5,6,7_31-good-of-54-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240214_153148118479",  # rec-1,2,4,5,6,7_25-good-of-53-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240214_153149495041",  # rec-1,2,4,5,6,7_30-good-of-46-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240214_153316380029",  # rec-1,2,4,5,6,7_18-good-of-31-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240214_153334665438",  # rec-1,2,4,5,6,7_12-good-of-17-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240214_153347273344",  # rec-1,2,4,5,6,7_31-good-of-57-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240214_153401298373",  # rec-1,2,4,5,6,7_27-good-of-42-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240214_153407203405",  # rec-1,2,4,5,6,7_34-good-of-53-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240214_153538178636",  # rec-1,2,4,5,6,7_17-good-of-26-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240214_153806432134",  # rec-1,2,4,5,6,7_13-good-of-22-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240216_183341479604",  # rec-1,2,4,5,6,7_17-good-of-22-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240216_183348853632",  # rec-1,2,4,5,6,7_21-good-of-42-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240216_183403044421",  # rec-1,2,4,5,6,7_20-good-of-31-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240216_183406807925",  # rec-1,2,4,5,6,7_26-good-of-45-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240216_183414027038",  # rec-1,2,4,5,6,7_26-good-of-46-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240216_183423910604",  # rec-1,2,4,5,6,7_38-good-of-60-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240216_183537605869",  # rec-1,2,4,5,6,7_6-good-of-8-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240216_183620332576",  # rec-1,2,4,5,6,7_21-good-of-33-total_Th,[2,1],spkTh,-9_vanilla_KS
        # # ## EMUsort with extended grid search, 200 noise, with sgolay filter to align templates (makes performance worse)
        # "20240214_144739887594",  # rec-1,2,4,5,6,7_18-good-of-30-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240214_144811189869",  # rec-1,2,4,5,6,7_21-good-of-31-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240214_144830531365",  # rec-1,2,4,5,6,7_16-good-of-26-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240214_144957506188",  # rec-1,2,4,5,6,7_16-good-of-24-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240214_145059496632",  # rec-1,2,4,5,6,7_13-good-of-24-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240214_145318874249",  # rec-1,2,4,5,6,7_25-good-of-36-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240214_145522004178",  # rec-1,2,4,5,6,7_20-good-of-34-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240214_145525495173",  # rec-1,2,4,5,6,7_14-good-of-22-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240214_145839623420",  # rec-1,2,4,5,6,7_19-good-of-27-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240214_145916013874",  # rec-1,2,4,5,6,7_14-good-of-18-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240214_145917650574",  # rec-1,2,4,5,6,7_27-good-of-46-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240214_150023501108",  # rec-1,2,4,5,6,7_12-good-of-18-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240214_150042946812",  # rec-1,2,4,5,6,7_16-good-of-21-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240214_150237730878",  # rec-1,2,4,5,6,7_21-good-of-33-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240214_150339423378",  # rec-1,2,4,5,6,7_13-good-of-22-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240214_150736638289",  # rec-1,2,4,5,6,7_32-good-of-44-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240214_150801332028",  # rec-1,2,4,5,6,7_23-good-of-40-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240214_151336958967",  # rec-1,2,4,5,6,7_11-good-of-14-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240214_151825607477",  # rec-1,2,4,5,6,7_15-good-of-24-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240214_152603425570",  # rec-1,2,4,5,6,7_12-good-of-15-total_Th,[2,1],spkTh,[-9]_EMUsort
        # ## EMUsort with extended grid search, 200 noise, without sgolay filter
        # "20240215_180243669553",  # rec-1,2,4,5,6,7_13-good-of-16-total_Th,[10,4],spkTh,[-3]
        # "20240215_180354552562",  # rec-1,2,4,5,6,7_18-good-of-28-total_Th,[10,4],spkTh,[-3,-6]
        # "20240215_180359238313",  # rec-1,2,4,5,6,7_19-good-of-28-total_Th,[7,3],spkTh,[-6,-9]
        # "20240215_180418024062",  # rec-1,2,4,5,6,7_24-good-of-37-total_Th,[7,3],spkTh,[-6]
        # "20240215_180503541101",  # rec-1,2,4,5,6,7_16-good-of-27-total_Th,[5,2],spkTh,[-9]
        # "20240215_180742023512",  # rec-1,2,4,5,6,7_33-good-of-46-total_Th,[10,4],spkTh,[-6]
        # "20240215_180847757030",  # rec-1,2,4,5,6,7_26-good-of-39-total_Th,[10,4],spkTh,[-6,-9]
        # "20240215_181021821778",  # rec-1,2,4,5,6,7_10-good-of-14-total_Th,[2,1],spkTh,[-3]
        # "20240215_181108819246",  # rec-1,2,4,5,6,7_22-good-of-32-total_Th,[5,2],spkTh,[-3]
        # "20240215_181235658941",  # rec-1,2,4,5,6,7_13-good-of-19-total_Th,[5,2],spkTh,[-3,-6]
        # "20240215_181238866957",  # rec-1,2,4,5,6,7_29-good-of-39-total_Th,[10,4],spkTh,[-9]
        # "20240215_181435714131",  # rec-1,2,4,5,6,7_17-good-of-23-total_Th,[7,3],spkTh,[-3]
        # "20240215_181608323357",  # rec-1,2,4,5,6,7_11-good-of-18-total_Th,[2,1],spkTh,[-3,-6]
        # "20240215_181907826058",  # rec-1,2,4,5,6,7_19-good-of-24-total_Th,[5,2],spkTh,[-6,-9]
        # "20240215_181912792960",  # rec-1,2,4,5,6,7_23-good-of-29-total_Th,[5,2],spkTh,[-6]
        # "20240215_182702857918",  # rec-1,2,4,5,6,7_22-good-of-32-total_Th,[2,1],spkTh,[-6]
        # "20240215_183102702036",  # rec-1,2,4,5,6,7_10-good-of-17-total_Th,[2,1],spkTh,[-6,-9]
        # "20240215_184111732692",  # rec-1,2,4,5,6,7_14-good-of-21-total_Th,[2,1],spkTh,[-9]
        # "20240215_185546826451",  # rec-1,2,4,5,6,7_30-good-of-47-total_Th,[7,3],spkTh,[-9]
        # "20240215_185550066478",  # rec-1,2,4,5,6,7_26-good-of-38-total_Th,[7,3],spkTh,[-3,-6]
        # Kilosort with comparable grid search, 300 noise
        # "20240216_145809891097",  # rec-1,2,4,5,6,7_13-good-of-21-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240216_145824836132",  # rec-1,2,4,5,6,7_28-good-of-45-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240216_145831608149",  # rec-1,2,4,5,6,7_20-good-of-32-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240216_145842258686",  # rec-1,2,4,5,6,7_17-good-of-35-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240216_145849806507",  # rec-1,2,4,5,6,7_30-good-of-48-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240216_150011205066",  # rec-1,2,4,5,6,7_17-good-of-32-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240216_150018099934",  # rec-1,2,4,5,6,7_34-good-of-57-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240216_150024637288",  # rec-1,2,4,5,6,7_23-good-of-34-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240216_150102108702",  # rec-1,2,4,5,6,7_16-good-of-27-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240216_150112224007",  # rec-1,2,4,5,6,7_28-good-of-44-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240216_150307573119",  # rec-1,2,4,5,6,7_20-good-of-45-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240216_150437299215",  # rec-1,2,4,5,6,7_18-good-of-24-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240216_185052713322",  # rec-1,2,4,5,6,7_23-good-of-36-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240216_185054792330",  # rec-1,2,4,5,6,7_20-good-of-34-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240216_185104438492",  # rec-1,2,4,5,6,7_22-good-of-37-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240216_185111507127",  # rec-1,2,4,5,6,7_23-good-of-40-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240216_185123001062",  # rec-1,2,4,5,6,7_19-good-of-39-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240216_185142040352",  # rec-1,2,4,5,6,7_21-good-of-39-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240216_185303933394",  # rec-1,2,4,5,6,7_18-good-of-28-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240216_185336519637",  # rec-1,2,4,5,6,7_17-good-of-30-total_Th,[2,1],spkTh,-6_vanilla_KS
        # EMUsort with extended grid search, 300 noise, without sgolay filter
        # "20240216_134504989837",  # rec-1,2,4,5,6,7_19-good-of-32-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240216_134602148811",  # rec-1,2,4,5,6,7_21-good-of-27-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240216_134607616517",  # rec-1,2,4,5,6,7_19-good-of-29-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240216_134626375162",  # rec-1,2,4,5,6,7_14-good-of-22-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240216_134655026839",  # rec-1,2,4,5,6,7_19-good-of-29-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240216_134945173531",  # rec-1,2,4,5,6,7_30-good-of-41-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240216_135111435841",  # rec-1,2,4,5,6,7_14-good-of-19-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240216_135111500902",  # rec-1,2,4,5,6,7_19-good-of-31-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240216_135301274397",  # rec-1,2,4,5,6,7_18-good-of-24-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240216_135324553331",  # rec-1,2,4,5,6,7_18-good-of-26-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240216_135407834980",  # rec-1,2,4,5,6,7_19-good-of-23-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240216_135626107203",  # rec-1,2,4,5,6,7_13-good-of-18-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240216_135637109358",  # rec-1,2,4,5,6,7_13-good-of-19-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240216_135737328414",  # rec-1,2,4,5,6,7_20-good-of-30-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240216_135737862223",  # rec-1,2,4,5,6,7_20-good-of-33-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240216_135945829764",  # rec-1,2,4,5,6,7_19-good-of-25-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240216_140854155213",  # rec-1,2,4,5,6,7_15-good-of-22-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240216_141231164527",  # rec-1,2,4,5,6,7_16-good-of-22-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240216_142720098104",  # rec-1,2,4,5,6,7_10-good-of-21-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240216_144236943320",  # rec-1,2,4,5,6,7_22-good-of-30-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        #####
        #####
        # >= 20240217 for godzilla 10 MU, 8 CH dataset
        ## EMUsort with extended grid search, None noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_224003193512",  # rec-1,2,4,5,6,7_30-good-of-56-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240217_224108054752",  # rec-1,2,4,5,6,7_30-good-of-53-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240217_224141239120",  # rec-1,2,4,5,6,7_20-good-of-47-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240217_224152269940",  # rec-1,2,4,5,6,7_20-good-of-33-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240217_224211353949",  # rec-1,2,4,5,6,7_21-good-of-36-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240217_224326279576",  # rec-1,2,4,5,6,7_31-good-of-53-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240217_224345530015",  # rec-1,2,4,5,6,7_26-good-of-43-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240217_224626823237",  # rec-1,2,4,5,6,7_28-good-of-47-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240217_224847712317",  # rec-1,2,4,5,6,7_31-good-of-48-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240217_225152732046",  # rec-1,2,4,5,6,7_19-good-of-44-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240217_225241237508",  # rec-1,2,4,5,6,7_19-good-of-35-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240217_225412138316",  # rec-1,2,4,5,6,7_25-good-of-53-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240217_225418313668",  # rec-1,2,4,5,6,7_22-good-of-39-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240217_225529556786",  # rec-1,2,4,5,6,7_27-good-of-44-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240217_225627826416",  # rec-1,2,4,5,6,7_33-good-of-60-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240217_225824031470",  # rec-1,2,4,5,6,7_24-good-of-48-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240217_225920326910",  # rec-1,2,4,5,6,7_42-good-of-61-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240217_230029875076",  # rec-1,2,4,5,6,7_20-good-of-34-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240217_230033887205",  # rec-1,2,4,5,6,7_18-good-of-28-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240217_230350788243",  # rec-1,2,4,5,6,7_26-good-of-55-total_Th,[5,2],spkTh,[-6]_EMUsort
        # KS3 with comp extended grid search, None noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_231633349195",  # rec-1,2,4,5,6,7_23-good-of-41-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240217_231655749865",  # rec-1,2,4,5,6,7_28-good-of-51-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240217_231659616705",  # rec-1,2,4,5,6,7_25-good-of-38-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240217_231713589705",  # rec-1,2,4,5,6,7_23-good-of-42-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240217_231717416227",  # rec-1,2,4,5,6,7_27-good-of-45-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240217_231729641361",  # rec-1,2,4,5,6,7_30-good-of-53-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240217_231732606702",  # rec-1,2,4,5,6,7_38-good-of-62-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240217_231741270098",  # rec-1,2,4,5,6,7_42-good-of-87-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240217_231923201778",  # rec-1,2,4,5,6,7_24-good-of-45-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240217_231940429488",  # rec-1,2,4,5,6,7_33-good-of-72-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240217_231958781297",  # rec-1,2,4,5,6,7_26-good-of-43-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240217_232015522833",  # rec-1,2,4,5,6,7_27-good-of-46-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240217_232442771911",  # rec-1,2,4,5,6,7_24-good-of-44-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240217_232508234168",  # rec-1,2,4,5,6,7_21-good-of-35-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240217_232510229797",  # rec-1,2,4,5,6,7_35-good-of-74-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240217_232514581960",  # rec-1,2,4,5,6,7_24-good-of-34-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240217_232521618753",  # rec-1,2,4,5,6,7_41-good-of-76-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240217_232527658456",  # rec-1,2,4,5,6,7_27-good-of-45-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240217_232542700124",  # rec-1,2,4,5,6,7_40-good-of-80-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240217_232555750144",  # rec-1,2,4,5,6,7_36-good-of-67-total_Th,[5,2],spkTh,-6_vanilla_KS
        # EMUsort with extended grid search, 100 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_192122267876",  # rec-1,2,4,5,6,7_24-good-of-38-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240217_192236125563",  # rec-1,2,4,5,6,7_23-good-of-44-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240217_192325659165",  # rec-1,2,4,5,6,7_27-good-of-41-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240217_192346544292",  # rec-1,2,4,5,6,7_18-good-of-34-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240217_192355158664",  # rec-1,2,4,5,6,7_27-good-of-46-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240217_192423273218",  # rec-1,2,4,5,6,7_12-good-of-21-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240217_192838365155",  # rec-1,2,4,5,6,7_9-good-of-16-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240217_192858798234",  # rec-1,2,4,5,6,7_13-good-of-19-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240217_192916906942",  # rec-1,2,4,5,6,7_21-good-of-42-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240217_193107995203",  # rec-1,2,4,5,6,7_18-good-of-36-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240217_193322235903",  # rec-1,2,4,5,6,7_20-good-of-25-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240217_193435393281",  # rec-1,2,4,5,6,7_15-good-of-33-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240217_193435570686",  # rec-1,2,4,5,6,7_20-good-of-28-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240217_193625407292",  # rec-1,2,4,5,6,7_18-good-of-34-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240217_193647102438",  # rec-1,2,4,5,6,7_17-good-of-24-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240217_194023805917",  # rec-1,2,4,5,6,7_12-good-of-18-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240217_194036970007",  # rec-1,2,4,5,6,7_21-good-of-41-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240217_194126031933",  # rec-1,2,4,5,6,7_15-good-of-24-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240217_194325607757",  # rec-1,2,4,5,6,7_18-good-of-33-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240217_194418940074",  # rec-1,2,4,5,6,7_22-good-of-33-total_Th,[2,1],spkTh,[-9]_EMUsort
        # KS3 with comp extended grid search, 100 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_233255126443",  # rec-1,2,4,5,6,7_17-good-of-27-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240217_233305449152",  # rec-1,2,4,5,6,7_25-good-of-40-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240217_233336792862",  # rec-1,2,4,5,6,7_40-good-of-67-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240217_233338174785",  # rec-1,2,4,5,6,7_31-good-of-51-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240217_233339883848",  # rec-1,2,4,5,6,7_34-good-of-65-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240217_233447782629",  # rec-1,2,4,5,6,7_27-good-of-39-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240217_233448277865",  # rec-1,2,4,5,6,7_15-good-of-25-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240217_233514747828",  # rec-1,2,4,5,6,7_24-good-of-34-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240217_233516880078",  # rec-1,2,4,5,6,7_25-good-of-41-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240217_233552967775",  # rec-1,2,4,5,6,7_28-good-of-54-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240217_233603235866",  # rec-1,2,4,5,6,7_29-good-of-53-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240217_233625779661",  # rec-1,2,4,5,6,7_35-good-of-53-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240217_234943274547",  # rec-1,2,4,5,6,7_26-good-of-35-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240217_234946979058",  # rec-1,2,4,5,6,7_19-good-of-37-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240217_235019441640",  # rec-1,2,4,5,6,7_30-good-of-51-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240217_235026880934",  # rec-1,2,4,5,6,7_29-good-of-54-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240217_235027999358",  # rec-1,2,4,5,6,7_35-good-of-53-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240217_235031070781",  # rec-1,2,4,5,6,7_27-good-of-53-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240217_235111528027",  # rec-1,2,4,5,6,7_15-good-of-23-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240217_235116483852",  # rec-1,2,4,5,6,7_27-good-of-38-total_Th,[2,1],spkTh,-6_vanilla_KS
        ## EMUsort with extended grid search, 200 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_195521501495",  # rec-1,2,4,5,6,7_17-good-of-29-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240217_195556814631",  # rec-1,2,4,5,6,7_15-good-of-19-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240217_195707422437",  # rec-1,2,4,5,6,7_17-good-of-21-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240217_195812260301",  # rec-1,2,4,5,6,7_13-good-of-23-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240217_195825049122",  # rec-1,2,4,5,6,7_17-good-of-23-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240217_195900555193",  # rec-1,2,4,5,6,7_19-good-of-32-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240217_200154269269",  # rec-1,2,4,5,6,7_15-good-of-19-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240217_200234978058",  # rec-1,2,4,5,6,7_21-good-of-29-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240217_200427297896",  # rec-1,2,4,5,6,7_21-good-of-30-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240217_200707001451",  # rec-1,2,4,5,6,7_16-good-of-22-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240217_200831447384",  # rec-1,2,4,5,6,7_24-good-of-36-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240217_200904395477",  # rec-1,2,4,5,6,7_18-good-of-32-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240217_200913864079",  # rec-1,2,4,5,6,7_17-good-of-25-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240217_200941881432",  # rec-1,2,4,5,6,7_16-good-of-26-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240217_201105858246",  # rec-1,2,4,5,6,7_13-good-of-20-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240217_201236641018",  # rec-1,2,4,5,6,7_19-good-of-27-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240217_201539444875",  # rec-1,2,4,5,6,7_16-good-of-31-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240217_201750151788",  # rec-1,2,4,5,6,7_15-good-of-22-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240217_202349700245",  # rec-1,2,4,5,6,7_10-good-of-19-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240217_202448036644",  # rec-1,2,4,5,6,7_16-good-of-23-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # ## KS3 with comp extended grid search, 200 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_235539758904",  # rec-1,2,4,5,6,7_15-good-of-22-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240217_235544170809",  # rec-1,2,4,5,6,7_16-good-of-24-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240217_235622239161",  # rec-1,2,4,5,6,7_27-good-of-45-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240217_235624051935",  # rec-1,2,4,5,6,7_25-good-of-49-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240217_235639705692",  # rec-1,2,4,5,6,7_29-good-of-57-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240217_235754992983",  # rec-1,2,4,5,6,7_24-good-of-33-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240217_235758396813",  # rec-1,2,4,5,6,7_12-good-of-14-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240217_235812008674",  # rec-1,2,4,5,6,7_28-good-of-46-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240217_235848229410",  # rec-1,2,4,5,6,7_38-good-of-65-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240217_235859286917",  # rec-1,2,4,5,6,7_23-good-of-40-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240217_235900367407",  # rec-1,2,4,5,6,7_16-good-of-25-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240217_235923628397",  # rec-1,2,4,5,6,7_23-good-of-35-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240218_000601635490",  # rec-1,2,4,5,6,7_25-good-of-36-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240218_000605135764",  # rec-1,2,4,5,6,7_24-good-of-41-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240218_000627680830",  # rec-1,2,4,5,6,7_20-good-of-41-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240218_000627762187",  # rec-1,2,4,5,6,7_35-good-of-55-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240218_000642207605",  # rec-1,2,4,5,6,7_39-good-of-79-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240218_000651006235",  # rec-1,2,4,5,6,7_31-good-of-57-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240218_000823934226",  # rec-1,2,4,5,6,7_12-good-of-16-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240218_000842388820",  # rec-1,2,4,5,6,7_15-good-of-24-total_Th,[2,1],spkTh,-9_vanilla_KS
        ## EMUsort with extended grid search, 300 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_205213152532",  # rec-1,2,4,5,6,7_17-good-of-29-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240217_205248779700",  # rec-1,2,4,5,6,7_14-good-of-23-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240217_205335452432",  # rec-1,2,4,5,6,7_14-good-of-20-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240217_205454241766",  # rec-1,2,4,5,6,7_22-good-of-36-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240217_205527660904",  # rec-1,2,4,5,6,7_14-good-of-20-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240217_205612737711",  # rec-1,2,4,5,6,7_15-good-of-29-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240217_205737795895",  # rec-1,2,4,5,6,7_17-good-of-23-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240217_205834644935",  # rec-1,2,4,5,6,7_18-good-of-29-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240217_210029838134",  # rec-1,2,4,5,6,7_19-good-of-24-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240217_210229533257",  # rec-1,2,4,5,6,7_15-good-of-25-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240217_210328042975",  # rec-1,2,4,5,6,7_17-good-of-25-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240217_210504142829",  # rec-1,2,4,5,6,7_16-good-of-25-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240217_210504798508",  # rec-1,2,4,5,6,7_13-good-of-20-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240217_210607971168",  # rec-1,2,4,5,6,7_12-good-of-15-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240217_210716250683",  # rec-1,2,4,5,6,7_14-good-of-20-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240217_210723914181",  # rec-1,2,4,5,6,7_19-good-of-32-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240217_211056792741",  # rec-1,2,4,5,6,7_14-good-of-19-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240217_211426290382",  # rec-1,2,4,5,6,7_13-good-of-21-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240217_212119870180",  # rec-1,2,4,5,6,7_14-good-of-23-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240217_212508770260",  # rec-1,2,4,5,6,7_13-good-of-18-total_Th,[2,1],spkTh,[-9]_EMUsort
        # ## KS3 with comp extended grid search, 300 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240218_001457617699",  # rec-1,2,4,5,6,7_24-good-of-39-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240218_001503051000",  # rec-1,2,4,5,6,7_26-good-of-42-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240218_001525264574",  # rec-1,2,4,5,6,7_19-good-of-38-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240218_001536187180",  # rec-1,2,4,5,6,7_24-good-of-42-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240218_001543746505",  # rec-1,2,4,5,6,7_28-good-of-37-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240218_001706723417",  # rec-1,2,4,5,6,7_17-good-of-24-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240218_001728982374",  # rec-1,2,4,5,6,7_30-good-of-50-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240218_001747381677",  # rec-1,2,4,5,6,7_29-good-of-49-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240218_001757303558",  # rec-1,2,4,5,6,7_25-good-of-35-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240218_001804812070",  # rec-1,2,4,5,6,7_24-good-of-28-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240218_001821946080",  # rec-1,2,4,5,6,7_24-good-of-44-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240218_001822451087",  # rec-1,2,4,5,6,7_27-good-of-33-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240218_003116183655",  # rec-1,2,4,5,6,7_19-good-of-24-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240218_003122532150",  # rec-1,2,4,5,6,7_20-good-of-33-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240218_003138491336",  # rec-1,2,4,5,6,7_23-good-of-37-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240218_003151256524",  # rec-1,2,4,5,6,7_19-good-of-38-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240218_003200849500",  # rec-1,2,4,5,6,7_23-good-of-35-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240218_003207952692",  # rec-1,2,4,5,6,7_24-good-of-45-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240218_003401886807",  # rec-1,2,4,5,6,7_24-good-of-28-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240218_003432066613",  # rec-1,2,4,5,6,7_29-good-of-34-total_Th,[2,1],spkTh,-9_vanilla_KS
        ## EMUsort with extended grid search, 400 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240217_214034268104",  # rec-1,2,4,5,6,7_16-good-of-20-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240217_214037411440",  # rec-1,2,4,5,6,7_21-good-of-34-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240217_214048686091",  # rec-1,2,4,5,6,7_14-good-of-17-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240217_214221908317",  # rec-1,2,4,5,6,7_14-good-of-22-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240217_214510207183",  # rec-1,2,4,5,6,7_17-good-of-25-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240217_214531242943",  # rec-1,2,4,5,6,7_17-good-of-32-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240217_214551377608",  # rec-1,2,4,5,6,7_15-good-of-19-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240217_214607656420",  # rec-1,2,4,5,6,7_28-good-of-34-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240217_214632862386",  # rec-1,2,4,5,6,7_15-good-of-24-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240217_214957744559",  # rec-1,2,4,5,6,7_27-good-of-35-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240217_215128168817",  # rec-1,2,4,5,6,7_10-good-of-17-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240217_215156202015",  # rec-1,2,4,5,6,7_17-good-of-28-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240217_215228005465",  # rec-1,2,4,5,6,7_18-good-of-26-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240217_215309688453",  # rec-1,2,4,5,6,7_12-good-of-20-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240217_215612328805",  # rec-1,2,4,5,6,7_17-good-of-27-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240217_215856145144",  # rec-1,2,4,5,6,7_13-good-of-24-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240217_220053259185",  # rec-1,2,4,5,6,7_17-good-of-26-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240217_220524277931",  # rec-1,2,4,5,6,7_12-good-of-26-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240217_221122178214",  # rec-1,2,4,5,6,7_16-good-of-22-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240217_221309252093",  # rec-1,2,4,5,6,7_15-good-of-21-total_Th,[2,1],spkTh,[-9]_EMUsort
        # KS3 with comp extended grid search, 400 noise, without sgolay filter, with MUsim force thresh bugfix
        # "20240218_004114984029",  # rec-1,2,4,5,6,7_17-good-of-30-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240218_004115109274",  # rec-1,2,4,5,6,7_21-good-of-34-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240218_004126413615",  # rec-1,2,4,5,6,7_11-good-of-21-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240218_004213492592",  # rec-1,2,4,5,6,7_17-good-of-28-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240218_004225655619",  # rec-1,2,4,5,6,7_20-good-of-29-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240218_004307384579",  # rec-1,2,4,5,6,7_15-good-of-23-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240218_004327328879",  # rec-1,2,4,5,6,7_17-good-of-29-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240218_004345707117",  # rec-1,2,4,5,6,7_22-good-of-39-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240218_004353871492",  # rec-1,2,4,5,6,7_11-good-of-16-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240218_004358918919",  # rec-1,2,4,5,6,7_14-good-of-20-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240218_004416239994",  # rec-1,2,4,5,6,7_13-good-of-22-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240218_004423785219",  # rec-1,2,4,5,6,7_12-good-of-14-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240218_004948647206",  # rec-1,2,4,5,6,7_17-good-of-25-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240218_004953834378",  # rec-1,2,4,5,6,7_20-good-of-35-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240218_005008033690",  # rec-1,2,4,5,6,7_12-good-of-22-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240218_005018587180",  # rec-1,2,4,5,6,7_23-good-of-42-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240218_005023676015",  # rec-1,2,4,5,6,7_12-good-of-14-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240218_005042258088",  # rec-1,2,4,5,6,7_20-good-of-29-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240218_005214186792",  # rec-1,2,4,5,6,7_14-good-of-20-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240218_005228258671",  # rec-1,2,4,5,6,7_12-good-of-22-total_Th,[2,1],spkTh,-9_vanilla_KS
        #####
        #####
        ## >= 20240220, Gaussian noise level comparison
        # EMUsort None noise, with 20 sorts no duplicates
        # "20240221_194225564737",  # rec-1,2,4,5,6,7_21-good-of-31-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240221_194616828336",  # rec-1,2,4,5,6,7_31-good-of-63-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240221_194804296657",  # rec-1,2,4,5,6,7_26-good-of-48-total_Th,[2,1],spkTh,[-6]_EMUsort $
        # "20240221_194806291347",  # rec-1,2,4,5,6,7_28-good-of-52-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240221_194853069212",  # rec-1,2,4,5,6,7_22-good-of-55-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240221_194926600290",  # rec-1,2,4,5,6,7_30-good-of-65-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240221_194945979086",  # rec-1,2,4,5,6,7_28-good-of-55-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240221_195051770093",  # rec-1,2,4,5,6,7_16-good-of-29-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240221_195307081960",  # rec-1,2,4,5,6,7_25-good-of-37-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240221_195847085585",  # rec-1,2,4,5,6,7_20-good-of-50-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240221_200311467782",  # rec-1,2,4,5,6,7_17-good-of-36-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240221_200528129089",  # rec-1,2,4,5,6,7_31-good-of-70-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240221_200602806964",  # rec-1,2,4,5,6,7_25-good-of-51-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240221_200617109817",  # rec-1,2,4,5,6,7_29-good-of-69-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240221_200707891813",  # rec-1,2,4,5,6,7_23-good-of-49-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240221_200744604104",  # rec-1,2,4,5,6,7_25-good-of-50-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240221_200944947431",  # rec-1,2,4,5,6,7_34-good-of-70-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240221_201018827148",  # rec-1,2,4,5,6,7_13-good-of-32-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240221_201259033819",  # rec-1,2,4,5,6,7_24-good-of-39-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240221_201911080030",  # rec-1,2,4,5,6,7_25-good-of-49-total_Th,[5,2],spkTh,[-6]_EMUsort
        # # Kilosort None noise, with 20 sorts no duplicates
        # "20240221_212948303345",  # rec-1,2,4,5,6,7_17-good-of-35-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240221_212957038595",  # rec-1,2,4,5,6,7_26-good-of-44-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240221_213030488264",  # rec-1,2,4,5,6,7_25-good-of-45-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240221_213048167806",  # rec-1,2,4,5,6,7_28-good-of-63-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240221_213107059441",  # rec-1,2,4,5,6,7_27-good-of-56-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240221_213112218765",  # rec-1,2,4,5,6,7_25-good-of-54-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240221_213119572208",  # rec-1,2,4,5,6,7_45-good-of-97-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240221_213124386835",  # rec-1,2,4,5,6,7_37-good-of-62-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240221_213239392454",  # rec-1,2,4,5,6,7_23-good-of-46-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240221_213310991514",  # rec-1,2,4,5,6,7_34-good-of-68-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240221_213455947947",  # rec-1,2,4,5,6,7_20-good-of-44-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240221_213456565400",  # rec-1,2,4,5,6,7_20-good-of-38-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240221_213503064652",  # rec-1,2,4,5,6,7_37-good-of-77-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240221_213506825237",  # rec-1,2,4,5,6,7_18-good-of-35-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240221_213517883880",  # rec-1,2,4,5,6,7_31-good-of-59-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240221_213526154348",  # rec-1,2,4,5,6,7_29-good-of-47-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240221_213542293664",  # rec-1,2,4,5,6,7_29-good-of-54-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240221_213543376363",  # rec-1,2,4,5,6,7_13-good-of-29-total_Th,[7,3],spkTh,-3_vanilla_KS $
        # "20240221_213738494238",  # rec-1,2,4,5,6,7_29-good-of-47-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240221_213754340661",  # rec-1,2,4,5,6,7_28-good-of-55-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        ## EMUsort 10 noise, with 20 sorts no duplicates
        # "20240221_203627096172",  # rec-1,2,4,5,6,7_21-good-of-44-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240221_203647102873",  # rec-1,2,4,5,6,7_31-good-of-69-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240221_203932254381",  # rec-1,2,4,5,6,7_18-good-of-39-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240221_203957961033",  # rec-1,2,4,5,6,7_21-good-of-47-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240221_204055583927",  # rec-1,2,4,5,6,7_32-good-of-55-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240221_204234589562",  # rec-1,2,4,5,6,7_32-good-of-68-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240221_204238440240",  # rec-1,2,4,5,6,7_32-good-of-71-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240221_204312223114",  # rec-1,2,4,5,6,7_32-good-of-64-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240221_204733389681",  # rec-1,2,4,5,6,7_23-good-of-49-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240221_204748017658",  # rec-1,2,4,5,6,7_25-good-of-45-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240221_205433548814",  # rec-1,2,4,5,6,7_23-good-of-45-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240221_205435321118",  # rec-1,2,4,5,6,7_22-good-of-38-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240221_205603333914",  # rec-1,2,4,5,6,7_17-good-of-36-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240221_205722810626",  # rec-1,2,4,5,6,7_18-good-of-38-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240221_205816993478",  # rec-1,2,4,5,6,7_31-good-of-71-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240221_205818291897",  # rec-1,2,4,5,6,7_15-good-of-30-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240221_205912382252",  # rec-1,2,4,5,6,7_30-good-of-85-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240221_210134793142",  # rec-1,2,4,5,6,7_22-good-of-41-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240221_210531778884",  # rec-1,2,4,5,6,7_29-good-of-81-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240221_210706476813",  # rec-1,2,4,5,6,7_26-good-of-51-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        ## Kilosort 10 noise, with 20 sorts no duplicates
        # "20240221_211707190848",  # rec-1,2,4,5,6,7_21-good-of-37-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240221_211714115022",  # rec-1,2,4,5,6,7_23-good-of-54-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240221_211744460642",  # rec-1,2,4,5,6,7_31-good-of-55-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240221_211758603061",  # rec-1,2,4,5,6,7_28-good-of-56-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240221_211813841457",  # rec-1,2,4,5,6,7_27-good-of-48-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240221_211825505862",  # rec-1,2,4,5,6,7_37-good-of-62-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240221_211843467657",  # rec-1,2,4,5,6,7_40-good-of-86-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240221_211846677218",  # rec-1,2,4,5,6,7_30-good-of-51-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240221_212018355182",  # rec-1,2,4,5,6,7_30-good-of-47-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240221_212032769602",  # rec-1,2,4,5,6,7_40-good-of-75-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240221_212145454994",  # rec-1,2,4,5,6,7_32-good-of-59-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240221_212206179477",  # rec-1,2,4,5,6,7_25-good-of-49-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240221_212210857042",  # rec-1,2,4,5,6,7_23-good-of-42-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240221_212237941038",  # rec-1,2,4,5,6,7_28-good-of-53-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240221_212300082997",  # rec-1,2,4,5,6,7_32-good-of-69-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240221_212306894517",  # rec-1,2,4,5,6,7_43-good-of-80-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240221_212308129120",  # rec-1,2,4,5,6,7_22-good-of-48-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240221_212333847315",  # rec-1,2,4,5,6,7_24-good-of-39-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240221_212428424425",  # rec-1,2,4,5,6,7_26-good-of-41-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240221_212523736963",  # rec-1,2,4,5,6,7_26-good-of-70-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # EMUsort 100 noise, with 20 sorts no duplicates
        # "20240221_163400874837",  # rec-1,2,4,5,6,7_17-good-of-37-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240221_163531190104",  # rec-1,2,4,5,6,7_34-good-of-59-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240221_163555134178",  # rec-1,2,4,5,6,7_21-good-of-43-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240221_163822292059",  # rec-1,2,4,5,6,7_30-good-of-56-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240221_163853065718",  # rec-1,2,4,5,6,7_24-good-of-49-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240221_163936955356",  # rec-1,2,4,5,6,7_22-good-of-55-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240221_164105628067",  # rec-1,2,4,5,6,7_21-good-of-47-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240221_164224318977",  # rec-1,2,4,5,6,7_22-good-of-65-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240221_164354820488",  # rec-1,2,4,5,6,7_26-good-of-52-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240221_164626135332",  # rec-1,2,4,5,6,7_28-good-of-47-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240221_164642673423",  # rec-1,2,4,5,6,7_12-good-of-18-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240221_165220456488",  # rec-1,2,4,5,6,7_35-good-of-59-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240221_165323789495",  # rec-1,2,4,5,6,7_22-good-of-41-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240221_165624793951",  # rec-1,2,4,5,6,7_15-good-of-32-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240221_165625595283",  # rec-1,2,4,5,6,7_22-good-of-47-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240221_165630520884",  # rec-1,2,4,5,6,7_22-good-of-39-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240221_165631118099",  # rec-1,2,4,5,6,7_29-good-of-55-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240221_165655445611",  # rec-1,2,4,5,6,7_28-good-of-62-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240221_165918163843",  # rec-1,2,4,5,6,7_25-good-of-60-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240221_170036244672",  # rec-1,2,4,5,6,7_18-good-of-37-total_Th,[7,3],spkTh,[-3,-6]_EMUsort $
        # ## Kilosort 100 noise, with 20 sorts no duplicates
        # "20240221_133106648415",  # rec-1,2,4,5,6,7_27-good-of-44-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240221_133118526648",  # rec-1,2,4,5,6,7_28-good-of-53-total_Th,[10,4],spkTh,-3_vanilla_KS $
        # "20240221_133126855960",  # rec-1,2,4,5,6,7_24-good-of-47-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240221_133140954797",  # rec-1,2,4,5,6,7_20-good-of-45-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240221_133143129221",  # rec-1,2,4,5,6,7_23-good-of-43-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240221_133153354496",  # rec-1,2,4,5,6,7_22-good-of-44-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240221_133227043901",  # rec-1,2,4,5,6,7_29-good-of-47-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240221_133254567367",  # rec-1,2,4,5,6,7_14-good-of-29-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240221_133353695871",  # rec-1,2,4,5,6,7_19-good-of-34-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240221_133408445884",  # rec-1,2,4,5,6,7_28-good-of-44-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240221_133519836519",  # rec-1,2,4,5,6,7_23-good-of-53-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240221_133529995108",  # rec-1,2,4,5,6,7_33-good-of-69-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240221_133540841801",  # rec-1,2,4,5,6,7_32-good-of-61-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240221_133551915635",  # rec-1,2,4,5,6,7_19-good-of-28-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240221_133554800560",  # rec-1,2,4,5,6,7_17-good-of-30-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240221_133644738491",  # rec-1,2,4,5,6,7_23-good-of-46-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240221_133645961600",  # rec-1,2,4,5,6,7_18-good-of-37-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240221_133700081344",  # rec-1,2,4,5,6,7_26-good-of-58-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240221_133808858843",  # rec-1,2,4,5,6,7_32-good-of-65-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240221_133846416679",  # rec-1,2,4,5,6,7_33-good-of-67-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # EMUsort 400 noise, with 20 sorts no duplicates
        # "20240221_171027803868",  # rec-1,2,4,5,6,7_21-good-of-32-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240221_171136805684",  # rec-1,2,4,5,6,7_28-good-of-42-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240221_171222698991",  # rec-1,2,4,5,6,7_25-good-of-37-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240221_171259584166",  # rec-1,2,4,5,6,7_14-good-of-25-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240221_171450419366",  # rec-1,2,4,5,6,7_16-good-of-29-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240221_171620122101",  # rec-1,2,4,5,6,7_22-good-of-37-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240221_171651647610",  # rec-1,2,4,5,6,7_17-good-of-33-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240221_171704540877",  # rec-1,2,4,5,6,7_17-good-of-32-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240221_172017725527",  # rec-1,2,4,5,6,7_22-good-of-38-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240221_172101877604",  # rec-1,2,4,5,6,7_20-good-of-27-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240221_172339017124",  # rec-1,2,4,5,6,7_15-good-of-23-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240221_172421082703",  # rec-1,2,4,5,6,7_14-good-of-22-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240221_172759742267",  # rec-1,2,4,5,6,7_16-good-of-23-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240221_172809998394",  # rec-1,2,4,5,6,7_15-good-of-30-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240221_172944067983",  # rec-1,2,4,5,6,7_20-good-of-37-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240221_173155880140",  # rec-1,2,4,5,6,7_17-good-of-25-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240221_173238936002",  # rec-1,2,4,5,6,7_18-good-of-39-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240221_173812058043",  # rec-1,2,4,5,6,7_11-good-of-19-total_Th,[2,1],spkTh,[-3]_EMUsort $
        # "20240221_174620631295",  # rec-1,2,4,5,6,7_16-good-of-31-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240221_174659581116",  # rec-1,2,4,5,6,7_17-good-of-33-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # ## Kilosort 400 noise, with 20 sorts no duplicates
        # "20240221_103958159298",  # rec-1,2,4,5,6,7_21-good-of-38-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240221_104007914622",  # rec-1,2,4,5,6,7_25-good-of-47-total_Th,[10,4],spkTh,-3_vanilla_KS $
        # "20240221_104017456029",  # rec-1,2,4,5,6,7_22-good-of-33-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240221_104035971236",  # rec-1,2,4,5,6,7_13-good-of-33-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240221_104041899650",  # rec-1,2,4,5,6,7_27-good-of-48-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240221_104049587617",  # rec-1,2,4,5,6,7_22-good-of-44-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240221_104126252826",  # rec-1,2,4,5,6,7_23-good-of-50-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240221_104127451785",  # rec-1,2,4,5,6,7_22-good-of-33-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240221_104246971496",  # rec-1,2,4,5,6,7_15-good-of-35-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240221_104257899573",  # rec-1,2,4,5,6,7_26-good-of-48-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240221_104408745141",  # rec-1,2,4,5,6,7_31-good-of-52-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240221_104409350916",  # rec-1,2,4,5,6,7_17-good-of-45-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240221_104431623312",  # rec-1,2,4,5,6,7_35-good-of-66-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240221_104441875432",  # rec-1,2,4,5,6,7_14-good-of-25-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240221_104455027659",  # rec-1,2,4,5,6,7_17-good-of-30-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240221_104517281029",  # rec-1,2,4,5,6,7_18-good-of-37-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240221_104534228130",  # rec-1,2,4,5,6,7_23-good-of-46-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240221_104549369709",  # rec-1,2,4,5,6,7_28-good-of-58-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240221_104653189225",  # rec-1,2,4,5,6,7_32-good-of-56-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240221_104727901706",  # rec-1,2,4,5,6,7_29-good-of-55-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        ## EMUsort 700 noise, with 20 sorts no duplicates
        # "20240221_182235280128",  # rec-1,2,4,5,6,7_12-good-of-21-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240221_182301041930",  # rec-1,2,4,5,6,7_15-good-of-22-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240221_182315124792",  # rec-1,2,4,5,6,7_15-good-of-21-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240221_182351054068",  # rec-1,2,4,5,6,7_21-good-of-30-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240221_182611169424",  # rec-1,2,4,5,6,7_16-good-of-21-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240221_182735082932",  # rec-1,2,4,5,6,7_15-good-of-26-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240221_182746882583",  # rec-1,2,4,5,6,7_13-good-of-18-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240221_182930426009",  # rec-1,2,4,5,6,7_18-good-of-25-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240221_182930978830",  # rec-1,2,4,5,6,7_15-good-of-23-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240221_183023601262",  # rec-1,2,4,5,6,7_11-good-of-21-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240221_183139584011",  # rec-1,2,4,5,6,7_11-good-of-18-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240221_183147976795",  # rec-1,2,4,5,6,7_18-good-of-24-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240221_183343095915",  # rec-1,2,4,5,6,7_12-good-of-24-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240221_183451574702",  # rec-1,2,4,5,6,7_13-good-of-23-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240221_183501976399",  # rec-1,2,4,5,6,7_11-good-of-24-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240221_184128661902",  # rec-1,2,4,5,6,7_9-good-of-13-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240221_184222639581",  # rec-1,2,4,5,6,7_12-good-of-22-total_Th,[5,2],spkTh,[-6]_EMUsort $
        # "20240221_184259011511",  # rec-1,2,4,5,6,7_14-good-of-26-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240221_184335330785",  # rec-1,2,4,5,6,7_11-good-of-18-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240221_185049475004",  # rec-1,2,4,5,6,7_9-good-of-13-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # # Kilosort 700 noise, with 20 sorts no duplicates, 4 with too few clusters
        # "20240221_152943899395",  # rec-1,2,4,5,6,7_10-good-of-14-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240221_152946957426",  # rec-1,2,4,5,6,7_10-good-of-17-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240221_153002412013",  # rec-1,2,4,5,6,7_9-good-of-15-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240221_153004071743",  # rec-1,2,4,5,6,7_11-good-of-14-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # # "NotEnough20240221_153022664090",  # rec-1,2,4,5,6,7_4-good-of-8-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240221_153037106638",  # rec-1,2,4,5,6,7_9-good-of-15-total_Th,[5,2],spkTh,-9_vanilla_KS
        # # "20240221_153116024668",  # rec-1,2,4,5,6,7_8-good-of-10-total_Th,[10,4],spkTh,-9_vanilla_KS
        # # "NotEnough20240221_153116696619",  # rec-1,2,4,5,6,7_3-good-of-5-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # # "20240221_153142988388",  # rec-1,2,4,5,6,7_7-good-of-10-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # # "20240221_153154984574",  # rec-1,2,4,5,6,7_6-good-of-10-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240221_153213996113",  # rec-1,2,4,5,6,7_9-good-of-16-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240221_153234926603",  # rec-1,2,4,5,6,7_6-good-of-11-total_Th,[5,2],spkTh,-3_vanilla_KS
        # # "NotEnough20240221_153235047729",  # rec-1,2,4,5,6,7_4-good-of-7-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240221_153257482975",  # rec-1,2,4,5,6,7_14-good-of-19-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240221_153319191572",  # rec-1,2,4,5,6,7_11-good-of-18-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240221_153354547073",  # rec-1,2,4,5,6,7_7-good-of-14-total_Th,[7,3],spkTh,-7.5_vanilla_KS $
        # "20240221_153419149885",  # rec-1,2,4,5,6,7_7-good-of-11-total_Th,[2,1],spkTh,-3_vanilla_KS
        # # "20240221_153441540746",  # rec-1,2,4,5,6,7_4-good-of-10-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # # "NotEnough20240221_153451938100",  # rec-1,2,4,5,6,7_3-good-of-8-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240221_153509208179",  # rec-1,2,4,5,6,7_7-good-of-13-total_Th,[2,1],spkTh,-6_vanilla_KS
        ## EMUsort 1000 noise, with 18 sorts no duplicates, 4 with too few clusters
        # "20240221_190427415412",  # rec-1,2,4,5,6,7_8-good-of-12-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240221_190524361270",  # rec-1,2,4,5,6,7_12-good-of-21-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240221_190535330732",  # rec-1,2,4,5,6,7_6-good-of-12-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240221_190542336633",  # rec-1,2,4,5,6,7_9-good-of-11-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240221_190550354473",  # rec-1,2,4,5,6,7_8-good-of-12-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240221_190806135165",  # rec-1,2,4,5,6,7_14-good-of-19-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240221_190813959298",  # rec-1,2,4,5,6,7_8-good-of-13-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240221_190925173107",  # rec-1,2,4,5,6,7_6-good-of-11-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240221_190930030474",  # rec-1,2,4,5,6,7_14-good-of-19-total_Th,[10,4],spkTh,[-9]_EMUsort
        # # "NotEnough20240221_191056134089",  # rec-1,2,4,5,6,7_3-good-of-9-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240221_191138302300",  # rec-1,2,4,5,6,7_7-good-of-12-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240221_191250203804",  # rec-1,2,4,5,6,7_3-good-of-10-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240221_191254400937",  # rec-1,2,4,5,6,7_7-good-of-12-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240221_191350956547",  # rec-1,2,4,5,6,7_5-good-of-17-total_Th,[7,3],spkTh,[-3]_EMUsort
        # # "NotEnough20240221_191722925280",  # rec-1,2,4,5,6,7_1-good-of-6-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240221_191826500790",  # rec-1,2,4,5,6,7_9-good-of-15-total_Th,[5,2],spkTh,[-6]_EMUsort
        # # "NotEnough20240221_191829037276",  # rec-1,2,4,5,6,7_0-good-of-6-total_Th,[2,1],spkTh,[-3]_EMUsort
        # # "NotEnough20240221_193034478126",  # rec-1,2,4,5,6,7_4-good-of-8-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # Kilosort 1000 noise, with ONLY 12 sorts no duplicates, 12 with too few clusters
        # "20240221_121733467853",  # rec-1,2,4,5,6,7_6-good-of-9-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240221_121745273893",  # rec-1,2,4,5,6,7_2-good-of-9-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240221_121801927101",  # rec-1,2,4,5,6,7_0-good-of-5-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240221_121842577778",  # rec-1,2,4,5,6,7_0-good-of-2-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240221_121843270820",  # rec-1,2,4,5,6,7_0-good-of-2-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240221_121908243345",  # rec-1,2,4,5,6,7_2-good-of-6-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240221_121913497684",  # rec-1,2,4,5,6,7_2-good-of-8-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240221_121938746887",  # rec-1,2,4,5,6,7_0-good-of-2-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240221_121940413280",  # rec-1,2,4,5,6,7_0-good-of-3-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240221_121956906602",  # rec-1,2,4,5,6,7_2-good-of-4-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240221_122029787712",  # rec-1,2,4,5,6,7_0-good-of-4-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240221_122149446511",  # rec-1,2,4,5,6,7_1-good-of-5-total_Th,[2,1],spkTh,-6_vanilla_KS
        ## >= 20240301, wave shape noise level comparison
        # EMUsort 0 STD noise, with 20 sorts no duplicates
        # "20240301_122312863983",  # rec-1,2,4,5,6,7_11-good-of-31-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240301_122538594125",  # rec-1,2,4,5,6,7_19-good-of-37-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240301_122929618233",  # rec-1,2,4,5,6,7_12-good-of-28-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240301_122953329965",  # rec-1,2,4,5,6,7_27-good-of-55-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240301_123011211951",  # rec-1,2,4,5,6,7_19-good-of-35-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240301_123019528097",  # rec-1,2,4,5,6,7_18-good-of-31-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240301_123020997903",  # rec-1,2,4,5,6,7_15-good-of-30-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240301_123203322699",  # rec-1,2,4,5,6,7_18-good-of-32-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240301_123357378317",  # rec-1,2,4,5,6,7_19-good-of-52-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240301_123428322868",  # rec-1,2,4,5,6,7_19-good-of-35-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240301_123718010597",  # rec-1,2,4,5,6,7_18-good-of-27-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240301_124019204267",  # rec-1,2,4,5,6,7_18-good-of-26-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240301_124242793433",  # rec-1,2,4,5,6,7_25-good-of-48-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240301_124308061052",  # rec-1,2,4,5,6,7_33-good-of-72-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240301_124329251534",  # rec-1,2,4,5,6,7_17-good-of-32-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240301_124439916219",  # rec-1,2,4,5,6,7_23-good-of-43-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240301_124440626058",  # rec-1,2,4,5,6,7_17-good-of-37-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240301_124533603518",  # rec-1,2,4,5,6,7_15-good-of-29-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240301_124942622096",  # rec-1,2,4,5,6,7_23-good-of-41-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240301_125210434668",  # rec-1,2,4,5,6,7_15-good-of-31-total_Th,[5,2],spkTh,[-6]_EMUsort
        # Kilosort 0 STD noise, with 20 sorts no duplicates
        # "20240301_113842282964",  # rec-1,2,4,5,6,7_22-good-of-34-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240301_113842843042",  # rec-1,2,4,5,6,7_22-good-of-51-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240301_113857970992",  # rec-1,2,4,5,6,7_31-good-of-57-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240301_113918890043",  # rec-1,2,4,5,6,7_30-good-of-67-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240301_113925436716",  # rec-1,2,4,5,6,7_37-good-of-69-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240301_113933011726",  # rec-1,2,4,5,6,7_27-good-of-50-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240301_113935059031",  # rec-1,2,4,5,6,7_39-good-of-66-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240301_114023655931",  # rec-1,2,4,5,6,7_36-good-of-75-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240301_114135306664",  # rec-1,2,4,5,6,7_41-good-of-70-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240301_114141502302",  # rec-1,2,4,5,6,7_28-good-of-48-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240301_114213728435",  # rec-1,2,4,5,6,7_20-good-of-30-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240301_114229999314",  # rec-1,2,4,5,6,7_20-good-of-44-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240301_114246903335",  # rec-1,2,4,5,6,7_33-good-of-56-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240301_114256225572",  # rec-1,2,4,5,6,7_23-good-of-48-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240301_114315026328",  # rec-1,2,4,5,6,7_28-good-of-47-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240301_114353330536",  # rec-1,2,4,5,6,7_30-good-of-47-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240301_114411210265",  # rec-1,2,4,5,6,7_30-good-of-54-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240301_114416319030",  # rec-1,2,4,5,6,7_27-good-of-57-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240301_114452842873",  # rec-1,2,4,5,6,7_39-good-of-81-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240301_114514444712",  # rec-1,2,4,5,6,7_39-good-of-78-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # EMUsort 4 STD noise, with 20 sorts no duplicates
        # "20240301_132434991522",  # rec-1,2,4,5,6,7_17-good-of-31-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240301_132705310036",  # rec-1,2,4,5,6,7_20-good-of-33-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240301_132744061285",  # rec-1,2,4,5,6,7_15-good-of-30-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240301_132830121945",  # rec-1,2,4,5,6,7_14-good-of-29-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240301_132921725780",  # rec-1,2,4,5,6,7_8-good-of-17-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240301_132930335594",  # rec-1,2,4,5,6,7_11-good-of-19-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240301_133054912884",  # rec-1,2,4,5,6,7_12-good-of-21-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240301_133110233706",  # rec-1,2,4,5,6,7_22-good-of-52-total_Th,[2,1],spkTh,[-6]_EMUsort $$$
        # "20240301_133132179288",  # rec-1,2,4,5,6,7_23-good-of-35-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240301_133630583455",  # rec-1,2,4,5,6,7_13-good-of-27-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240301_133756748613",  # rec-1,2,4,5,6,7_14-good-of-24-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240301_133818723699",  # rec-1,2,4,5,6,7_14-good-of-30-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240301_134038235279",  # rec-1,2,4,5,6,7_15-good-of-23-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240301_134107303908",  # rec-1,2,4,5,6,7_15-good-of-25-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240301_134109442962",  # rec-1,2,4,5,6,7_14-good-of-31-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240301_134130019011",  # rec-1,2,4,5,6,7_12-good-of-21-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240301_134253515838",  # rec-1,2,4,5,6,7_18-good-of-39-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240301_134418703906",  # rec-1,2,4,5,6,7_12-good-of-22-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240301_134606685159",  # rec-1,2,4,5,6,7_19-good-of-39-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240301_134923569673",  # rec-1,2,4,5,6,7_18-good-of-45-total_Th,[5,2],spkTh,[-6]_EMUsort
        # # Kilosort 4 STD noise, with 20 sorts no duplicates
        # "20240301_115337058337",  # rec-1,2,4,5,6,7_25-good-of-49-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240301_115351211877",  # rec-1,2,4,5,6,7_27-good-of-44-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240301_115351271071",  # rec-1,2,4,5,6,7_24-good-of-46-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240301_115359551091",  # rec-1,2,4,5,6,7_30-good-of-46-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240301_115402220013",  # rec-1,2,4,5,6,7_37-good-of-68-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240301_115405777742",  # rec-1,2,4,5,6,7_16-good-of-31-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240301_115508833978",  # rec-1,2,4,5,6,7_37-good-of-84-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240301_115516319850",  # rec-1,2,4,5,6,7_28-good-of-60-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240301_115617904756",  # rec-1,2,4,5,6,7_25-good-of-49-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240301_115629846826",  # rec-1,2,4,5,6,7_15-good-of-24-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240301_115653576343",  # rec-1,2,4,5,6,7_22-good-of-42-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240301_115700367145",  # rec-1,2,4,5,6,7_21-good-of-27-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240301_115713962466",  # rec-1,2,4,5,6,7_26-good-of-53-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240301_115747240342",  # rec-1,2,4,5,6,7_28-good-of-38-total_Th,[2,1],spkTh,-6_vanilla_KS $$$
        # "20240301_115814659680",  # rec-1,2,4,5,6,7_40-good-of-82-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240301_115831978837",  # rec-1,2,4,5,6,7_17-good-of-33-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240301_115834981387",  # rec-1,2,4,5,6,7_31-good-of-56-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240301_115846151964",  # rec-1,2,4,5,6,7_26-good-of-46-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240301_115856230111",  # rec-1,2,4,5,6,7_20-good-of-34-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240301_120015026330",  # rec-1,2,4,5,6,7_26-good-of-41-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # # EMUsort 8 STD noise, with 20 sorts no duplicates
        # "20240301_141319996376",  # rec-1,2,4,5,6,7_16-good-of-28-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240301_141400483806",  # rec-1,2,4,5,6,7_11-good-of-22-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240301_141452929209",  # rec-1,2,4,5,6,7_12-good-of-24-total_Th,[2,1],spkTh,[-6]_EMUsort $$$
        # "20240301_141546426474",  # rec-1,2,4,5,6,7_10-good-of-22-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240301_141703230392",  # rec-1,2,4,5,6,7_18-good-of-28-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240301_141739817508",  # rec-1,2,4,5,6,7_21-good-of-32-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240301_141745897621",  # rec-1,2,4,5,6,7_16-good-of-34-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240301_141758038505",  # rec-1,2,4,5,6,7_17-good-of-28-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240301_142144149370",  # rec-1,2,4,5,6,7_18-good-of-36-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240301_142203248993",  # rec-1,2,4,5,6,7_21-good-of-35-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240301_142357295835",  # rec-1,2,4,5,6,7_13-good-of-22-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240301_142555300616",  # rec-1,2,4,5,6,7_17-good-of-26-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240301_142622444864",  # rec-1,2,4,5,6,7_10-good-of-20-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240301_142740496091",  # rec-1,2,4,5,6,7_15-good-of-24-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240301_142750318231",  # rec-1,2,4,5,6,7_15-good-of-25-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240301_142758435169",  # rec-1,2,4,5,6,7_15-good-of-29-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240301_142810752158",  # rec-1,2,4,5,6,7_17-good-of-26-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240301_143046339605",  # rec-1,2,4,5,6,7_12-good-of-20-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240301_143342883293",  # rec-1,2,4,5,6,7_17-good-of-32-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240301_143347022692",  # rec-1,2,4,5,6,7_20-good-of-34-total_Th,[5,2],spkTh,[-6]_EMUsort
        # # Kilosort 8 STD noise, with 20 sorts no duplicates
        # "20240301_120712311011",  # rec-1,2,4,5,6,7_14-good-of-28-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240301_120719121278",  # rec-1,2,4,5,6,7_19-good-of-31-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240301_120739497754",  # rec-1,2,4,5,6,7_25-good-of-44-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240301_120740044454",  # rec-1,2,4,5,6,7_27-good-of-42-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240301_120741522617",  # rec-1,2,4,5,6,7_20-good-of-38-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240301_120742255011",  # rec-1,2,4,5,6,7_24-good-of-40-total_Th,[5,2],spkTh,-9_vanilla_KS $$$
        # "20240301_120801940277",  # rec-1,2,4,5,6,7_28-good-of-44-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240301_120822929452",  # rec-1,2,4,5,6,7_25-good-of-40-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240301_120927758674",  # rec-1,2,4,5,6,7_20-good-of-30-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240301_120958769658",  # rec-1,2,4,5,6,7_23-good-of-40-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240301_121036967635",  # rec-1,2,4,5,6,7_18-good-of-31-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240301_121043260029",  # rec-1,2,4,5,6,7_25-good-of-43-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240301_121043822784",  # rec-1,2,4,5,6,7_17-good-of-37-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240301_121056507565",  # rec-1,2,4,5,6,7_19-good-of-40-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240301_121103898879",  # rec-1,2,4,5,6,7_32-good-of-54-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240301_121130778317",  # rec-1,2,4,5,6,7_18-good-of-32-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240301_121140397332",  # rec-1,2,4,5,6,7_35-good-of-65-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240301_121214614314",  # rec-1,2,4,5,6,7_36-good-of-53-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240301_121246595172",  # rec-1,2,4,5,6,7_25-good-of-45-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240301_121303961042",  # rec-1,2,4,5,6,7_34-good-of-56-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # # EMUsort 16 STD noise, with 20 sorts no duplicates
        # "20240302_112702002964",  # rec-1,2,4,5,6,7_13-good-of-16-total_Th,[10,4],spkTh,[-3]_EMUsort $$$
        # "20240302_112803188533",  # rec-1,2,4,5,6,7_11-good-of-21-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240302_113052102055",  # rec-1,2,4,5,6,7_20-good-of-30-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240302_113100156017",  # rec-1,2,4,5,6,7_22-good-of-46-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240302_113108763570",  # rec-1,2,4,5,6,7_20-good-of-32-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240302_113115083892",  # rec-1,2,4,5,6,7_20-good-of-31-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240302_113235066750",  # rec-1,2,4,5,6,7_20-good-of-39-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240302_113239839208",  # rec-1,2,4,5,6,7_19-good-of-35-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240302_113340469772",  # rec-1,2,4,5,6,7_16-good-of-23-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240302_113531500337",  # rec-1,2,4,5,6,7_14-good-of-25-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240302_113920671723",  # rec-1,2,4,5,6,7_17-good-of-30-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240302_113930237690",  # rec-1,2,4,5,6,7_14-good-of-30-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240302_113956559443",  # rec-1,2,4,5,6,7_20-good-of-36-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240302_114000257286",  # rec-1,2,4,5,6,7_17-good-of-27-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240302_114114545473",  # rec-1,2,4,5,6,7_16-good-of-27-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240302_114147865626",  # rec-1,2,4,5,6,7_20-good-of-37-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240302_114215528817",  # rec-1,2,4,5,6,7_21-good-of-36-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240302_114240957772",  # rec-1,2,4,5,6,7_19-good-of-34-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240302_114436468337",  # rec-1,2,4,5,6,7_15-good-of-26-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240302_114510985387",  # rec-1,2,4,5,6,7_18-good-of-34-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # # Kilosort 16 STD noise, with 20 sorts no duplicates
        # "20240302_110113407629",  # rec-1,2,4,5,6,7_12-good-of-19-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240302_110134852106",  # rec-1,2,4,5,6,7_18-good-of-33-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240302_110157204706",  # rec-1,2,4,5,6,7_22-good-of-36-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240302_110157818862",  # rec-1,2,4,5,6,7_24-good-of-39-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240302_110201325581",  # rec-1,2,4,5,6,7_23-good-of-34-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240302_110201471445",  # rec-1,2,4,5,6,7_20-good-of-25-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240302_110223918284",  # rec-1,2,4,5,6,7_22-good-of-38-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240302_110232775545",  # rec-1,2,4,5,6,7_19-good-of-40-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240302_110324070063",  # rec-1,2,4,5,6,7_16-good-of-25-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240302_110356512941",  # rec-1,2,4,5,6,7_19-good-of-28-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240302_110446463905",  # rec-1,2,4,5,6,7_22-good-of-34-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240302_110454493915",  # rec-1,2,4,5,6,7_16-good-of-26-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240302_110454795089",  # rec-1,2,4,5,6,7_17-good-of-27-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240302_110506073784",  # rec-1,2,4,5,6,7_22-good-of-35-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240302_110523397034",  # rec-1,2,4,5,6,7_18-good-of-25-total_Th,[2,1],spkTh,-6_vanilla_KS
        # "20240302_110526749376",  # rec-1,2,4,5,6,7_22-good-of-37-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240302_110535665413",  # rec-1,2,4,5,6,7_24-good-of-33-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240302_110559659069",  # rec-1,2,4,5,6,7_22-good-of-34-total_Th,[10,4],spkTh,-6_vanilla_KS $$$
        # "20240302_110642581922",  # rec-1,2,4,5,6,7_17-good-of-26-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # "20240302_110643165194",  # rec-1,2,4,5,6,7_24-good-of-41-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # # EMUsort 32 STD noise, with 20 sorts no duplicates
        # ## "TooFew_20240302_122050506516",  # rec-1,2,4,5,6,7_2-good-of-7-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240302_122105112048",  # rec-1,2,4,5,6,7_7-good-of-13-total_Th,[10,4],spkTh,[-3]_EMUsort
        # ## "TooFew_20240302_122124709200",  # rec-1,2,4,5,6,7_4-good-of-7-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240302_122228405573",  # rec-1,2,4,5,6,7_9-good-of-17-total_Th,[7,3],spkTh,[-6]_EMUsort
        # ## "TooFew_20240302_122228522631",  # rec-1,2,4,5,6,7_4-good-of-9-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240302_122307404512",  # rec-1,2,4,5,6,7_6-good-of-16-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240302_122316621141",  # rec-1,2,4,5,6,7_7-good-of-15-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240302_122414910693",  # rec-1,2,4,5,6,7_6-good-of-17-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240302_122624894941",  # rec-1,2,4,5,6,7_5-good-of-14-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # ## "TooFew_20240302_122634992494",  # rec-1,2,4,5,6,7_4-good-of-9-total_Th,[5,2],spkTh,[-3]_EMUsort
        # ## "TooFew_20240302_122646828158",  # rec-1,2,4,5,6,7_2-good-of-6-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240302_122658762478",  # rec-1,2,4,5,6,7_3-good-of-12-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240302_122705882104",  # rec-1,2,4,5,6,7_7-good-of-19-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240302_122709729708",  # rec-1,2,4,5,6,7_6-good-of-12-total_Th,[7,3],spkTh,[-9]_EMUsort
        # ## "TooFew_20240302_122747883648",  # rec-1,2,4,5,6,7_2-good-of-6-total_Th,[2,1],spkTh,[-9]_EMUsort
        # ## "TooFew_20240302_122918633973",  # rec-1,2,4,5,6,7_3-good-of-6-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240302_122925229679",  # rec-1,2,4,5,6,7_5-good-of-19-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240302_123035464951",  # rec-1,2,4,5,6,7_5-good-of-14-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240302_123049010200",  # rec-1,2,4,5,6,7_8-good-of-14-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240302_123117787026",  # rec-1,2,4,5,6,7_5-good-of-13-total_Th,[7,3],spkTh,[-3,-6]"_EMUsort
        # # # Kilosort 32 STD noise, with 20 sorts no duplicates
        # ## "TooFew_20240302_120939352682",  # rec-1,2,4,5,6,7_0-good-of-3-total_Th,[5,2],spkTh,-9_vanilla_KS
        # ## "TooFew_20240302_120939491097",  # rec-1,2,4,5,6,7_4-good-of-9-total_Th,[10,4],spkTh,-3_vanilla_KS
        # ## "TooFew_20240302_120939955560",  # rec-1,2,4,5,6,7_4-good-of-9-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # ## "TooFew_20240302_120941163246",  # rec-1,2,4,5,6,7_2-good-of-6-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # ## "TooFew_20240302_120945737221",  # rec-1,2,4,5,6,7_0-good-of-7-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # ## "TooFew_20240302_120946268209",  # rec-1,2,4,5,6,7_2-good-of-8-total_Th,[7,3],spkTh,-9_vanilla_KS
        # ## "TooFew_20240302_120946271887",  # rec-1,2,4,5,6,7_2-good-of-6-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # ## "TooFew_20240302_120947444159",  # rec-1,2,4,5,6,7_4-good-of-8-total_Th,[5,2],spkTh,-6_vanilla_KS
        # ## "TooFew_20240302_121121351437",  # rec-1,2,4,5,6,7_0-good-of-3-total_Th,[10,4],spkTh,-9_vanilla_KS
        # ## "TooFew_20240302_121127271809",  # rec-1,2,4,5,6,7_5-good-of-8-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # ## "TooFew_20240302_121129703127",  # rec-1,2,4,5,6,7_0-good-of-6-total_Th,[7,3],spkTh,-6_vanilla_KS
        # ## "TooFew_20240302_121134652070",  # rec-1,2,4,5,6,7_0-good-of-3-total_Th,[2,1],spkTh,-9_vanilla_KS
        # ## "TooFew_20240302_121137225230",  # rec-1,2,4,5,6,7_1-good-of-4-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # ## "TooFew_20240302_121140700946",  # rec-1,2,4,5,6,7_3-good-of-7-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240302_121142925661",  # rec-1,2,4,5,6,7_3-good-of-11-total_Th,[2,1],spkTh,-6_vanilla_KS
        # ## "TooFew_20240302_121226554550",  # rec-1,2,4,5,6,7_0-good-of-3-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240302_121246276326",  # rec-1,2,4,5,6,7_5-good-of-10-total_Th,[10,4],spkTh,-6_vanilla_KS
        # ## "TooFew_20240302_121246849496",  # rec-1,2,4,5,6,7_0-good-of-4-total_Th,[7,3],spkTh,-3_vanilla_KS
        # ## "TooFew_20240302_121251662558",  # rec-1,2,4,5,6,7_0-good-of-4-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # ## "TooFew_20240302_121305322001",  # rec-1,2,4,5,6,7_2-good-of-5-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # >###########################################################################################
        ### # EMUsort, Konstantin 2020 simulation 10/18 MUs detectable reconstruction
        # "20240314_133031478643",  # rec-1_23-good-of-44-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240314_133037442256",  # rec-1_31-good-of-63-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240314_133043455404",  # rec-1_28-good-of-51-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240314_133050202064",  # rec-1_35-good-of-62-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240314_133056657041",  # rec-1_40-good-of-80-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240314_133218549973",  # rec-1_55-good-of-87-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240314_133242957398",  # rec-1_65-good-of-116-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240314_133421858859",  # rec-1_27-good-of-52-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240314_133441670071",  # rec-1_27-good-of-54-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240314_133442683512",  # rec-1_26-good-of-47-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240314_133535163377",  # rec-1_55-good-of-93-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240314_133620578243",  # rec-1_35-good-of-80-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240314_133719454359",  # rec-1_36-good-of-70-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240314_133815205560",  # rec-1_20-good-of-46-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240314_133850117335",  # rec-1_25-good-of-60-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240314_133852009547",  # rec-1_28-good-of-57-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240314_133941186486",  # rec-1_72-good-of-112-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240314_134036123142",  # rec-1_54-good-of-90-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240314_134104220405",  # rec-1_41-good-of-79-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240314_134431073784",  # rec-1_52-good-of-105-total_Th,[2,1],spkTh,[-9]_EMUsort
        ### # Kilosort, Konstantin 2020 simulation 10/18 MUs detectable reconstruction
        # "20240314_130436324157",  # rec-1_45-good-of-82-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240314_130436535281",  # rec-1_53-good-of-93-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240314_130439530174",  # rec-1_48-good-of-82-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240314_130441924825",  # rec-1_61-good-of-104-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240314_130515824181",  # rec-1_76-good-of-148-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240314_130637016676",  # rec-1_30-good-of-57-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240314_130640813332",  # rec-1_109-good-of-224-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240314_130657414489",  # rec-1_59-good-of-85-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240314_130700582218",  # rec-1_109-good-of-206-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240314_130709080675",  # rec-1_55-good-of-102-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240314_130732540178",  # rec-1_70-good-of-135-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240314_130750627529",  # rec-1_69-good-of-113-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240314_130850692336",  # rec-1_36-good-of-56-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240314_130859228057",  # rec-1_63-good-of-103-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240314_130920017501",  # rec-1_52-good-of-90-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240314_130953221015",  # rec-1_59-good-of-100-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # "20240314_131022913817",  # rec-1_68-good-of-125-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240314_131059650387",  # rec-1_99-good-of-228-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240314_131106869584",  # rec-1_120-good-of-233-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240314_131445242868",  # rec-1_115-good-of-241-total_Th,[2,1],spkTh,-6_vanilla_KS
        # # EMUsort, Konstantin 2020 simulation 11/18 MUs detectable, detectable reconstruction
        # "20240321_124209509936",  # rec-1_27-good-of-52-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240321_124347474108",  # rec-1_52-good-of-102-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240321_124356310172",  # rec-1_52-good-of-97-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240321_124525653167",  # rec-1_82-good-of-137-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240321_124527914172",  # rec-1_61-good-of-111-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240321_124611775942",  # rec-1_72-good-of-134-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240321_124755737983",  # rec-1_38-good-of-68-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240321_124917574479",  # rec-1_37-good-of-75-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240321_125018953591",  # rec-1_94-good-of-185-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240321_125141544407",  # rec-1_45-good-of-85-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240321_125402791107",  # rec-1_26-good-of-49-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240321_125415943412",  # rec-1_70-good-of-123-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240321_125429386729",  # rec-1_92-good-of-174-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240321_125451209827",  # rec-1_123-good-of-236-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240321_125759609494",  # rec-1_85-good-of-158-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240321_125854033708",  # rec-1_64-good-of-120-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240321_130035330897",  # rec-1_109-good-of-231-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240321_130158482804",  # rec-1_82-good-of-178-total_Th,[2,1],spkTh,[-9]_EMUsort
        # "20240321_130201817036",  # rec-1_72-good-of-155-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240321_131048018338",  # rec-1_131-good-of-303-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # # Kilosort, Konstantin 2020 simulation 11/18 MUs detectable, detectable reconstruction
        # "20240321_133209692667",  # rec-1_57-good-of-125-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240321_133243583907",  # rec-1_75-good-of-155-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240321_133248601234",  # rec-1_89-good-of-162-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240321_133309025779",  # rec-1_95-good-of-188-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240321_133316963669",  # rec-1_100-good-of-190-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240321_133324488479",  # rec-1_105-good-of-209-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240321_133553167271",  # rec-1_80-good-of-150-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240321_133558003097",  # rec-1_132-good-of-325-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # "20240321_133636085220",  # rec-1_145-good-of-362-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240321_133636411457",  # rec-1_90-good-of-159-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240321_133701211118",  # rec-1_84-good-of-186-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240321_133821043152",  # rec-1_104-good-of-211-total_Th,[5,2],spkTh,-3_vanilla_KS
        # "20240321_133833413527",  # rec-1_109-good-of-221-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # "20240321_133954304194",  # rec-1_74-good-of-135-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240321_134010603543",  # rec-1_80-good-of-188-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240321_134054976498",  # rec-1_84-good-of-178-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240321_134101067184",  # rec-1_153-good-of-353-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240321_134204300555",  # rec-1_87-good-of-186-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # "20240321_134253644348",  # rec-1_150-good-of-342-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240321_134308362350",  # rec-1_146-good-of-363-total_Th,[2,1],spkTh,-6_vanilla_KS
        # # # EMUsort, Konstantin 2020 simulation 8/9 MUs detectable, detectable reconstruction, 08CH
        # "20240401_143451686926",  # rec-1_18-good-of-27-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240401_143454991388",  # rec-1_13-good-of-26-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240401_143501570809",  # rec-1_14-good-of-25-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240401_143527052851",  # rec-1_23-good-of-37-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240401_143546714288",  # rec-1_26-good-of-47-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240401_143824317076",  # rec-1_14-good-of-21-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240401_143838536818",  # rec-1_64-good-of-117-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240401_143845329427",  # rec-1_16-good-of-35-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240401_143919097698",  # rec-1_19-good-of-43-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240401_143926410519",  # rec-1_19-good-of-33-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240401_143947277034",  # rec-1_19-good-of-35-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240401_144240331885",  # rec-1_21-good-of-39-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240401_144240342857",  # rec-1_23-good-of-37-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240401_144249514326",  # rec-1_21-good-of-46-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240401_144255771593",  # rec-1_13-good-of-27-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240401_144306570488",  # rec-1_18-good-of-30-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240401_144316446726",  # rec-1_14-good-of-25-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240401_144332718017",  # rec-1_77-good-of-216-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240401_144801044062",  # rec-1_34-good-of-62-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240401_145556888106",  # rec-1_41-good-of-84-total_Th,[2,1],spkTh,[-9]_EMUsort
        # # # EMUsort, Konstantin 2020 simulation 8/9 MUs detectable, detectable reconstruction, 16CH
        # "20240327_202524918615",  # rec-1_16-good-of-24-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240327_202534934565",  # rec-1_16-good-of-32-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240327_202538819805",  # rec-1_17-good-of-29-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240327_202538954207",  # rec-1_16-good-of-28-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240327_202551304503",  # rec-1_18-good-of-27-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240327_202818195257",  # rec-1_10-good-of-19-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240327_202857555716",  # rec-1_17-good-of-28-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240327_202921488830",  # rec-1_28-good-of-35-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240327_203059286948",  # rec-1_46-good-of-77-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240327_203129813399",  # rec-1_52-good-of-91-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240327_203135535338",  # rec-1_15-good-of-26-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240327_203241326354",  # rec-1_25-good-of-42-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240327_203318218134",  # rec-1_30-good-of-45-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240327_203508156240",  # rec-1_31-good-of-47-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240327_203543388847",  # rec-1_48-good-of-82-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240327_203730973208",  # rec-1_95-good-of-180-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240327_203942197910",  # rec-1_108-good-of-241-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240327_205742347003",  # rec-1_124-good-of-250-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240327_205841679532",  # rec-1_117-good-of-258-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240327_211546636381",  # rec-1_112-good-of-230-total_Th,[2,1],spkTh,[-9]_EMUsort
        # # Kilosort, Konstantin 2020 simulation 8/9 MUs detectable, detectable reconstruction, 16CH
        # "20240327_165217236323",  # rec-1_50-good-of-86-total_Th,[10,4],spkTh,-7.5_vanilla_KS
        # "20240327_165258391997",  # rec-1_73-good-of-129-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240327_165307736652",  # rec-1_71-good-of-145-total_Th,[10,4],spkTh,-3_vanilla_KS
        # "20240327_165315704587",  # rec-1_74-good-of-145-total_Th,[7,3],spkTh,-4.5_vanilla_KS
        # "20240327_165334240992",  # rec-1_81-good-of-162-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240327_165439325375",  # rec-1_45-good-of-91-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240327_165538286205",  # rec-1_64-good-of-99-total_Th,[10,4],spkTh,-4.5_vanilla_KS
        # "20240327_165550206411",  # rec-1_65-good-of-120-total_Th,[7,3],spkTh,-6_vanilla_KS
        # # "20240327_165553696807",  # rec-1_106-good-of-228-total_Th,[2,1],spkTh,-7.5_vanilla_KS
        # "20240327_165715859555",  # rec-1_77-good-of-172-total_Th,[5,2],spkTh,-7.5_vanilla_KS
        # # "20240327_165718744046",  # rec-1_95-good-of-214-total_Th,[5,2],spkTh,-3_vanilla_KS
        # # "20240327_165738255526",  # rec-1_148-good-of-337-total_Th,[2,1],spkTh,-3_vanilla_KS
        # "20240327_165754033485",  # rec-1_50-good-of-86-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240327_165807022550",  # rec-1_83-good-of-167-total_Th,[7,3],spkTh,-3_vanilla_KS
        # "20240327_165840929214",  # rec-1_69-good-of-127-total_Th,[7,3],spkTh,-7.5_vanilla_KS
        # "20240327_170052113611",  # rec-1_99-good-of-181-total_Th,[5,2],spkTh,-9_vanilla_KS
        # # "20240327_170125444286",  # rec-1_111-good-of-213-total_Th,[5,2],spkTh,-4.5_vanilla_KS
        # # "20240327_170151247601",  # rec-1_127-good-of-304-total_Th,[2,1],spkTh,-9_vanilla_KS
        # # "20240327_170431682694",  # rec-1_124-good-of-325-total_Th,[2,1],spkTh,-4.5_vanilla_KS
        # # "20240327_171000259112",  # rec-1_125-good-of-288-total_Th,[2,1],spkTh,-6_vanilla_KS
        ############################################################################################
        #### Below are with new 16 channel, triple rat dataset
        # simulated20231219:
        # "20231220_180513756759"  # SNR-400-constant_jitter-0std_files-11, vanilla Kilosort, Th=[10,4], spkTh=[-6]
        # "20231220_172352030313"  # SNR-400-constant_jitter-0std_files-11, EMUsort, Th=[5,2], spkTh=[-3,-6]
        ############################################################################################
        #### Below are for the monkey dataset (8CH)
        # EMU=[0.675, 0.868, 0.817, 0.357, 0.813, 0.818,0.726,0.882,0.793] # mean accuracies, grand mean: 0.74989
        # KS=[0.688,0.652,0.833,0.811,0.810,0.764,0.830,0.616,0.878,0.763,0.804,0.890] # mean accuracies, grand mean: 0.77825
        # EMUsort with comparable grid search, with sgolay filter to align templates (makes performance worse)
        # "20240206_180600872332",  # rec-1_2-good-of-3-total_Th,[10,4],spkTh,[-6]_EMUsort # too few
        # "20240206_180621637339",  # rec-1_3-good-of-3-total_Th,[10,4],spkTh,[-6,-9]_EMUsort # too few
        # "20240206_180658042556",  # rec-1_10-good-of-16-total_Th,[7,3],spkTh,[-3,-6]_EMUsort # Average accuracy: 0.675 +/- 0.354
        # "20240206_180802824443",  # rec-1_25-good-of-44-total_Th,[5,2],spkTh,[-6]_EMUsort # Average accuracy: 0.868 +/- 0.094
        # "20240206_180806779104",  # rec-1_25-good-of-42-total_Th,[5,2],spkTh,[-6,-9]_EMUsort # Average accuracy: 0.817 +/- 0.160
        # "20240206_180856961919",  # rec-1_5-good-of-6-total_Th,[10,4],spkTh,[-3,-6]_EMUsort # Average accuracy: 0.357 +/- 0.395
        # "20240206_180954426237",  # rec-1_23-good-of-38-total_Th,[2,1],spkTh,[-3,-6]_EMUsort # Average accuracy: 0.813 +/- 0.126
        # "20240206_181007303651",  # rec-1_9-good-of-10-total_Th,[7,3],spkTh,[-6,-9]_EMUsort # Average accuracy: 0.818 +/- 0.176
        # "20240206_181234362303",  # rec-1_25-good-of-41-total_Th,[5,2],spkTh,[-3,-6]_EMUsort # Average accuracy: 0.726 +/- 0.173
        # "20240206_181342142506",  # rec-1_23-good-of-35-total_Th,[2,1],spkTh,[-6]_EMUsort # Average accuracy: 0.882 +/- 0.101 ### BEST EMU
        # "20240206_181510823697",  # rec-1_30-good-of-56-total_Th,[2,1],spkTh,[-6,-9]_EMUsort # Average accuracy: 0.793 +/- 0.187
        # ## Kilosort with comparable grid search, original dataset
        # "20240207_164752921257",  # rec-1_11-good-of-16-total_Th,[10,4],spkTh,-9_vanilla_KS # Average accuracy: 0.688 +/- 0.375
        # "20240207_164803728144",  # rec-1_16-good-of-24-total_Th,[10,4],spkTh,-3_vanilla_KS # Average accuracy: 0.652 +/- 0.473
        # "20240207_164826157090",  # rec-1_24-good-of-37-total_Th,[5,2],spkTh,-9_vanilla_KS # Average accuracy: 0.833 +/- 0.126
        # "20240207_164832762566",  # rec-1_32-good-of-49-total_Th,[7,3],spkTh,-6_vanilla_KS # Average accuracy: 0.811 +/- 0.174
        # "20240207_164840302634",  # rec-1_27-good-of-47-total_Th,[5,2],spkTh,-3_vanilla_KS # Average accuracy: 0.810 +/- 0.134
        # "20240207_164923767820",  # rec-1_27-good-of-51-total_Th,[2,1],spkTh,-6_vanilla_KS # Average accuracy: 0.764 +/- 0.145
        # "20240207_164943630654",  # rec-1_20-good-of-28-total_Th,[7,3],spkTh,-3_vanilla_KS # Average accuracy: 0.830 +/- 0.092
        # "20240207_164957316623",  # rec-1_16-good-of-26-total_Th,[10,4],spkTh,-6_vanilla_KS # Average accuracy: 0.616 +/- 0.437
        # "20240207_165018647090",  # rec-1_19-good-of-24-total_Th,[7,3],spkTh,-9_vanilla_KS # Average accuracy: 0.878 +/- 0.130
        # "20240207_165106699721",  # rec-1_31-good-of-53-total_Th,[2,1],spkTh,-3_vanilla_KS # Average accuracy: 0.763 +/- 0.175
        # "20240207_165116929820",  # rec-1_43-good-of-70-total_Th,[5,2],spkTh,-6_vanilla_KS # Average accuracy: 0.804 +/- 0.134
        # "20240207_165147332421",  # rec-1_39-good-of-68-total_Th,[2,1],spkTh,-9_vanilla_KS # Average accuracy: 0.890 +/- 0.083 ### BEST KS
        # "20240216_191316750227",  # rec-1_10-good-of-13-total_Th,[10,4],spkTh,-9_vanilla_KS
        # "20240216_191326377677",  # rec-1_15-good-of-25-total_Th,[10,4],spkTh,-6_vanilla_KS
        # "20240216_191327893194",  # rec-1_18-good-of-20-total_Th,[7,3],spkTh,-9_vanilla_KS
        # "20240216_191355111675",  # rec-1_27-good-of-41-total_Th,[5,2],spkTh,-9_vanilla_KS
        # "20240216_191402604654",  # rec-1_29-good-of-55-total_Th,[7,3],spkTh,-6_vanilla_KS
        # "20240216_191419005820",  # rec-1_38-good-of-65-total_Th,[5,2],spkTh,-6_vanilla_KS
        # "20240216_191419080866",  # rec-1_40-good-of-70-total_Th,[2,1],spkTh,-9_vanilla_KS
        # "20240216_191457383864",  # rec-1_42-good-of-89-total_Th,[2,1],spkTh,-6_vanilla_KS
        ## EMUsort with comparable grid search, 100 noise, with sgolay filter to align templates (makes performance worse)
        # "20240209_001234411226",  # rec-1_4-good-of-4-total_Th,[10,4],spkTh,[-9]_EMUsort # too few
        # "20240209_001249124363", # rec-1_3-good-of-3-total_Th,[10,4],spkTh,[-3]_EMUsort # too few
        # "20240209_001342251355",  # rec-1_8-good-of-11-total_Th,[7,3],spkTh,[-6]_EMUsort # Average accuracy: 0.718 +/- 0.353
        # "20240209_001445351930",  # rec-1_19-good-of-34-total_Th,[5,2],spkTh,[-9]_EMUsort #
        # "20240209_001538771592",  # rec-1_24-good-of-41-total_Th,[5,2],spkTh,[-3]_EMUsort #
        # "20240209_001550701418",  # rec-1_7-good-of-9-total_Th,[10,4],spkTh,[-6]_EMUsort #
        # "20240209_001552552507",  # rec-1_14-good-of-19-total_Th,[7,3],spkTh,[-3]_EMUsort #
        # "20240209_001644282197",  # rec-1_8-good-of-11-total_Th,[7,3],spkTh,[-9]_EMUsort #
        # "20240209_001806825843",  # rec-1_39-good-of-66-total_Th,[2,1],spkTh,[-6]_EMUsort #
        # "20240209_002012987541",  # rec-1_23-good-of-45-total_Th,[5,2],spkTh,[-6]_EMUsort #
        # "20240209_002053696849",  # rec-1_37-good-of-61-total_Th,[2,1],spkTh,[-3]_EMUsort #
        # "20240209_002354264296",  # rec-1_29-good-of-51-total_Th,[2,1],spkTh,[-9]_EMUsort #
        ## EMUsort with comparable grid search, original dataset, without sgolay filter
        # "20240216_192730211679",  # rec-1_3-good-of-3-total_Th,[10,4],spkTh,[-3]_EMUsort
        # "20240216_192807887876",  # rec-1_4-good-of-8-total_Th,[10,4],spkTh,[-3,-6]_EMUsort
        # "20240216_192840766905",  # rec-1_12-good-of-19-total_Th,[7,3],spkTh,[-6]_EMUsort
        # "20240216_192854939238",  # rec-1_13-good-of-16-total_Th,[7,3],spkTh,[-6,-9]_EMUsort
        # "20240216_193003232681",  # rec-1_27-good-of-41-total_Th,[5,2],spkTh,[-9]_EMUsort
        # "20240216_193044965859",  # rec-1_21-good-of-42-total_Th,[5,2],spkTh,[-6,-9]_EMUsort
        # "20240216_193101192432",  # rec-1_5-good-of-8-total_Th,[10,4],spkTh,[-6]_EMUsort
        # "20240216_193148919140",  # rec-1_4-good-of-4-total_Th,[10,4],spkTh,[-6,-9]_EMUsort
        # "20240216_193218301080",  # rec-1_9-good-of-10-total_Th,[7,3],spkTh,[-9]_EMUsort
        # "20240216_193355807410",  # rec-1_25-good-of-53-total_Th,[2,1],spkTh,[-6]_EMUsort
        # "20240216_193359569907",  # rec-1_7-good-of-8-total_Th,[10,4],spkTh,[-9]_EMUsort
        # "20240216_193405597538",  # rec-1_27-good-of-48-total_Th,[2,1],spkTh,[-3,-6]_EMUsort
        # "20240216_193508998821",  # rec-1_8-good-of-10-total_Th,[7,3],spkTh,[-3]_EMUsort
        # "20240216_193515712692",  # rec-1_31-good-of-56-total_Th,[5,2],spkTh,[-3]_EMUsort
        # "20240216_193646480820",  # rec-1_30-good-of-47-total_Th,[5,2],spkTh,[-3,-6]_EMUsort
        # "20240216_193651613970",  # rec-1_7-good-of-11-total_Th,[7,3],spkTh,[-3,-6]_EMUsort
        # "20240216_193740433882",  # rec-1_30-good-of-43-total_Th,[2,1],spkTh,[-3]_EMUsort
        # "20240216_193932148031",  # rec-1_24-good-of-39-total_Th,[2,1],spkTh,[-6,-9]_EMUsort
        # "20240216_194008953376",  # rec-1_28-good-of-43-total_Th,[5,2],spkTh,[-6]_EMUsort
        # "20240216_194021792336",  # rec-1_26-good-of-46-total_Th,[2,1],spkTh,[-9]_EMUsort
        ## Kilosort4 testing with 8 channel dataset, 04STD
        # "20240517_152329188318",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[6]_KS4
        # "20240517_152350963799",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[3]_KS4
        # "20240517_152411079877",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[4.5]_KS4
        # "20240517_152431413903",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[7.5]_KS4  $$$
        # "20240517_152451566303",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[9]_KS4
        # "20240517_152523509477",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[6]_KS4
        # "20240517_152602016091",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[3]_KS4
        # "20240517_152638621044",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[4.5]_KS4
        # "20240517_152711810938",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[7.5]_KS4
        # "20240517_152742714336",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[9]_KS4
        # "20240517_152822693087",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[6]_KS4
        # "20240517_152909981442",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[3]_KS4
        # "20240517_152952088600",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[4.5]_KS4
        # "20240517_153031807232",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[7.5]_KS4
        # "20240517_153110320621",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[9]_KS4
        # "20240517_153213394517",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[6]_KS4
        # "20240517_153325618329",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[3]_KS4
        # "20240517_153433577999",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[4.5]_KS4
        # "20240517_153534181572",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[7.5]_KS4
        # "20240517_153634868068",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[9]_KS4
        # "20240517_153829175073",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[6]_KS4
        # "20240517_154033999597",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[3]_KS4
        # "20240517_154231173231",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[4.5]_KS4
        # "20240517_154420186873",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[7.5]_KS4
        # "20240517_154608113816",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[9]_KS4
        ## Kilosort4 testing with 8 channel dataset, 08STD
        # "20240517_114933273272",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[6]_KS4  $$$
        # "20240517_114954325855",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[3]_KS4
        # "20240517_115013790102",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[4.5]_KS4
        # "20240517_115032946326",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[7.5]_KS4
        # "20240517_115051947408",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[9]_KS4
        # "20240517_115127066366",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[6]_KS4
        # "20240517_115200227220",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[3]_KS4
        # "20240517_115233247950",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[4.5]_KS4
        # "20240517_115303686236",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[7.5]_KS4
        # "20240517_115335904473",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[9]_KS4
        # "20240517_115417581883",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[6]_KS4
        # "20240517_115504061679",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[3]_KS4
        # "20240517_115548446113",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[4.5]_KS4
        # "20240517_115627032945",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[7.5]_KS4
        # "20240517_115705829776",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[9]_KS4
        # "20240517_115810719367",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[6]_KS4
        # "20240517_115922217679",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[3]_KS4
        # "20240517_120032320483",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[4.5]_KS4
        # "20240517_120136454300",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[7.5]_KS4
        # "20240517_120240022229",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[9]_KS4
        # "20240517_120442549150",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[6]_KS4
        # "20240517_120650579202",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[3]_KS4
        # "20240517_120854052506",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[4.5]_KS4
        # "20240517_121057146542",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[7.5]_KS4
        # "20240517_121257574046",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[9]_KS4
        ## Kilosort4 testing with 8 channel dataset, 16STD
        # "20240506_180000000000",  # default settings
        # "20240508_201259971096",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[6]_KS4  $$$
        # "20240508_201319874725",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[3]_KS4
        # "20240508_201339072807",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[9]_KS4
        # "20240508_201412820291",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[6]_KS4
        # "20240508_201449760292",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[3]_KS4
        # "20240508_201523962938",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[9]_KS4
        # "20240508_201610602826",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[6]_KS4
        # "20240508_201700527644",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[3]_KS4
        # "20240508_201747653121",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[9]_KS4
        # "20240508_201904901109",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[6]_KS4
        # "20240508_202027853995",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[3]_KS4
        # "20240508_202145893192",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[9]_KS4
        # "20240508_202400082406",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[6]_KS4
        # "20240508_202626611651",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[3]_KS4
        # "20240508_202849312799",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[9]_KS4
        ## Kilosort4 testing with 8 channel dataset, 32STD
        # "20240517_140513143227",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[6]_KS4
        # "20240517_140533892693",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[3]_KS4
        # "20240517_140554966940",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[4.5]_KS4
        # "20240517_140616065741",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[7.5]_KS4
        # "20240517_140636290619",  # rec-1,2,4,5,6,7_Th,[9,8],spkTh,[9]_KS4
        # "20240517_140718799866",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[6]_KS4
        # "20240517_140801661810",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[3]_KS4
        # "20240517_140844020808",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[4.5]_KS4
        # "20240517_140926519551",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[7.5]_KS4
        # "20240517_141008601337",  # rec-1,2,4,5,6,7_Th,[10,4],spkTh,[9]_KS4
        # "20240517_141112174797",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[6]_KS4
        # "20240517_141215972843",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[3]_KS4
        # "20240517_141318064677",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[4.5]_KS4
        # "20240517_141419613775",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[7.5]_KS4
        # "20240517_141523224949",  # rec-1,2,4,5,6,7_Th,[7,3],spkTh,[9]_KS4
        # "20240517_141709653109",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[6]_KS4  $$$
        # "20240517_141856014542",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[3]_KS4
        # "20240517_142040574730",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[4.5]_KS4
        # "20240517_142220933220",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[7.5]_KS4
        # "20240517_142406458528",  # rec-1,2,4,5,6,7_Th,[5,2],spkTh,[9]_KS4
        # "20240517_142651094141",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[6]_KS4
        # "20240517_142922983916",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[3]_KS4
        # "20240517_143208762418",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[4.5]_KS4
        # "20240517_143453298458",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[7.5]_KS4
        # "20240517_143733334027",  # rec-1,2,4,5,6,7_Th,[2,1],spkTh,[9]_KS4
        ## EMUsort_ks4_v1 with 8 channel dataset, 0.00 shape noise
        # "20240612_162143451249",  # Th_10,4_spkTh_6,_EMUsort
        # "20240612_162139766714",  # Th_10,4_spkTh_6,9_EMUsort
        # "20240612_162150905785",  # Th_9,8_spkTh_9,_EMUsort
        # "20240612_162142465959",  # Th_9,8_spkTh_6,_EMUsort
        # "20240612_162131935574",  # Th_10,4_spkTh_3,6_EMUsort
        # "20240612_162147487573",  # Th_9,8_spkTh_3,_EMUsort
        # "20240612_162154221771",  # Th_7,3_spkTh_6,_EMUsort
        # "20240612_162155570255",  # Th_7,3_spkTh_9,_EMUsort
        # "20240612_162138953904",  # Th_10,4_spkTh_9,_EMUsort
        # "20240612_162148789494",  # Th_5,2_spkTh_6,_EMUsort
        # "20240612_162151882880",  # Th_9,8_spkTh_3,6_EMUsort
        # "20240612_162156310600",  # Th_7,3_spkTh_3,6_EMUsort
        # "20240612_162154219415",  # Th_7,3_spkTh_3,_EMUsort
        # "20240612_162148289976",  # Th_5,2_spkTh_9,_EMUsort
        # "20240612_162154205928",  # Th_10,4_spkTh_3,_EMUsort
        # "20240612_162148326324",  # Th_5,2_spkTh_6,9_EMUsort
        # "20240612_162154329563",  # Th_7,3_spkTh_6,9_EMUsort
        # "20240612_162148499755",  # Th_9,8_spkTh_6,9_EMUsort
        # "20240612_162203691845",  # Th_5,2_spkTh_3,6_EMUsort
        # "20240612_162206574759",  # Th_5,2_spkTh_3,_EMUsort
        # "20240612_162205096952",  # Th_2,1_spkTh_6,_EMUsort
        # "20240612_162209655269",  # Th_2,1_spkTh_3,6_EMUsort
        # "20240612_162214057904",  # Th_2,1_spkTh_9,_EMUsort
        # "20240612_162211565554",  # Th_2,1_spkTh_6,9_EMUsort
        # "20240612_162222369356",  # Th_2,1_spkTh_3,_EMUsort
        ## EMUsort_ks4_v1 with 8 channel dataset, 0.75 shape noise
        # "20240612_162355715439",  # Th_10,4_spkTh_6,_EMUsort
        # "20240612_162400262151",  # Th_10,4_spkTh_3,_EMUsort
        # "20240612_162403395197",  # Th_9,8_spkTh_3,6_EMUsort
        # "20240612_162356570854",  # Th_10,4_spkTh_6,9_EMUsort
        # "20240612_162404604213",  # Th_7,3_spkTh_9,_EMUsort
        # "20240612_162404529967",  # Th_9,8_spkTh_3,_EMUsort
        # "20240612_162418702439",  # Th_7,3_spkTh_3,_EMUsort
        # "20240612_162357416605",  # Th_10,4_spkTh_3,6_EMUsort
        # "20240612_162403181521",  # Th_9,8_spkTh_9,_EMUsort
        # "20240612_162354425780",  # Th_9,8_spkTh_6,_EMUsort
        # "20240612_162405451672",  # Th_7,3_spkTh_6,_EMUsort
        # "20240612_162357571829",  # Th_9,8_spkTh_6,9_EMUsort
        # "20240612_162403504099",  # Th_5,2_spkTh_6,_EMUsort
        # "20240612_162415949808",  # Th_5,2_spkTh_3,6_EMUsort
        # "20240612_162403479050",  # Th_10,4_spkTh_9,_EMUsort
        # "20240612_162411610102",  # Th_7,3_spkTh_3,6_EMUsort
        # "20240612_162416537121",  # Th_5,2_spkTh_3,_EMUsort
        # "20240612_162411023381",  # Th_5,2_spkTh_9,_EMUsort
        # "20240612_162411034407",  # Th_7,3_spkTh_6,9_EMUsort
        # "20240612_162407310267",  # Th_5,2_spkTh_6,9_EMUsort
        # "20240612_162421710184",  # Th_2,1_spkTh_3,6_EMUsort
        # "20240612_162420296488",  # Th_2,1_spkTh_6,_EMUsort
        # "20240612_162434554470",  # Th_2,1_spkTh_3,_EMUsort
        # "20240612_162415028926",  # Th_2,1_spkTh_6,9_EMUsort
        # "20240612_162427330280",  # Th_2,1_spkTh_9,_EMUsort
        ## EMUsort_ks4_v1 with 8 channel dataset, 1.50 shape noise
        # "20240612_162755880224",  # Th_9,8_spkTh_6,_EMUsort
        # "20240612_162756150412",  # Th_9,8_spkTh_3,_EMUsort
        # "20240612_162747945179",  # Th_10,4_spkTh_3,_EMUsort
        # "20240612_162749610111",  # Th_10,4_spkTh_6,_EMUsort
        # "20240612_162756772278",  # Th_7,3_spkTh_9,_EMUsort
        # "20240612_162755808166",  # Th_5,2_spkTh_3,_EMUsort
        # "20240612_162758787758",  # Th_7,3_spkTh_6,_EMUsort
        # "20240612_162756324188",  # Th_9,8_spkTh_9,_EMUsort
        # "20240612_162816121535",  # Th_7,3_spkTh_3,_EMUsort
        # "20240612_162741571454",  # Th_10,4_spkTh_9,_EMUsort
        # "20240612_162753276027",  # Th_9,8_spkTh_6,9_EMUsort
        # "20240612_162755653598",  # Th_5,2_spkTh_6,_EMUsort
        # "20240612_162741961706",  # Th_10,4_spkTh_3,6_EMUsort
        # "20240612_162755686295",  # Th_5,2_spkTh_9,_EMUsort
        # "20240612_162744224294",  # Th_9,8_spkTh_3,6_EMUsort
        # "20240612_162741807514",  # Th_10,4_spkTh_6,9_EMUsort
        # "20240612_162758897953",  # Th_7,3_spkTh_3,6_EMUsort
        # "20240612_162758587511",  # Th_7,3_spkTh_6,9_EMUsort
        # "20240612_162753771335",  # Th_5,2_spkTh_3,6_EMUsort
        # "20240612_162747515657",  # Th_5,2_spkTh_6,9_EMUsort
        # "20240612_162804267385",  # Th_2,1_spkTh_6,_EMUsort
        # "20240612_162811862295",  # Th_2,1_spkTh_6,9_EMUsort
        # "20240612_162813665969",  # Th_2,1_spkTh_9,_EMUsort
        # "20240612_162807780547",  # Th_2,1_spkTh_3,6_EMUsort
        # "20240612_162830774983",  # Th_2,1_spkTh_3,_EMUsort
        ## EMUsort_ks4_v1 with 8 channel dataset, 2.25 shape noise
        # "20240612_162951201953",  # Th_7,3_spkTh_6,_EMUsort
        # "20240612_162937624103",  # Th_10,4_spkTh_6,9_EMUsort
        # "20240612_162956613148",  # Th_9,8_spkTh_3,_EMUsort
        # "20240612_162949400686",  # Th_10,4_spkTh_3,_EMUsort
        # "20240612_162939821907",  # Th_9,8_spkTh_6,9_EMUsort
        # "20240612_162953594450",  # Th_7,3_spkTh_3,6_EMUsort
        # "20240612_162949352674",  # Th_7,3_spkTh_3,_EMUsort
        # "20240612_162947110279",  # Th_9,8_spkTh_3,6_EMUsort
        # "20240612_162952640058",  # Th_9,8_spkTh_9,_EMUsort
        # "20240612_162938982551",  # Th_10,4_spkTh_6,_EMUsort
        # "20240612_163003125561",  # Th_7,3_spkTh_9,_EMUsort
        # "20240612_162948608810",  # Th_5,2_spkTh_3,_EMUsort
        # "20240612_162939924132",  # Th_10,4_spkTh_3,6_EMUsort
        # "20240612_162945433175",  # Th_9,8_spkTh_6,_EMUsort
        # "20240612_162948745674",  # Th_10,4_spkTh_9,_EMUsort
        # "20240612_162949557504",  # Th_7,3_spkTh_6,9_EMUsort
        # "20240612_162952639866",  # Th_5,2_spkTh_6,_EMUsort
        # "20240612_162949005976",  # Th_5,2_spkTh_6,9_EMUsort
        # "20240612_162949985506",  # Th_5,2_spkTh_9,_EMUsort
        # "20240612_162948685329",  # Th_5,2_spkTh_3,6_EMUsort
        # "20240612_162955522948",  # Th_2,1_spkTh_6,9_EMUsort
        # "20240612_163005128853",  # Th_2,1_spkTh_6,_EMUsort
        "20240612_163003203155",  # Th_2,1_spkTh_3,6_EMUsort $$$
        # "20240612_162957571579",  # Th_2,1_spkTh_9,_EMUsort
        # "20240612_163002704257",  # Th_2,1_spkTh_3,_EMUsort
        # ## EMUsort replication of "full" EMUsort with 8 channel rat, 2.25 noise
        # "20240806_120615841738",  # Th_9,8_spkTh_6,9
        # "20240806_120616745005",  # Th_9,8_spkTh_3,6
        # "20240806_120616753651",  # Th_10,4_spkTh_6
        # "20240806_120617940585",  # Th_10,4_spkTh_6,9
        # "20240806_120619598012",  # Th_5,2_spkTh_9
        # "20240806_120619791756",  # Th_5,2_spkTh_3,6
        # "20240806_120619965658",  # Th_10,4_spkTh_3,6
        # "20240806_120621031098",  # Th_10,4_spkTh_3
        # "20240806_120621052823",  # Th_10,4_spkTh_9
        # "20240806_120621092749",  # Th_9,8_spkTh_3
        # "20240806_120621348820",  # Th_7,3_spkTh_6,9
        # "20240806_120622408905",  # Th_9,8_spkTh_6
        # "20240806_120623503157",  # Th_7,3_spkTh_9
        # "20240806_120624762432",  # Th_7,3_spkTh_6
        # "20240806_120625885872",  # Th_5,2_spkTh_3
        # "20240806_120625987775",  # Th_9,8_spkTh_9
        # "20240806_120626282989",  # Th_7,3_spkTh_3,6
        # "20240806_120626435160",  # Th_7,3_spkTh_3
        # "20240806_120627436681",  # Th_5,2_spkTh_6,9
        # "20240806_120628582330",  # Th_2,1_spkTh_6
        # "20240806_120629598433",  # Th_5,2_spkTh_6
        # "20240806_120632568044",  # Th_2,1_spkTh_6,9
        # "20240806_120634402481",  # Th_2,1_spkTh_3
        # "20240806_120635115658",  # Th_2,1_spkTh_3,6
        # "20240806_120637076301",  # Th_2,1_spkTh_9
        # ## EMUsort replication of "full" EMUsort with 8 channel rat, 2.25 noise, without init temp normalization
        # "20240806_141311661338",  # Th_10,4_spkTh_6
        # "20240806_141311718974",  # Th_9,8_spkTh_3,6
        # "20240806_141312808406",  # Th_10,4_spkTh_3
        # "20240806_141314646354",  # Th_10,4_spkTh_3,6
        # "20240806_141315105731",  # Th_5,2_spkTh_3
        # "20240806_141315728141",  # Th_9,8_spkTh_6,9
        # "20240806_141315873286",  # Th_5,2_spkTh_9
        # "20240806_141317034662",  # Th_5,2_spkTh_6,9
        # "20240806_141317102903",  # Th_7,3_spkTh_6,9
        # "20240806_141319208055",  # Th_9,8_spkTh_6
        # "20240806_141319295566",  # Th_5,2_spkTh_3,6
        # "20240806_141320972546",  # Th_10,4_spkTh_6,9
        # "20240806_141321311716",  # Th_10,4_spkTh_9
        # "20240806_141321321316",  # Th_7,3_spkTh_3
        # "20240806_141321371298",  # Th_9,8_spkTh_9
        # "20240806_141322347198",  # Th_7,3_spkTh_6
        # "20240806_141323042120",  # Th_5,2_spkTh_6
        # "20240806_141324311571",  # Th_7,3_spkTh_3,6
        # "20240806_141324415003",  # Th_2,1_spkTh_6,9
        # "20240806_141325535909",  # Th_9,8_spkTh_3
        # "20240806_141326064773",  # Th_7,3_spkTh_9
        # "20240806_141327377276",  # Th_2,1_spkTh_6
        # "20240806_141329278340",  # Th_2,1_spkTh_9
        # "20240806_141331066000",  # Th_2,1_spkTh_3,6
        # "20240806_141337957671",  # Th_2,1_spkTh_3
        # ## EMUsort spikedetect_orig, EMUsort with 8 channel rat dataset, 2.25 shape noise
        # "20240805_181823766336",  # Th_10,4_spkTh_9
        # "20240805_181829910981",  # Th_5,2_spkTh_6,9
        # "20240805_181831265373",  # Th_7,3_spkTh_9
        # "20240805_181832036707",  # Th_7,3_spkTh_3
        # "20240805_181833060633",  # Th_7,3_spkTh_6
        # "20240805_181834029400",  # Th_10,4_spkTh_6,9
        # "20240805_181836948674",  # Th_7,3_spkTh_6,9
        # "20240805_181837028862",  # Th_5,2_spkTh_9
        # "20240805_181837169887",  # Th_5,2_spkTh_6
        # "20240805_181837282023",  # Th_9,8_spkTh_9
        # "20240805_181837292127",  # Th_7,3_spkTh_3,6
        # "20240805_181837673358",  # Th_9,8_spkTh_6
        # "20240805_181838315875",  # Th_10,4_spkTh_6
        # "20240805_181839293019",  # Th_10,4_spkTh_3,6
        # "20240805_181841031304",  # Th_10,4_spkTh_3
        # "20240805_181841992562",  # Th_2,1_spkTh_6
        # "20240805_181842739280",  # Th_5,2_spkTh_3,6
        # "20240805_181844255664",  # Th_9,8_spkTh_3
        # "20240805_181844715498",  # Th_9,8_spkTh_3,6
        # "20240805_181844735641",  # Th_5,2_spkTh_3
        # "20240805_181844792752",  # Th_9,8_spkTh_6,9
        # "20240805_181845895495",  # Th_2,1_spkTh_9
        # "20240805_181848518479",  # Th_2,1_spkTh_6,9
        # "20240805_181852707476",  # Th_2,1_spkTh_3
        # "20240805_181853639707",  # Th_2,1_spkTh_3,6
        ## EMUsort_ks4_v1 with 8 channel dataset, 3.00 shape noise
        # "20240612_163456847912",  # Th_10,4_spkTh_3,_EMUsort
        # "20240612_163509588959",  # Th_9,8_spkTh_9,_EMUsort
        # "20240612_163509402189",  # Th_9,8_spkTh_3,_EMUsort
        # "20240612_163514818342",  # Th_7,3_spkTh_3,_EMUsort
        # "20240612_163457087553",  # Th_9,8_spkTh_6,_EMUsort
        # "20240612_163500641022",  # Th_10,4_spkTh_6,_EMUsort
        # "20240612_163457958695",  # Th_5,2_spkTh_3,_EMUsort
        # "20240612_163511976094",  # Th_9,8_spkTh_3,6_EMUsort
        # "20240612_163511649836",  # Th_10,4_spkTh_9,_EMUsort
        # "20240612_163516092712",  # Th_7,3_spkTh_9,_EMUsort
        # "20240612_163515710161",  # Th_7,3_spkTh_3,6_EMUsort
        # "20240612_163509504122",  # Th_10,4_spkTh_3,6_EMUsort
        # "20240612_163509663660",  # Th_7,3_spkTh_6,_EMUsort
        # "20240612_163451886749",  # Th_5,2_spkTh_9,_EMUsort
        # "20240612_163503054240",  # Th_9,8_spkTh_6,9_EMUsort
        # "20240612_163509664155",  # Th_5,2_spkTh_6,_EMUsort
        # "20240612_163502951561",  # Th_7,3_spkTh_6,9_EMUsort
        # "20240612_163513531758",  # Th_5,2_spkTh_3,6_EMUsort
        # "20240612_163454592660",  # Th_10,4_spkTh_6,9_EMUsort
        # "20240612_163512799163",  # Th_5,2_spkTh_6,9_EMUsort
        # "20240612_163509714597",  # Th_2,1_spkTh_3,6_EMUsort
        # "20240612_163511305800",  # Th_2,1_spkTh_3,_EMUsort
        # "20240612_163517768562",  # Th_2,1_spkTh_6,9_EMUsort
        # "20240612_163515072613",  # Th_2,1_spkTh_6,_EMUsort
        # "20240612_163511836776",  # Th_2,1_spkTh_9,_EMUsort
        ## EMUsort_ks4_v1 with 8 channel dataset, 3.75 shape noise
        # "20240612_163612266382",  # Th_9,8_spkTh_3,6_EMUsort
        # "20240612_163624275296",  # Th_9,8_spkTh_3,_EMUsort
        # "20240612_163625449893",  # Th_10,4_spkTh_3,_EMUsort
        # "20240612_163625421120",  # Th_10,4_spkTh_9,_EMUsort
        # "20240612_163624916656",  # Th_7,3_spkTh_3,_EMUsort
        # "20240612_163620890276",  # Th_10,4_spkTh_3,6_EMUsort
        # "20240612_163624023981",  # Th_5,2_spkTh_3,_EMUsort
        # "20240612_163627036954",  # Th_10,4_spkTh_6,_EMUsort
        # "20240612_163623531411",  # Th_9,8_spkTh_6,_EMUsort
        # "20240612_163626332791",  # Th_7,3_spkTh_9,_EMUsort
        # "20240612_163612906850",  # Th_10,4_spkTh_6,9_EMUsort
        # "20240612_163612163443",  # Th_9,8_spkTh_9,_EMUsort
        # "20240612_163623125667",  # Th_7,3_spkTh_6,_EMUsort
        # "20240612_163623968848",  # Th_5,2_spkTh_9,_EMUsort
        # "20240612_163627176252",  # Th_7,3_spkTh_3,6_EMUsort
        # "20240612_163619453868",  # Th_9,8_spkTh_6,9_EMUsort
        # "20240612_163626812564",  # Th_5,2_spkTh_6,_EMUsort
        # "20240612_163625346642",  # Th_7,3_spkTh_6,9_EMUsort
        # "20240612_163627685734",  # Th_5,2_spkTh_3,6_EMUsort
        # "20240612_163626778080",  # Th_5,2_spkTh_6,9_EMUsort
        # "20240612_163627578147",  # Th_2,1_spkTh_3,_EMUsort
        # "20240612_163626951675",  # Th_2,1_spkTh_9,_EMUsort
        # "20240612_163613236762",  # Th_2,1_spkTh_3,6_EMUsort
        # "20240612_163625666562",  # Th_2,1_spkTh_6,9_EMUsort
        # "20240612_163625450028",  # Th_2,1_spkTh_6,_EMUsort
        ## KS4 with 8 channel dataset, 0.00 shape noise
        # "20240612_160623265839",  # Th_9,8_spkTh_7.5,_KS4
        # "20240612_160634085558",  # Th_10,4_spkTh_9,_KS4
        # "20240612_160634005376",  # Th_10,4_spkTh_7.5,_KS4
        # "20240612_160625198414",  # Th_7,3_spkTh_9,_KS4
        # "20240612_160627251249",  # Th_9,8_spkTh_9,_KS4
        # "20240612_160633955887",  # Th_10,4_spkTh_4.5,_KS4
        # "20240612_160642552998",  # Th_9,8_spkTh_4.5,_KS4
        # "20240612_160629838486",  # Th_7,3_spkTh_7.5,_KS4
        # "20240612_160633955006",  # Th_7,3_spkTh_4.5,_KS4
        # "20240612_160649986542",  # Th_10,4_spkTh_6,_KS4
        # "20240612_160630173018",  # Th_5,2_spkTh_7.5,_KS4
        # "20240612_160630052462",  # Th_5,2_spkTh_9,_KS4
        # "20240612_160642484030",  # Th_9,8_spkTh_6,_KS4
        # "20240612_160634245620",  # Th_5,2_spkTh_4.5,_KS4
        # "20240612_160650289299",  # Th_9,8_spkTh_3,_KS4
        # "20240612_160645577431",  # Th_10,4_spkTh_3,_KS4
        # "20240612_160644229947",  # Th_7,3_spkTh_6,_KS4
        # "20240612_160707122596",  # Th_7,3_spkTh_3,_KS4
        # "20240612_160649771331",  # Th_5,2_spkTh_6,_KS4
        # "20240612_160658545558",  # Th_5,2_spkTh_3,_KS4
        # "20240612_160712004267",  # Th_2,1_spkTh_9,_KS4
        # "20240612_160703613226",  # Th_2,1_spkTh_7.5,_KS4
        # "20240612_160710165326",  # Th_2,1_spkTh_4.5,_KS4
        # "20240612_160717616064",  # Th_2,1_spkTh_6,_KS4
        # "20240612_160714931896",  # Th_2,1_spkTh_3,_KS4
        ## KS4 with 8 channel dataset, 0.75 shape noise
        # "20240612_191022229064",  # Th_10,4_spkTh_9,_KS4
        # "20240612_191056956312",  # Th_9,8_spkTh_3,_KS4
        # "20240612_191016130666",  # Th_9,8_spkTh_7.5,_KS4
        # "20240612_191033805655",  # Th_10,4_spkTh_4.5,_KS4
        # "20240612_191013225870",  # Th_10,4_spkTh_7.5,_KS4
        # "20240612_191028681729",  # Th_9,8_spkTh_9,_KS4
        # "20240612_191032332682",  # Th_9,8_spkTh_6,_KS4
        # "20240612_191026833973",  # Th_10,4_spkTh_6,_KS4
        # "20240612_191022232900",  # Th_7,3_spkTh_7.5,_KS4
        # "20240612_191033238242",  # Th_9,8_spkTh_4.5,_KS4
        # "20240612_191026165297",  # Th_7,3_spkTh_4.5,_KS4
        # "20240612_191019025773",  # Th_7,3_spkTh_9,_KS4
        # "20240612_191040335594",  # Th_10,4_spkTh_3,_KS4
        # "20240612_191026282145",  # Th_7,3_spkTh_6,_KS4
        # "20240612_191015670046",  # Th_5,2_spkTh_9,_KS4
        # "20240612_191018551000",  # Th_5,2_spkTh_7.5,_KS4
        # "20240612_191040918457",  # Th_5,2_spkTh_4.5,_KS4
        # "20240612_191045310532",  # Th_5,2_spkTh_6,_KS4
        # "20240612_191045336727",  # Th_7,3_spkTh_3,_KS4
        # "20240612_191052408651",  # Th_5,2_spkTh_3,_KS4
        # "20240612_191045379213",  # Th_2,1_spkTh_9,_KS4
        # "20240612_191049000592",  # Th_2,1_spkTh_7.5,_KS4
        # "20240612_191049559495",  # Th_2,1_spkTh_6,_KS4
        # "20240612_191037764189",  # Th_2,1_spkTh_4.5,_KS4
        # "20240612_191056602909",  # Th_2,1_spkTh_3,_KS4
        ## KS4 with 8 channel dataset, 1.50 shape noise
        # "20240610_205549162846",  # Th_5,2_spkTh_9,_KS4
        # "20240610_205549194908",  # Th_9,8_spkTh_7.5,_KS4
        # "20240610_205550531163",  # Th_10,4_spkTh_6,_KS4
        # "20240610_205556016089",  # Th_7,3_spkTh_9,_KS4
        # "20240610_205556780966",  # Th_9,8_spkTh_6,_KS4
        # "20240610_205558961843",  # Th_10,4_spkTh_9,_KS4
        # "20240610_205600645282",  # Th_5,2_spkTh_7.5,_KS4
        # "20240610_205600880436",  # Th_7,3_spkTh_7.5,_KS4
        # "20240610_205602139135",  # Th_9,8_spkTh_9,_KS4
        # "20240610_205603619106",  # Th_5,2_spkTh_6,_KS4
        # "20240610_205603694940",  # Th_10,4_spkTh_7.5,_KS4
        # "20240610_205605944399",  # Th_2,1_spkTh_4.5,_KS4
        # "20240610_205607516496",  # Th_9,8_spkTh_4.5,_KS4
        # "20240610_205609352554",  # Th_9,8_spkTh_3,_KS4
        # "20240610_205609838050",  # Th_2,1_spkTh_9,_KS4
        # "20240610_205610429996",  # Th_10,4_spkTh_4.5,_KS4
        # "20240610_205612701074",  # Th_7,3_spkTh_6,_KS4
        # "20240610_205614381413",  # Th_7,3_spkTh_4.5,_KS4
        # "20240610_205615543443",  # Th_5,2_spkTh_4.5,_KS4
        # "20240610_205621258722",  # Th_2,1_spkTh_6,_KS4
        # "20240610_205621934206",  # Th_7,3_spkTh_3,_KS4
        # "20240610_205621998693",  # Th_10,4_spkTh_3,_KS4
        # "20240610_205624703785",  # Th_2,1_spkTh_7.5,_KS4
        # "20240610_205625128096",  # Th_2,1_spkTh_3,_KS4
        # "20240610_205628446770",  # Th_5,2_spkTh_3,_KS4
        ## KS4 with 8 channel dataset, 2.25 shape noise
        # "20240610_205951805102",  # Th_9,8_spkTh_4.5,_KS4
        # "20240610_205951877368",  # Th_10,4_spkTh_9,_KS4
        # "20240610_205952087384",  # Th_9,8_spkTh_9,_KS4
        # "20240610_205955006571",  # Th_7,3_spkTh_7.5,_KS4 $$$
        # "20240610_205955424268",  # Th_9,8_spkTh_6,_KS4
        # "20240610_205955427991",  # Th_9,8_spkTh_3,_KS4
        # "20240610_205955432490",  # Th_7,3_spkTh_9,_KS4
        # "20240610_205955630742",  # Th_10,4_spkTh_4.5,_KS4
        # "20240610_205959412603",  # Th_10,4_spkTh_6,_KS4
        # "20240610_205959752576",  # Th_7,3_spkTh_6,_KS4
        # "20240610_210002336032",  # Th_10,4_spkTh_7.5,_KS4
        # "20240610_210003584446",  # Th_5,2_spkTh_7.5,_KS4
        # "20240610_210006079118",  # Th_5,2_spkTh_9,_KS4
        # "20240610_210006373858",  # Th_9,8_spkTh_7.5,_KS4
        # "20240610_210009082010",  # Th_2,1_spkTh_9,_KS4
        # "20240610_210009254314",  # Th_7,3_spkTh_3,_KS4
        # "20240610_210010776885",  # Th_2,1_spkTh_7.5,_KS4
        # "20240610_210011434312",  # Th_5,2_spkTh_6,_KS4
        # "20240610_210011766282",  # Th_10,4_spkTh_3,_KS4
        # "20240610_210013813475",  # Th_2,1_spkTh_6,_KS4
        # "20240610_210017139135",  # Th_2,1_spkTh_4.5,_KS4
        # "20240610_210018295377",  # Th_5,2_spkTh_4.5,_KS4
        # "20240610_210023435827",  # Th_7,3_spkTh_4.5,_KS4
        # "20240610_210026557813",  # Th_5,2_spkTh_3,_KS4
        # "20240610_210041069900",  # Th_2,1_spkTh_3,_KS4
        ## KS4 with 8 channel dataset, 3.00 shape noise
        # "20240610_210111895998",  # Th_9,8_spkTh_9,_KS4
        # "20240610_210115143763",  # Th_10,4_spkTh_7.5,_KS4
        # "20240610_210116222087",  # Th_7,3_spkTh_9,_KS4
        # "20240610_210117345585",  # Th_10,4_spkTh_4.5,_KS4
        # "20240610_210118503720",  # Th_10,4_spkTh_6,_KS4
        # "20240610_210118514944",  # Th_7,3_spkTh_6,_KS4
        # "20240610_210120157951",  # Th_9,8_spkTh_3,_KS4
        # "20240610_210120217065",  # Th_10,4_spkTh_9,_KS4
        # "20240610_210122537963",  # Th_9,8_spkTh_7.5,_KS4
        # "20240610_210123687997",  # Th_5,2_spkTh_7.5,_KS4
        # "20240610_210125007541",  # Th_10,4_spkTh_3,_KS4
        # "20240610_210126618780",  # Th_9,8_spkTh_4.5,_KS4
        # "20240610_210129200103",  # Th_7,3_spkTh_7.5,_KS4
        # "20240610_210132034489",  # Th_5,2_spkTh_9,_KS4
        # "20240610_210133057620",  # Th_9,8_spkTh_6,_KS4
        # "20240610_210133938145",  # Th_7,3_spkTh_4.5,_KS4
        # "20240610_210138166466",  # Th_7,3_spkTh_3,_KS4
        # "20240610_210138771464",  # Th_5,2_spkTh_4.5,_KS4
        # "20240610_210141206281",  # Th_5,2_spkTh_6,_KS4
        # "20240610_210141238019",  # Th_2,1_spkTh_9,_KS4
        # "20240610_210143013031",  # Th_2,1_spkTh_7.5,_KS4
        # "20240610_210143020063",  # Th_2,1_spkTh_6,_KS4
        # "20240610_210145823086",  # Th_2,1_spkTh_4.5,_KS4
        # "20240610_210155999130",  # Th_5,2_spkTh_3,_KS4
        # "20240610_210205815765",  # Th_2,1_spkTh_3,_KS4
        ## KS4 with 8 channel dataset, 3.75 shape noise
        # "20240612_160907046373",  # Th_10,4_spkTh_7.5,_KS4
        # "20240612_160909489176",  # Th_9,8_spkTh_3,_KS4
        # "20240612_160855813475",  # Th_10,4_spkTh_6,_KS4
        # "20240612_160908262884",  # Th_9,8_spkTh_7.5,_KS4
        # "20240612_160900942945",  # Th_9,8_spkTh_6,_KS4
        # "20240612_160905404942",  # Th_10,4_spkTh_9,_KS4
        # "20240612_160857606471",  # Th_10,4_spkTh_4.5,_KS4
        # "20240612_160908592167",  # Th_9,8_spkTh_9,_KS4
        # "20240612_160905869272",  # Th_9,8_spkTh_4.5,_KS4
        # "20240612_160919584949",  # Th_10,4_spkTh_3,_KS4
        # "20240612_160909410527",  # Th_7,3_spkTh_7.5,_KS4
        # "20240612_160916819364",  # Th_7,3_spkTh_9,_KS4
        # "20240612_160916236028",  # Th_7,3_spkTh_6,_KS4
        # "20240612_160916056380",  # Th_7,3_spkTh_4.5,_KS4
        # "20240612_160918292115",  # Th_7,3_spkTh_3,_KS4
        # "20240612_160913099076",  # Th_5,2_spkTh_7.5,_KS4
        # "20240612_160910452481",  # Th_5,2_spkTh_9,_KS4
        # "20240612_160903696016",  # Th_5,2_spkTh_6,_KS4
        # "20240612_160921096259",  # Th_5,2_spkTh_4.5,_KS4
        # "20240612_160920183500",  # Th_5,2_spkTh_3,_KS4
        # "20240612_160917766395",  # Th_2,1_spkTh_6,_KS4
        # "20240612_160928954596",  # Th_2,1_spkTh_7.5,_KS4
        # "20240612_160919396268",  # Th_2,1_spkTh_3,_KS4
        # "20240612_160921224979",  # Th_2,1_spkTh_9,_KS4
        # "20240612_160920824169",  # Th_2,1_spkTh_4.5,_KS4
        ## EMUsort with 8 channel monkey dataset, 0.00 shape noise
        # "20240729_144735275390",  # Th_10,4_spkTh_3,6
        # "20240729_144738481381",  # Th_9,8_spkTh_9
        # "20240729_144738483859",  # Th_10,4_spkTh_9
        # "20240729_144739221455",  # Th_9,8_spkTh_6,9
        # "20240729_144740493837",  # Th_9,8_spkTh_6
        # "20240729_144740752816",  # Th_9,8_spkTh_3
        # "20240729_144741087491",  # Th_9,8_spkTh_3,6
        # "20240729_144741906760",  # Th_10,4_spkTh_3
        # "20240729_144742462640",  # Th_10,4_spkTh_6,9
        # "20240729_144743054720",  # Th_10,4_spkTh_6
        # "20240729_144744222918",  # Th_7,3_spkTh_9
        # "20240729_144744808447",  # Th_7,3_spkTh_3
        # "20240729_144747762023",  # Th_7,3_spkTh_6
        # "20240729_144747810927",  # Th_7,3_spkTh_6,9
        # "20240729_144749739034",  # Th_7,3_spkTh_3,6
        # "20240729_144757059368",  # Th_5,2_spkTh_6
        # "20240729_144757132848",  # Th_5,2_spkTh_3,6
        # "20240729_144758382361",  # Th_5,2_spkTh_9
        # "20240729_144759068036",  # Th_5,2_spkTh_6,9
        # "20240729_144802124317",  # Th_5,2_spkTh_3
        # "20240729_144807290099",  # Th_2,1_spkTh_6
        # "20240729_144808734701",  # Th_2,1_spkTh_3,6
        # "20240729_144808780519",  # Th_2,1_spkTh_9
        # "20240729_144808875007",  # Th_2,1_spkTh_6,9
        # "20240729_144809879874",  # Th_2,1_spkTh_3
        # ## EMUsort with 8 channel monkey dataset, 0.75 shape noise
        # "20240729_150216499934",  # Th_10,4_spkTh_3,6
        # "20240729_150219065196",  # Th_10,4_spkTh_3
        # "20240729_150221584248",  # Th_7,3_spkTh_9
        # "20240729_150222169724",  # Th_9,8_spkTh_6
        # "20240729_150222644309",  # Th_10,4_spkTh_6
        # "20240729_150224042483",  # Th_9,8_spkTh_3,6
        # "20240729_150224138534",  # Th_7,3_spkTh_3
        # "20240729_150224157341",  # Th_10,4_spkTh_6,9
        # "20240729_150224228967",  # Th_9,8_spkTh_9
        # "20240729_150225213599",  # Th_9,8_spkTh_3
        # "20240729_150229272967",  # Th_10,4_spkTh_9
        # "20240729_150229302285",  # Th_7,3_spkTh_6,9
        # "20240729_150229394357",  # Th_9,8_spkTh_6,9
        # "20240729_150229988247",  # Th_7,3_spkTh_6
        # "20240729_150232954844",  # Th_7,3_spkTh_3,6
        # "20240729_150234946209",  # Th_5,2_spkTh_3
        # "20240729_150240449711",  # Th_5,2_spkTh_6,9
        # "20240729_150241924340",  # Th_5,2_spkTh_3,6
        # "20240729_150242872867",  # Th_5,2_spkTh_9
        # "20240729_150243403948",  # Th_5,2_spkTh_6
        # "20240729_150243406427",  # Th_2,1_spkTh_6,9
        # "20240729_150245536823",  # Th_2,1_spkTh_6
        # "20240729_150249459168",  # Th_2,1_spkTh_9
        # "20240729_150249562005",  # Th_2,1_spkTh_3,6
        # "20240729_150252521026",  # Th_2,1_spkTh_3
        # ## EMUsort with 8 channel monkey dataset, 1.50 shape noise
        # "20240729_153638201680",  # Th_10,4_spkTh_6
        # "20240729_153642346039",  # Th_10,4_spkTh_3
        # "20240729_153643758861",  # Th_9,8_spkTh_6
        # "20240729_153646235535",  # Th_10,4_spkTh_6,9
        # "20240729_153646240857",  # Th_9,8_spkTh_9
        # "20240729_153646383431",  # Th_9,8_spkTh_3,6
        # "20240729_153647728445",  # Th_9,8_spkTh_3
        # "20240729_153648458060",  # Th_10,4_spkTh_9
        # "20240729_153648592723",  # Th_9,8_spkTh_6,9
        # "20240729_153648697869",  # Th_10,4_spkTh_3,6
        # "20240729_153649473712",  # Th_7,3_spkTh_3
        # "20240729_153650577782",  # Th_7,3_spkTh_3,6
        # "20240729_153653534123",  # Th_7,3_spkTh_6
        # "20240729_153654612759",  # Th_7,3_spkTh_6,9
        # "20240729_153654647117",  # Th_7,3_spkTh_9
        # "20240729_153654982896",  # Th_5,2_spkTh_6,9
        # "20240729_153655116680",  # Th_5,2_spkTh_6
        # "20240729_153656599003",  # Th_5,2_spkTh_9
        # "20240729_153658395492",  # Th_5,2_spkTh_3,6
        # "20240729_153659199267",  # Th_2,1_spkTh_6,9
        # "20240729_153701007096",  # Th_2,1_spkTh_9
        # "20240729_153702086907",  # Th_2,1_spkTh_6
        # "20240729_153702180908",  # Th_5,2_spkTh_3
        # "20240729_153706636335",  # Th_2,1_spkTh_3
        # "20240729_153708142777",  # Th_2,1_spkTh_3,6
        # ## EMUsort with 8 channel monkey dataset, 2.25 shape noise
        # "20240729_154815433183",  # Th_9,8_spkTh_6
        # "20240729_154820083604",  # Th_9,8_spkTh_3,6
        # "20240729_154820478686",  # Th_9,8_spkTh_9
        # "20240729_154822471039",  # Th_10,4_spkTh_3
        # "20240729_154822525582",  # Th_10,4_spkTh_6,9
        # "20240729_154823085597",  # Th_9,8_spkTh_6,9
        # "20240729_154823091090",  # Th_10,4_spkTh_3,6
        # "20240729_154823218983",  # Th_10,4_spkTh_9
        # "20240729_154823265184",  # Th_10,4_spkTh_6
        # "20240729_154824447139",  # Th_7,3_spkTh_3,6
        # "20240729_154824462380",  # Th_9,8_spkTh_3
        # "20240729_154824559408",  # Th_7,3_spkTh_6,9
        # "20240729_154826396020",  # Th_7,3_spkTh_9
        # "20240729_154829670530",  # Th_7,3_spkTh_3
        # "20240729_154830155217",  # Th_5,2_spkTh_3,6
        # "20240729_154830191641",  # Th_5,2_spkTh_6
        # "20240729_154830294346",  # Th_2,1_spkTh_3,6
        # "20240729_154830349476",  # Th_7,3_spkTh_6
        # "20240729_154830359810",  # Th_5,2_spkTh_3
        # "20240729_154831184258",  # Th_2,1_spkTh_6,9
        # "20240729_154832844760",  # Th_5,2_spkTh_9
        # "20240729_154836406699",  # Th_2,1_spkTh_9
        # "20240729_154836467501",  # Th_5,2_spkTh_6,9
        # "20240729_154837774767",  # Th_2,1_spkTh_6
        # "20240729_154838963555",  # Th_2,1_spkTh_3
        # ## SPIKEDETECT_ORIG, EMUsort with 8 channel monkey dataset, 2.25 shape noise
        # "20240802_194643174275",  # Th_10,4_spkTh_9
        # "20240802_194645442549",  # Th_9,8_spkTh_3,6
        # "20240802_194646169533",  # Th_10,4_spkTh_3,6
        # "20240802_194646225376",  # Th_10,4_spkTh_3
        # "20240802_194647090719",  # Th_9,8_spkTh_3
        # "20240802_194648464324",  # Th_9,8_spkTh_6
        # "20240802_194649703320",  # Th_10,4_spkTh_6,9
        # "20240802_194650064474",  # Th_9,8_spkTh_9
        # "20240802_194650526025",  # Th_9,8_spkTh_6,9
        # "20240802_194650754331",  # Th_7,3_spkTh_9
        # "20240802_194650962863",  # Th_10,4_spkTh_6
        # "20240802_194654872038",  # Th_5,2_spkTh_9
        # "20240802_194655707829",  # Th_7,3_spkTh_3
        # "20240802_194656562543",  # Th_5,2_spkTh_3
        # "20240802_194657224726",  # Th_7,3_spkTh_3,6
        # "20240802_194658559079",  # Th_2,1_spkTh_9
        # "20240802_194658604594",  # Th_5,2_spkTh_3,6
        # "20240802_194659131796",  # Th_7,3_spkTh_6,9
        # "20240802_194702806798",  # Th_5,2_spkTh_6,9
        # "20240802_194703645927",  # Th_7,3_spkTh_6
        # "20240802_194704166921",  # Th_5,2_spkTh_6
        # "20240802_194707030313",  # Th_2,1_spkTh_6
        # "20240802_194707464796",  # Th_2,1_spkTh_3
        # "20240802_194707484350",  # Th_2,1_spkTh_6,9
        # "20240802_194709242683",  # Th_2,1_spkTh_3,6
        # ## SPIKEDETECT_EMU_100% 0.2 TUKEY, EMUsort with 8 channel monkey dataset, 2.25 shape noise
        # "20240802_200457811866",  # Th_9,8_spkTh_3
        # "20240802_200500148536",  # Th_9,8_spkTh_9
        # "20240802_200500420255",  # Th_10,4_spkTh_3
        # "20240802_200501996593",  # Th_10,4_spkTh_6,9
        # "20240802_200502481051",  # Th_9,8_spkTh_6
        # "20240802_200502780055",  # Th_10,4_spkTh_9
        # "20240802_200504386943",  # Th_10,4_spkTh_6
        # "20240802_200504831966",  # Th_9,8_spkTh_3,6
        # "20240802_200505916978",  # Th_10,4_spkTh_3,6
        # "20240802_200505987510",  # Th_9,8_spkTh_6,9
        # "20240802_200508232376",  # Th_7,3_spkTh_9
        # "20240802_200510325663",  # Th_2,1_spkTh_9
        # "20240802_200512641524",  # Th_7,3_spkTh_3,6
        # "20240802_200512687576",  # Th_7,3_spkTh_6
        # "20240802_200513042258",  # Th_7,3_spkTh_6,9
        # "20240802_200513204543",  # Th_5,2_spkTh_3
        # "20240802_200514355943",  # Th_5,2_spkTh_6
        # "20240802_200514584973",  # Th_7,3_spkTh_3
        # "20240802_200516987098",  # Th_5,2_spkTh_3,6
        # "20240802_200517445757",  # Th_5,2_spkTh_9
        # "20240802_200518405242",  # Th_5,2_spkTh_6,9
        # "20240802_200519605359",  # Th_2,1_spkTh_3,6
        # "20240802_200521451163",  # Th_2,1_spkTh_6,9
        # "20240802_200521476452",  # Th_2,1_spkTh_6
        # "20240802_200521538874",  # Th_2,1_spkTh_3
        # ## EMUsort GMM on spikes, no windowing, EMUsort with 8 channel monkey dataset, 2.25 shape noise
        # "20240805_165547997309",  # Th_10,4_spkTh_9
        # "20240805_165553499487",  # Th_9,8_spkTh_9
        # "20240805_165554016046",  # Th_10,4_spkTh_6
        # "20240805_165556182143",  # Th_9,8_spkTh_3
        # "20240805_165557295304",  # Th_10,4_spkTh_3,6
        # "20240805_165557832891",  # Th_10,4_spkTh_6,9
        # "20240805_165558237005",  # Th_10,4_spkTh_3
        # "20240805_165558269418",  # Th_9,8_spkTh_6
        # "20240805_165602089271",  # Th_7,3_spkTh_3
        # "20240805_165603328702",  # Th_7,3_spkTh_3,6
        # "20240805_165604974114",  # Th_5,2_spkTh_3,6
        # "20240805_165606017878",  # Th_9,8_spkTh_3,6
        # "20240805_165606135730",  # Th_7,3_spkTh_9
        # "20240805_165607729516",  # Th_9,8_spkTh_6,9
        # "20240805_165609646815",  # Th_2,1_spkTh_9
        # "20240805_165612330276",  # Th_7,3_spkTh_6
        # "20240805_165614132667",  # Th_2,1_spkTh_3
        # "20240805_165617363192",  # Th_5,2_spkTh_3
        # "20240805_165617606739",  # Th_7,3_spkTh_6,9
        # "20240805_165617791964",  # Th_5,2_spkTh_9
        # "20240805_165619154697",  # Th_2,1_spkTh_6,9
        # "20240805_165619166760",  # Th_2,1_spkTh_3,6
        # "20240805_165619212573",  # Th_5,2_spkTh_6,9
        # "20240805_165620223399",  # Th_5,2_spkTh_6
        # "20240805_165622541837",  # Th_2,1_spkTh_6
        # ## EMUsort with 8 channel monkey dataset, 3.00 shape noise
        # "20240729_154051215897",  # Th_9,8_spkTh_6,9
        # "20240729_154051281291",  # Th_9,8_spkTh_3,6
        # "20240729_154053068421",  # Th_10,4_spkTh_9
        # "20240729_154053283394",  # Th_10,4_spkTh_6
        # "20240729_154053836935",  # Th_10,4_spkTh_3,6
        # "20240729_154053881040",  # Th_5,2_spkTh_3,6
        # "20240729_154053941322",  # Th_9,8_spkTh_9
        # "20240729_154053950189",  # Th_9,8_spkTh_6
        # "20240729_154053956661",  # Th_9,8_spkTh_3
        # "20240729_154054420152",  # Th_5,2_spkTh_9
        # "20240729_154056948639",  # Th_10,4_spkTh_6,9
        # "20240729_154058093740",  # Th_5,2_spkTh_6
        # "20240729_154058133967",  # Th_7,3_spkTh_9
        # "20240729_154058279493",  # Th_7,3_spkTh_6,9
        # "20240729_154059229195",  # Th_5,2_spkTh_3
        # "20240729_154101979990",  # Th_10,4_spkTh_3
        # "20240729_154102018455",  # Th_7,3_spkTh_6
        # "20240729_154102240950",  # Th_2,1_spkTh_9
        # "20240729_154102249721",  # Th_2,1_spkTh_6,9
        # "20240729_154103427245",  # Th_7,3_spkTh_3,6
        # "20240729_154104382574",  # Th_7,3_spkTh_3
        # "20240729_154104382768",  # Th_5,2_spkTh_6,9
        # "20240729_154104735560",  # Th_2,1_spkTh_3
        # "20240729_154104899705",  # Th_2,1_spkTh_6
        # "20240729_154106508617",  # Th_2,1_spkTh_3,6
        # ## EMUsort with 8 channel monkey dataset, 3.75 shape noise
        # "20240729_154411898143",  # Th_10,4_spkTh_9
        # "20240729_154413273499",  # Th_9,8_spkTh_6,9
        # "20240729_154414029444",  # Th_10,4_spkTh_3
        # "20240729_154416286899",  # Th_10,4_spkTh_6,9
        # "20240729_154416353633",  # Th_9,8_spkTh_3,6
        # "20240729_154417028970",  # Th_9,8_spkTh_3
        # "20240729_154417401777",  # Th_5,2_spkTh_6
        # "20240729_154417749110",  # Th_7,3_spkTh_3,6
        # "20240729_154418840782",  # Th_10,4_spkTh_3,6
        # "20240729_154420344559",  # Th_10,4_spkTh_6
        # "20240729_154420658756",  # Th_9,8_spkTh_9
        # "20240729_154420723857",  # Th_7,3_spkTh_6,9
        # "20240729_154422131131",  # Th_5,2_spkTh_9
        # "20240729_154422161995",  # Th_9,8_spkTh_6
        # "20240729_154422217428",  # Th_5,2_spkTh_3
        # "20240729_154425105818",  # Th_5,2_spkTh_3,6
        # "20240729_154425182835",  # Th_2,1_spkTh_9
        # "20240729_154426375115",  # Th_2,1_spkTh_6,9
        # "20240729_154426480872",  # Th_7,3_spkTh_9
        # "20240729_154428225909",  # Th_2,1_spkTh_6
        # "20240729_154428985312",  # Th_5,2_spkTh_6,9
        # "20240729_154429056582",  # Th_7,3_spkTh_3
        # "20240729_154430164593",  # Th_7,3_spkTh_6
        # "20240729_154430652381",  # Th_2,1_spkTh_3,6
        # "20240729_154431649770",  # Th_2,1_spkTh_3
        # ## Kilosort4 with 8 channel monkey dataset, 0.00 shape noise
        # "20240731_155143927861",  # Th_10,4_spkTh_4.5_KS4
        # "20240731_155144546038",  # Th_10,4_spkTh_3_KS4
        # "20240731_155146623895",  # Th_10,4_spkTh_9_KS4
        # "20240731_155146632971",  # Th_9,8_spkTh_3_KS4
        # "20240731_155146873163",  # Th_10,4_spkTh_6_KS4
        # "20240731_155148098711",  # Th_9,8_spkTh_9_KS4
        # "20240731_155148253302",  # Th_9,8_spkTh_4.5_KS4
        # "20240731_155149183343",  # Th_9,8_spkTh_6_KS4
        # "20240731_155150170841",  # Th_9,8_spkTh_7.5_KS4
        # "20240731_155151225890",  # Th_10,4_spkTh_7.5_KS4
        # "20240731_155155494558",  # Th_7,3_spkTh_9_KS4
        # "20240731_155159718197",  # Th_7,3_spkTh_7.5_KS4
        # "20240731_155201123960",  # Th_7,3_spkTh_6_KS4
        # "20240731_155202192757",  # Th_7,3_spkTh_4.5_KS4
        # "20240731_155205062814",  # Th_5,2_spkTh_4.5_KS4
        # "20240731_155206031652",  # Th_5,2_spkTh_6_KS4
        # "20240731_155207666356",  # Th_7,3_spkTh_3_KS4
        # "20240731_155208355282",  # Th_5,2_spkTh_3_KS4
        # "20240731_155211995331",  # Th_5,2_spkTh_7.5_KS4
        # "20240731_155213428310",  # Th_2,1_spkTh_7.5_KS4
        # "20240731_155216652715",  # Th_2,1_spkTh_6_KS4
        # "20240731_155218072961",  # Th_5,2_spkTh_9_KS4
        # "20240731_155221834342",  # Th_2,1_spkTh_9_KS4
        # "20240731_155222322474",  # Th_2,1_spkTh_3_KS4
        # "20240731_155225655170",  # Th_2,1_spkTh_4.5_KS4
        # ## Kilosort4 with 8 channel monkey dataset, 0.75 shape noise
        # "20240731_155546444343",  # Th_10,4_spkTh_9_KS4
        # "20240731_155547686611",  # Th_10,4_spkTh_3_KS4
        # "20240731_155550201616",  # Th_10,4_spkTh_4.5_KS4
        # "20240731_155550327636",  # Th_10,4_spkTh_7.5_KS4
        # "20240731_155550506189",  # Th_9,8_spkTh_6_KS4
        # "20240731_155550928517",  # Th_9,8_spkTh_7.5_KS4
        # "20240731_155551259486",  # Th_10,4_spkTh_6_KS4
        # "20240731_155551429393",  # Th_9,8_spkTh_3_KS4
        # "20240731_155551511145",  # Th_9,8_spkTh_9_KS4
        # "20240731_155552525380",  # Th_9,8_spkTh_4.5_KS4
        # "20240731_155554078826",  # Th_7,3_spkTh_9_KS4
        # "20240731_155557544692",  # Th_7,3_spkTh_6_KS4
        # "20240731_155557695040",  # Th_5,2_spkTh_7.5_KS4
        # "20240731_155559361998",  # Th_7,3_spkTh_4.5_KS4
        # "20240731_155600296905",  # Th_7,3_spkTh_3_KS4
        # "20240731_155600415189",  # Th_7,3_spkTh_7.5_KS4
        # "20240731_155605450418",  # Th_5,2_spkTh_9_KS4
        # "20240731_155606946769",  # Th_5,2_spkTh_6_KS4
        # "20240731_155608716424",  # Th_5,2_spkTh_4.5_KS4
        # "20240731_155610177911",  # Th_5,2_spkTh_3_KS4
        # "20240731_155610756090",  # Th_2,1_spkTh_9_KS4
        # "20240731_155614397259",  # Th_2,1_spkTh_6_KS4
        # "20240731_155616423085",  # Th_2,1_spkTh_3_KS4
        # "20240731_155617843617",  # Th_2,1_spkTh_4.5_KS4
        # "20240731_155621659040",  # Th_2,1_spkTh_7.5_KS4
        # ## Kilosort4 with 8 channel monkey dataset, 1.50 shape noise
        # "20240731_160707390186",  # Th_9,8_spkTh_3_KS4
        # "20240731_160708270994",  # Th_9,8_spkTh_9_KS4
        # "20240731_160709473998",  # Th_9,8_spkTh_4.5_KS4
        # "20240731_160710703719",  # Th_10,4_spkTh_6_KS4
        # "20240731_160710828336",  # Th_10,4_spkTh_9_KS4
        # "20240731_160714083854",  # Th_10,4_spkTh_3_KS4
        # "20240731_160714091635",  # Th_10,4_spkTh_7.5_KS4
        # "20240731_160714106228",  # Th_10,4_spkTh_4.5_KS4
        # "20240731_160714246362",  # Th_9,8_spkTh_7.5_KS4
        # "20240731_160714548207",  # Th_9,8_spkTh_6_KS4
        # "20240731_160715631015",  # Th_7,3_spkTh_3_KS4
        # "20240731_160716936064",  # Th_7,3_spkTh_9_KS4
        # "20240731_160718614697",  # Th_7,3_spkTh_6_KS4
        # "20240731_160720435419",  # Th_7,3_spkTh_4.5_KS4
        # "20240731_160721402304",  # Th_7,3_spkTh_7.5_KS4
        # "20240731_160722654317",  # Th_5,2_spkTh_6_KS4
        # "20240731_160725220054",  # Th_5,2_spkTh_4.5_KS4
        # "20240731_160725995100",  # Th_5,2_spkTh_9_KS4
        # "20240731_160727126094",  # Th_5,2_spkTh_7.5_KS4
        # "20240731_160728080110",  # Th_2,1_spkTh_9_KS4
        # "20240731_160729379832",  # Th_2,1_spkTh_4.5_KS4
        # "20240731_160731386534",  # Th_2,1_spkTh_6_KS4
        # "20240731_160732208700",  # Th_5,2_spkTh_3_KS4
        # "20240731_160736101825",  # Th_2,1_spkTh_7.5_KS4
        # "20240731_160741656037",  # Th_2,1_spkTh_3_KS4
        # ## Kilosort4 with 8 channel monkey dataset, 2.25 shape noise
        # "20240731_161302330839",  # Th_10,4_spkTh_3_KS4
        # "20240731_161302431627",  # Th_10,4_spkTh_6_KS4
        # "20240731_161302989672",  # Th_9,8_spkTh_6_KS4
        # "20240731_161304556120",  # Th_9,8_spkTh_3_KS4
        # "20240731_161305308614",  # Th_10,4_spkTh_7.5_KS4
        # "20240731_161305977181",  # Th_10,4_spkTh_9_KS4
        # "20240731_161307110971",  # Th_9,8_spkTh_9_KS4
        # "20240731_161307997726",  # Th_9,8_spkTh_7.5_KS4
        # "20240731_161308443423",  # Th_10,4_spkTh_4.5_KS4
        # "20240731_161310151970",  # Th_9,8_spkTh_4.5_KS4
        # "20240731_161310945830",  # Th_7,3_spkTh_7.5_KS4
        # "20240731_161311616465",  # Th_5,2_spkTh_7.5_KS4
        # "20240731_161314101781",  # Th_5,2_spkTh_9_KS4
        # "20240731_161314677105",  # Th_7,3_spkTh_4.5_KS4
        # "20240731_161316623068",  # Th_7,3_spkTh_6_KS4
        # "20240731_161316627759",  # Th_7,3_spkTh_9_KS4
        # "20240731_161319916195",  # Th_5,2_spkTh_6_KS4
        # "20240731_161320605504",  # Th_7,3_spkTh_3_KS4
        # "20240731_161321884103",  # Th_5,2_spkTh_4.5_KS4
        # "20240731_161323937747",  # Th_5,2_spkTh_3_KS4
        # "20240731_161324738434",  # Th_2,1_spkTh_9_KS4
        # "20240731_161326432641",  # Th_2,1_spkTh_7.5_KS4
        # "20240731_161328087769",  # Th_2,1_spkTh_4.5_KS4
        # "20240731_161328416665",  # Th_2,1_spkTh_6_KS4
        # "20240731_161331209863",  # Th_2,1_spkTh_3_KS4
        # ## Kilosort4 with 8 channel monkey dataset, 3.00 shape noise
        # "20240731_162137010229",  # Th_9,8_spkTh_3_KS4
        # "20240731_162138924904",  # Th_10,4_spkTh_7.5_KS4
        # "20240731_162142361541",  # Th_9,8_spkTh_6_KS4
        # "20240731_162142403957",  # Th_10,4_spkTh_9_KS4
        # "20240731_162143375234",  # Th_9,8_spkTh_4.5_KS4
        # "20240731_162143600790",  # Th_9,8_spkTh_9_KS4
        # "20240731_162143628911",  # Th_10,4_spkTh_6_KS4
        # "20240731_162145001297",  # Th_10,4_spkTh_4.5_KS4
        # "20240731_162145102627",  # Th_10,4_spkTh_3_KS4
        # "20240731_162147539394",  # Th_9,8_spkTh_7.5_KS4
        # "20240731_162149291646",  # Th_7,3_spkTh_6_KS4
        # "20240731_162150594576",  # Th_7,3_spkTh_7.5_KS4
        # "20240731_162150780826",  # Th_7,3_spkTh_4.5_KS4
        # "20240731_162152116648",  # Th_7,3_spkTh_9_KS4
        # "20240731_162154678819",  # Th_5,2_spkTh_9_KS4
        # "20240731_162155059627",  # Th_7,3_spkTh_3_KS4
        # "20240731_162155106263",  # Th_5,2_spkTh_7.5_KS4
        # "20240731_162155116877",  # Th_2,1_spkTh_9_KS4
        # "20240731_162157016037",  # Th_2,1_spkTh_6_KS4
        # "20240731_162158563424",  # Th_5,2_spkTh_4.5_KS4
        # "20240731_162158621279",  # Th_5,2_spkTh_6_KS4
        # "20240731_162200001246",  # Th_2,1_spkTh_4.5_KS4
        # "20240731_162200825833",  # Th_5,2_spkTh_3_KS4
        # "20240731_162208171282",  # Th_2,1_spkTh_7.5_KS4
        # "20240731_162210263399",  # Th_2,1_spkTh_3_KS4
        # ## Kilosort4 with 8 channel monkey dataset, 3.75 shape noise
        # "20240731_164150648764",  # Th_10,4_spkTh_6_KS4
        # "20240731_164151949196",  # Th_9,8_spkTh_4.5_KS4
        # "20240731_164153510177",  # Th_9,8_spkTh_7.5_KS4
        # "20240731_164155681829",  # Th_9,8_spkTh_6_KS4
        # "20240731_164155924911",  # Th_10,4_spkTh_7.5_KS4
        # "20240731_164156714278",  # Th_10,4_spkTh_3_KS4
        # "20240731_164156983224",  # Th_10,4_spkTh_4.5_KS4
        # "20240731_164159708632",  # Th_9,8_spkTh_3_KS4
        # "20240731_164202844717",  # Th_7,3_spkTh_6_KS4
        # "20240731_164202859632",  # Th_9,8_spkTh_9_KS4
        # "20240731_164202871331",  # Th_7,3_spkTh_7.5_KS4
        # "20240731_164202873704",  # Th_10,4_spkTh_9_KS4
        # "20240731_164206081027",  # Th_5,2_spkTh_3_KS4
        # "20240731_164206487111",  # Th_7,3_spkTh_9_KS4
        # "20240731_164207541393",  # Th_5,2_spkTh_6_KS4
        # "20240731_164207541585",  # Th_7,3_spkTh_3_KS4
        # "20240731_164207614482",  # Th_5,2_spkTh_7.5_KS4
        # "20240731_164207653077",  # Th_5,2_spkTh_4.5_KS4
        # "20240731_164211033125",  # Th_2,1_spkTh_6_KS4
        # "20240731_164211525898",  # Th_7,3_spkTh_4.5_KS4
        # "20240731_164212559678",  # Th_5,2_spkTh_9_KS4
        # "20240731_164216099639",  # Th_2,1_spkTh_9_KS4
        # "20240731_164216227174",  # Th_2,1_spkTh_3_KS4
        # "20240731_164216405140",  # Th_2,1_spkTh_7.5_KS4
        # "20240731_164218400584",  # Th_2,1_spkTh_4.5_KS4
        # ## 20240815 New LOF removal GMM EMUsort 8CH, 2.25 shape noise
        # "20240815_135426166293",  # Th_10,4_spkTh_6,9,12
        # "20240815_135428527819",  # Th_7,3_spkTh_6,9
        # "20240815_135430029665",  # Th_10,4_spkTh_6,9,12,15
        # "20240815_135430032514",  # Th_7,3_spkTh_6
        # "20240815_135430073426",  # Th_10,4_spkTh_6,9
        # "20240815_135431522092",  # Th_5,2_spkTh_6,9,12,15
        # "20240815_135431550709",  # Th_10,4_spkTh_3,6,9
        # "20240815_135431962282",  # Th_5,2_spkTh_6
        # "20240815_135432970642",  # Th_10,4_spkTh_6
        # "20240815_135433008326",  # Th_9,8_spkTh_6,9
        # "20240815_135435469161",  # Th_7,3_spkTh_6,9,12,15
        # "20240815_135435490534",  # Th_7,3_spkTh_3,6,9
        # "20240815_135435494113",  # Th_7,3_spkTh_6,9,12
        # "20240815_135435494110",  # Th_9,8_spkTh_3,6,9
        # "20240815_135436993445",  # Th_5,2_spkTh_3,6,9
        # "20240815_135437457800",  # Th_5,2_spkTh_6,9,12
        # "20240815_135437461873",  # Th_9,8_spkTh_6
        # "20240815_135437546299",  # Th_9,8_spkTh_6,9,12
        # "20240815_135437630192",  # Th_5,2_spkTh_6,9
        # "20240815_135442080980",  # Th_2,1_spkTh_3,6,9
        # "20240815_135442920146",  # Th_2,1_spkTh_6,9
        # "20240815_135442950342",  # Th_2,1_spkTh_6,9,12,15
        # "20240815_135444685513",  # Th_9,8_spkTh_6,9,12,15
        # "20240815_135444772600",  # Th_2,1_spkTh_6
        # "20240815_135446823866",  # Th_2,1_spkTh_6,9,12
        # ## 20240815 spikedetect_orig, duplicated thresholds, EMUsort 8CH, 2.25 shape noise
        # "20240815_171552756333",  # Th_10,4_spkTh_6,9,12
        # "20240815_171553752996",  # Th_10,4_spkTh_6,9,12,15
        # "20240815_171554003032",  # Th_5,2_spkTh_6,9,12,15
        # "20240815_171556859518",  # Th_7,3_spkTh_6,9,12
        # "20240815_171557737488",  # Th_9,8_spkTh_6,9
        # "20240815_171557776036",  # Th_5,2_spkTh_6,9,12
        # "20240815_171557851521",  # Th_7,3_spkTh_3,6,9
        # "20240815_171558035399",  # Th_7,3_spkTh_6,9
        # "20240815_171558912477",  # Th_5,2_spkTh_6,9
        # "20240815_171559548855",  # Th_7,3_spkTh_6
        # "20240815_171559812430",  # Th_9,8_spkTh_6
        # "20240815_171559940013",  # Th_7,3_spkTh_6,9,12,15
        # "20240815_171601987528",  # Th_5,2_spkTh_6
        # "20240815_171603252153",  # Th_9,8_spkTh_6,9,12,15
        # "20240815_171604103320",  # Th_9,8_spkTh_6,9,12
        # "20240815_171604504069",  # Th_2,1_spkTh_6,9,12
        # "20240815_171604530051",  # Th_10,4_spkTh_6,9
        # "20240815_171606317842",  # Th_2,1_spkTh_6
        # "20240815_171606365924",  # Th_10,4_spkTh_6
        # "20240815_171606370357",  # Th_10,4_spkTh_3,6,9
        # "20240815_171607952320",  # Th_9,8_spkTh_3,6,9
        # "20240815_171609128293",  # Th_2,1_spkTh_6,9
        # "20240815_171611049601",  # Th_5,2_spkTh_3,6,9
        # "20240815_171611933506",  # Th_2,1_spkTh_3,6,9
        # "20240815_171612099781",  # Th_2,1_spkTh_6,9,12,15
        # ## 20240815 spikedetect_orig, but new multithreshold, EMUsort 8CH, 2.25 shape noise
        # "20240815_182228658014",  # Th_10,4_spkTh_6,9
        # "20240815_182231077916",  # Th_10,4_spkTh_6
        # "20240815_182232096484",  # Th_9,8_spkTh_6,9
        # "20240815_182233253040",  # Th_5,2_spkTh_6,9,12,15
        # "20240815_182233883947",  # Th_5,2_spkTh_6,9,12
        # "20240815_182235363692",  # Th_5,2_spkTh_6
        # "20240815_182237882421",  # Th_10,4_spkTh_6,9,12
        # "20240815_182238696967",  # Th_7,3_spkTh_6
        # "20240815_182244489714",  # Th_7,3_spkTh_6,9,12
        # "20240815_182244690436",  # Th_9,8_spkTh_6
        # "20240815_182245604285",  # Th_10,4_spkTh_6,9,12,15
        # "20240815_182246217221",  # Th_10,4_spkTh_3,6,9
        # "20240815_182246878572",  # Th_7,3_spkTh_6,9,12,15
        # "20240815_182247047453",  # Th_7,3_spkTh_6,9
        # "20240815_182247100868",  # Th_5,2_spkTh_3,6,9
        # "20240815_182247107367",  # Th_9,8_spkTh_6,9,12,15
        # "20240815_182249272599",  # Th_9,8_spkTh_6,9,12
        # "20240815_182250493729",  # Th_7,3_spkTh_3,6,9
        # "20240815_182251503401",  # Th_9,8_spkTh_3,6,9
        # "20240815_182251794642",  # Th_5,2_spkTh_6,9
        # "20240815_182252769322",  # Th_2,1_spkTh_6,9,12,15
        # "20240815_182253031491",  # Th_2,1_spkTh_6,9
        # "20240815_182253055137",  # Th_2,1_spkTh_6,9,12
        # "20240815_182253066893",  # Th_2,1_spkTh_6
        # "20240815_182255312648",  # Th_2,1_spkTh_3,6,9
        # ## 20240815 New LOF removal Kmeans EMUsort 8CH, 2.25 shape noise
        # "20240815_184458459195",  # Th_9,8_spkTh_3,6,9
        # "20240815_184459640520",  # Th_7,3_spkTh_6,9,12
        # "20240815_184501720853",  # Th_5,2_spkTh_6
        # "20240815_184503657686",  # Th_7,3_spkTh_6,9,12,15
        # "20240815_184504910787",  # Th_10,4_spkTh_6,9,12,15
        # "20240815_184504910216",  # Th_10,4_spkTh_6,9,12
        # "20240815_184505895777",  # Th_5,2_spkTh_3,6,9
        # "20240815_184510049529",  # Th_7,3_spkTh_3,6,9
        # "20240815_184510664338",  # Th_9,8_spkTh_6,9,12,15
        # "20240815_184511527298",  # Th_10,4_spkTh_3,6,9
        # "20240815_184512425088",  # Th_7,3_spkTh_6,9
        # "20240815_184515128529",  # Th_5,2_spkTh_6,9
        # "20240815_184516987910",  # Th_9,8_spkTh_6
        # "20240815_184517876281",  # Th_9,8_spkTh_6,9
        # "20240815_184518194288",  # Th_10,4_spkTh_6,9
        # "20240815_184518265247",  # Th_7,3_spkTh_6
        # "20240815_184519878705",  # Th_9,8_spkTh_6,9,12
        # "20240815_184520038929",  # Th_2,1_spkTh_6,9
        # "20240815_184520586598",  # Th_5,2_spkTh_6,9,12,15
        # "20240815_184520603795",  # Th_5,2_spkTh_6,9,12
        # "20240815_184520626109",  # Th_10,4_spkTh_6
        # "20240815_184522539747",  # Th_2,1_spkTh_3,6,9
        # "20240815_184523980141",  # Th_2,1_spkTh_6,9,12,15
        # "20240815_184527375839",  # Th_2,1_spkTh_6,9,12
        # "20240815_184529585442",  # Th_2,1_spkTh_6
        # ## 20240815 spikedetect_orig, substituted thresholds, EMUsort 8CH, 2.25 shape noise
        # "20240815_191437269809",  # Th_10,4_spkTh_10.5
        # "20240815_191437333148",  # Th_10,4_spkTh_4.5
        # "20240815_191439296205",  # Th_5,2_spkTh_9
        # "20240815_191439296137",  # Th_10,4_spkTh_6
        # "20240815_191440434038",  # Th_7,3_spkTh_9
        # "20240815_191441476060",  # Th_9,8_spkTh_10.5
        # "20240815_191442010721",  # Th_9,8_spkTh_6
        # "20240815_191442461036",  # Th_7,3_spkTh_6
        # "20240815_191443407876",  # Th_10,4_spkTh_9
        # "20240815_191445382232",  # Th_5,2_spkTh_10.5
        # "20240815_191446162752",  # Th_7,3_spkTh_7.5
        # "20240815_191447020263",  # Th_7,3_spkTh_10.5
        # "20240815_191450023979",  # Th_9,8_spkTh_9
        # "20240815_191450107825",  # Th_10,4_spkTh_7.5
        # "20240815_191450539183",  # Th_9,8_spkTh_4.5
        # "20240815_191452726165",  # Th_7,3_spkTh_4.5
        # "20240815_191453085931",  # Th_9,8_spkTh_7.5
        # "20240815_191454161385",  # Th_5,2_spkTh_6
        # "20240815_191454403584",  # Th_5,2_spkTh_4.5
        # "20240815_191455545043",  # Th_2,1_spkTh_9
        # "20240815_191456204399",  # Th_2,1_spkTh_7.5
        # "20240815_191457217199",  # Th_5,2_spkTh_7.5
        # "20240815_191500012582",  # Th_2,1_spkTh_6
        # "20240815_191502404902",  # Th_2,1_spkTh_4.5
        # "20240815_191503851038",  # Th_2,1_spkTh_10.5
        ## KS4 with 8 channel dataset, 2.25 shape noise, substituted thresholds, worse perf than other reference set (with 20240610_205955006571)
        # "20240815_195154718843",  # Th_10,4_spkTh_9_KS4
        # "20240815_195155859002",  # Th_5,2_spkTh_10.5_KS4
        # "20240815_195159617168",  # Th_9,8_spkTh_6_KS4
        # "20240815_195206197698",  # Th_10,4_spkTh_6_KS4
        # "20240815_195206614464",  # Th_7,3_spkTh_6_KS4
        # "20240815_195208328443",  # Th_9,8_spkTh_9_KS4
        # "20240815_195208651613",  # Th_10,4_spkTh_7.5_KS4
        # "20240815_195208780396",  # Th_2,1_spkTh_6_KS4
        # "20240815_195208780906",  # Th_5,2_spkTh_7.5_KS4
        # "20240815_195210577885",  # Th_9,8_spkTh_4.5_KS4
        # "20240815_195210644923",  # Th_7,3_spkTh_10.5_KS4
        # "20240815_195210650540",  # Th_10,4_spkTh_4.5_KS4
        # "20240815_195210851865",  # Th_10,4_spkTh_10.5_KS4
        # "20240815_195213150845",  # Th_7,3_spkTh_9_KS4
        # "20240815_195213156403",  # Th_5,2_spkTh_9_KS4
        # "20240815_195213335217",  # Th_9,8_spkTh_10.5_KS4
        # "20240815_195215436767",  # Th_7,3_spkTh_7.5_KS4
        # "20240815_195216238256",  # Th_9,8_spkTh_7.5_KS4
        # "20240815_195219056140",  # Th_2,1_spkTh_7.5_KS4
        # "20240815_195222081080",  # Th_2,1_spkTh_9_KS4
        # "20240815_195222620229",  # Th_5,2_spkTh_6_KS4
        # "20240815_195223290221",  # Th_2,1_spkTh_10.5_KS4
        # "20240815_195225535852",  # Th_5,2_spkTh_4.5_KS4
        # "20240815_195226256096",  # Th_2,1_spkTh_4.5_KS4
        # "20240815_195227064717",  # Th_7,3_spkTh_4.5_KS4
        ## Monkey KS4 with 8 channel dataset, 2.25 shape noise, substituted thresholds, maybe new reference if perf is much diff
        # "20240815_200404022653"  # Th_10,4_spkTh_4.5_KS4
        # "20240815_200405708056"  # Th_9,8_spkTh_6_KS4
        # "20240815_200409303251"  # Th_9,8_spkTh_7.5_KS4
        # "20240815_200409341153"  # Th_10,4_spkTh_7.5_KS4
        # "20240815_200410394373"  # Th_9,8_spkTh_4.5_KS4
        # "20240815_200410862090"  # Th_10,4_spkTh_9_KS4
        # "20240815_200410987944"  # Th_10,4_spkTh_6_KS4
        # "20240815_200412862511"  # Th_10,4_spkTh_10.5_KS4
        # "20240815_200414693906"  # Th_7,3_spkTh_10.5_KS4
        # "20240815_200416779182"  # Th_9,8_spkTh_10.5_KS4
        # "20240815_200417510389"  # Th_9,8_spkTh_9_KS4
        # "20240815_200419026041"  # Th_7,3_spkTh_7.5_KS4
        # "20240815_200420011955"  # Th_7,3_spkTh_6_KS4
        # "20240815_200420073254"  # Th_5,2_spkTh_7.5_KS4
        # "20240815_200421720884"  # Th_5,2_spkTh_9_KS4
        # "20240815_200422005863"  # Th_7,3_spkTh_4.5_KS4
        # "20240815_200422607861"  # Th_7,3_spkTh_9_KS4
        # "20240815_200426785945"  # Th_5,2_spkTh_4.5_KS4
        # "20240815_200428867325"  # Th_2,1_spkTh_10.5_KS4
        # "20240815_200429129847"  # Th_2,1_spkTh_4.5_KS4
        # "20240815_200429831866"  # Th_5,2_spkTh_10.5_KS4
        # "20240815_200430288129"  # Th_2,1_spkTh_6_KS4
        # "20240815_200432894285"  # Th_2,1_spkTh_7.5_KS4
        # "20240815_200434728996"  # Th_5,2_spkTh_6_KS4
        # "20240815_200435507544"  # Th_2,1_spkTh_9_KS4
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

    # find the folder name which ends in _myo and append to the paths_to_session_folders
    paths_to_each_myo_folder = []
    for iDir in paths_to_KS_session_folders:
        myo = [f for f in iDir.iterdir() if (f.is_dir() and f.name.endswith("_myo"))]
        if len(myo) == 0:
            assert (
                len(paths_to_KS_session_folders) == 1
            ), f"There should be one _myo folder in each session folder, but there were {len(myo)} in {iDir}"
            myo = paths_to_KS_session_folders  # set to the session folder if no _myo folder is found
        assert (
            len(myo) == 1
        ), f"There should be one _myo folder in each session folder, but there were {len(myo)} in {iDir}"
        paths_to_each_myo_folder.append(myo[0])

    # inside each _myo folder, find the folder name which constains sort_from_each_path_to_load string
    list_of_paths_to_sorted_folders = []
    for iPath in paths_to_each_myo_folder:
        matches_init = [
            f
            for f in iPath.iterdir()
            if f.is_dir() and any(s in f.name for s in sorts_from_each_path_to_load)
        ]
        # sort the matches by the order in sorts_from_each_path_to_load
        matches = []
        for iSort in sorts_from_each_path_to_load:
            matches.append([f for f in matches_init if iSort in f.name][0])
        # assert (
        #     len(matches) == 1
        # ), f"There should be one sort folder match in each _myo folder, but there were {len(matches)} in {iPath}"
        if use_custom_merge_clusters:
            # append the path to the custom_merge_clusters folder
            list_of_paths_to_sorted_folders.append(
                [
                    matches[i].joinpath("custom_merges/final_merge")
                    for i in range(len(matches))
                ]
            )
        elif from_spike_interface:
            # ensure there is an initial element for each path
            list_of_paths_to_sorted_folders.append([])
            # append the paths to the matched sort folders
            for iMatch in range(len(matches)):
                sorter_output_exists = (
                    matches[iMatch].joinpath("sorter_output").is_dir()
                )
                if sorter_output_exists:
                    list_of_paths_to_sorted_folders[-1].append(
                        matches[iMatch] / "sorter_output"
                    )
                else:
                    list_of_paths_to_sorted_folders[-1].append(matches[iMatch])
        else:
            list_of_paths_to_sorted_folders[-1].append(matches)

    if automatically_assign_cluster_mapping:
        # tracemalloc.start()
        # automatically assign cluster mapping by extracting the waves at the spike times for all
        # clusters, getting the median waveform for each cluster using both groundtruth and the sort
        # by using the respective spike times, then computing the correlation between each cluster's
        # median wave and the median waves of the ground truth clusters, pairing the clusters with
        # the highest correlation match, and then using those pairs to reorder the clusters
        # in 'clusters_to_take_from' to match the ground truth clusters
        # also need to be sure to check all lags between the ground truth and the sort median waves
        # to make sure that the correlation is not being computed between two waves that are
        # misaligned in time, which would result in an errantly and artificially low correlation

        # drop any clusters with <300 spikes
        # clusters_in_sort_to_use = clusters_in_sort_to_use[
        #     np.array(
        #         [
        #             np.sum(spike_clusters == iCluster) >= 300
        #             for iCluster in clusters_in_sort_to_use
        #         ]
        #     ).astype(int)
        # ]

        if method_for_automatic_cluster_mapping != "accuracies":
            spike_times_list = [
                np.load(
                    str(path_to_sorted_folder.joinpath("spike_times.npy")),
                    # mmap_mode="r",
                ).flatten()
                for path_to_sorted_folder in list_of_paths_to_sorted_folders[0]
            ]

            spike_clusters_list = [
                np.load(
                    str(path_to_sorted_folder.joinpath("spike_clusters.npy")),
                    # mmap_mode="r",
                ).flatten()
                for path_to_sorted_folder in list_of_paths_to_sorted_folders[0]
            ]

            # clusters_in_sort_to_use = np.unique(
            #     spike_clusters_list[0]
            # )  # take all clusters in the first sort

            # get the spike times for each cluster
            spike_times = spike_times_list[0]
            spike_clusters = spike_clusters_list[0]
            clusters_in_sort_to_use = clusters_to_take_from[
                sorts_from_each_path_to_load[0]
            ]
            # get the spike times for each cluster
            spike_times_for_each_cluster = [
                spike_times[spike_clusters == iCluster]
                for iCluster in clusters_in_sort_to_use
            ]

            # now do the same for the ground truth spikes. Load the ground truth spike times
            # which are 1's and 0's, where 1's indicate a spike and 0's indicate no spike
            # each column is a different unit, and row is a different time point in the recording
            # use np.where to get the spike times for each cluster

            ground_truth_spike_times = np.load(
                str(ground_truth_path)
            )  # , mmap_mode="r")
            GT_spike_times_for_each_cluster = [
                np.where(ground_truth_spike_times[:, iCluster] == 1)[0]
                for iCluster in GT_clusters_to_use
            ]

        # now use the metric of choice to map the clusters according to best
        # correlation score across all pairs of clusters and time lags
        if method_for_automatic_cluster_mapping == "accuracies":

            def compute_accuracy_for_each_GT_cluster(
                ground_truth_path,
                jCluster_GT,
                KS_clusters_to_consider,  # list of candidate clusters for this GT cluster
                all_matched_KS_clusters,
                random_seed_entropy,
                correlation_alignment,
                precorrelation_rebin_width_ms,
                preaccuracy_rebin_width,
                ephys_fs,
                time_frame,
                path_to_sorted_folder,
                simulation_method,
                use_bins_for_matching=True,
            ):

                # use MUsim object to load and rebin ground truth data
                mu_GT = MUsim(random_seed_entropy)
                mu_GT.load_MUs(
                    # npy_file_path
                    ground_truth_path,
                    1 / ephys_fs,
                    load_as="trial",
                    slice=time_frame,
                    load_type=simulation_method,
                )
                mu_GT_bin_width_ms_orig = mu_GT.bin_width * 1000
                if simulation_method == "konstantin":
                    # load the .mat variables from the ground truth path:
                    # amplititude_sorted_idxs.mat, detectable_ind.mat
                    # then use amplititude_sorted_idxs[detectable_ind] indexes to slice the
                    # mu_GT.spikes[-1] entry to get the detectable MUs, sorted by amplitude
                    amplitude_sorted_idxs = (
                        loadmat(
                            ground_truth_path.joinpath("amplitude_sorted_idxs.mat")
                        )["amplitude_sorted_idxs"].astype(int)
                        - 1
                    )  # subtract 1 to use 0 indexing
                    detectable_ind = (
                        loadmat(ground_truth_path.joinpath("detectable_ind.mat"))[
                            "detectable_ind"
                        ].astype(int)
                        - 1
                    )  # subtract 1 to use 0 indexing
                    detectable_MU_idxs = amplitude_sorted_idxs[detectable_ind]
                    mu_GT.spikes[-1] = mu_GT.spikes[-1][:, detectable_MU_idxs]

                # use MUsim object to load and rebin Kilosort data
                mu_KS = MUsim(random_seed_entropy)
                mu_KS.load_MUs(
                    # npy_file_path
                    path_to_sorted_folder,
                    1 / ephys_fs,
                    load_as="trial",
                    slice=time_frame,
                    load_type="kilosort",
                )
                mu_KS_bin_width_ms_orig = mu_KS.bin_width * 1000
                if all_matched_KS_clusters is not None:
                    mu_KS_other = mu_KS.deepcopy()
                    all_other_KS_clusters = np.setdiff1d(
                        all_matched_KS_clusters, KS_clusters_to_consider
                    )
                    mu_KS_other.spikes[-1] = mu_KS_other.spikes[-1][
                        :, all_other_KS_clusters
                    ]
                if KS_clusters_to_consider is not None:
                    mu_KS.spikes[-1] = mu_KS.spikes[-1][:, KS_clusters_to_consider]
                    # ensure its 2d
                    if len(mu_KS.spikes[-1].shape) == 1:
                        mu_KS.spikes[-1] = mu_KS.spikes[-1][:, np.newaxis]

                # ensure kilosort_spikes and ground_truth_spikes have the same duration
                # add more kilosort bins to match ground truth
                # (fill missing allocation with zeros due to no spikes near end of recording)
                if mu_KS.spikes[-1].shape[0] < mu_GT.spikes[-1].shape[0]:
                    if len(mu_KS.spikes[-1].shape) == 2:
                        zeros_shape_tuple = (
                            mu_GT.spikes[-1].shape[0] - mu_KS.spikes[-1].shape[0],
                            mu_KS.spikes[-1].shape[1],
                        )
                        mu_KS.spikes[-1] = np.vstack(
                            (
                                mu_KS.spikes[-1],
                                np.zeros(zeros_shape_tuple),
                            )
                        )
                    else:
                        raise ValueError(
                            f"mu_KS.spikes[-1] must be 2D, but has shape {mu_KS.spikes[-1].shape}"
                        )
                if all_matched_KS_clusters is not None:
                    if mu_KS_other.spikes[-1].shape[0] < mu_GT.spikes[-1].shape[0]:
                        if len(mu_KS_other.spikes[-1].shape) == 2:
                            zeros_shape_tuple = (
                                mu_GT.spikes[-1].shape[0]
                                - mu_KS_other.spikes[-1].shape[0],
                                mu_KS_other.spikes[-1].shape[1],
                            )
                            mu_KS_other.spikes[-1] = np.vstack(
                                (
                                    mu_KS_other.spikes[-1],
                                    np.zeros(zeros_shape_tuple),
                                )
                            )
                        else:
                            raise ValueError(
                                f"mu_KS_other.spikes[-1] must be 2D, but has shape {mu_KS_other.spikes[-1].shape}"
                            )
                # compute the correlation between the two spike trains for each unit
                # use the correlation to determine the shift for each unit
                # use the shift to align the spike trains
                # use the aligned spike trains to compute the metrics
                if correlation_alignment:

                    min_delay_ms = -1  # ms
                    max_delay_ms = 1  # ms

                    if precorrelation_rebin_width_ms is not None:
                        # precorrelation alignment rebinning
                        mu_GT.rebin_trials(
                            precorrelation_rebin_width_ms / 1000
                        )  # rebin to rebin_width ms bins
                        mu_KS.rebin_trials(
                            precorrelation_rebin_width_ms / 1000
                        )  # rebin to rebin_width ms bins
                        mu_KS_other.rebin_trials(precorrelation_rebin_width_ms / 1000)

                        min_delay_samples = int(
                            round(min_delay_ms / precorrelation_rebin_width_ms)
                        )
                        max_delay_samples = int(
                            round(max_delay_ms / precorrelation_rebin_width_ms)
                        )
                    else:
                        min_delay_samples = int(round(min_delay_ms * ephys_fs / 1000))
                        max_delay_samples = int(round(max_delay_ms * ephys_fs / 1000))

                    for iUnit in range(mu_KS.spikes[-1].shape[1]):
                        # skip the columns which contain nan's, which represent nonexistent clusters
                        if np.any(np.isnan(mu_KS.spikes[-1][:, iUnit])):
                            continue

                        correlation = correlate(
                            mu_KS.spikes[-1][:, iUnit],
                            mu_GT.spikes[-1][:, jCluster_GT],
                            "same",
                        )
                        lags = correlation_lags(
                            len(mu_KS.spikes[-1][:, iUnit]),
                            len(mu_GT.spikes[-1][:, jCluster_GT]),
                            "same",
                        )
                        lag_constraint_idxs = np.where(
                            np.logical_and(
                                lags >= min_delay_samples, lags <= max_delay_samples
                            )
                        )[0]
                        # limit the lags to range from min_delay_samples to max_delay_samples
                        # lags = lags[(lags >= min_delay_samples) & (lags <= max_delay_samples)]
                        # find the lag with the highest correlation
                        max_correlation_index = np.argmax(
                            correlation[lag_constraint_idxs]
                        )
                        # find the lag with the highest correlation in the opposite direction
                        min_correlation_index = np.argmin(
                            correlation[lag_constraint_idxs]
                        )
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
                        mu_KS.spikes[-1][:, iUnit] = np.roll(
                            mu_KS.spikes[-1][:, iUnit], -shift
                        )
                        # make sure shift hasn't gone to the edge of the min or max delay
                        if shift <= min_delay_samples or shift >= max_delay_samples:
                            print(
                                f"WARNING: Shifted Kilosort spikes for GT unit {jCluster_GT} and KS unit {KS_clusters_to_consider} by {shift} samples"
                            )
                        else:
                            print(
                                f"Done with correlation-based shifting for GT unit {jCluster_GT} and KS unit {KS_clusters_to_consider} by {shift} samples"
                            )

                # if last iteration and spike_isolation_radius_ms != None, then remove spikes
                if iRepeat == num_repeats - 1 and spike_isolation_radius_ms is not None:
                    assert type(spike_isolation_radius_ms) in [
                        int,
                        float,
                    ], "Type of spike_isolation_radius_ms must be int or float"
                    assert (
                        spike_isolation_radius_ms > 0
                    ), "spike_isolation_radius_ms must be >0"

                    # account for rebinning, if it occurred
                    if precorrelation_rebin_width_ms is not None:
                        assert (
                            mu_KS_bin_width_ms_orig == mu_GT_bin_width_ms_orig
                        ), "mu_KS_bin_width_ms_orig should equal mu_GT_bin_width_ms_orig"
                        spike_isolation_radius_pts = int(
                            spike_isolation_radius_ms
                            * ephys_fs
                            * (mu_GT_bin_width_ms_orig / precorrelation_rebin_width_ms)
                            / 1000
                        )
                        # print(f"precorr: {precorrelation_rebin_width_ms}")
                        # print(f"orig: {mu_GT_bin_width_ms_orig}")
                    else:
                        spike_isolation_radius_pts = int(
                            spike_isolation_radius_ms * ephys_fs / 1000
                        )

                    # delete any spikes from mu_KS and mu_GT which are not within
                    # spike_isolation_radius_ms of spikes from neighboring MUs
                    # print(f"GT bin_width: {mu_GT.bin_width}")
                    # print(f"KS bin_width: {mu_KS.bin_width}")
                    GT_spike_counts_before = mu_GT.spikes[-1].sum(axis=0)
                    GT_spike_count_before = GT_spike_counts_before[jCluster_GT]
                    mu_GT = remove_isolated_spikes(mu_GT, spike_isolation_radius_pts)
                    GT_spike_counts_after = mu_GT.spikes[-1].sum(axis=0)
                    GT_spike_count_after = GT_spike_counts_after[jCluster_GT]
                    print(
                        f"Overlap fraction for cluster {jCluster_GT} is {GT_spike_count_after / GT_spike_count_before} with {spike_isolation_radius_pts} pt radius"
                    )
                    print(
                        f"Total average overlap fraction is {GT_spike_counts_before.sum() / GT_spike_counts_after.sum()}"
                    )
                    # concatenate the mu_KS_other spikes into mu_KS.spikes[-1] before removing isolated spikes
                    if all_matched_KS_clusters is not None:
                        # apply same roll to mu_KS_other.spikes
                        mu_KS_other.spikes[-1] = np.roll(
                            mu_KS_other.spikes[-1], -shift, axis=0
                        )
                        mu_KS.spikes[-1] = np.hstack(
                            (mu_KS.spikes[-1], mu_KS_other.spikes[-1])
                        )
                    mu_KS = remove_isolated_spikes(mu_KS, spike_isolation_radius_pts)
                    # make sure there are no isolated spikes remaining in mu_KS by taking the sum of a rolling 1ms window to make sure it never equals 1 (only zero or >1)
                    # only take the sum if a spike is in the center time point though
                    for iW in range(
                        len(mu_KS.spikes[-1]) - 2 * spike_isolation_radius_pts - 1
                    ):
                        spike_in_center_time_point = (
                            mu_KS.spikes[-1][iW + spike_isolation_radius_pts + 1].sum()
                            >= 1
                        )
                        if spike_in_center_time_point:
                            num_spk_in_window = np.sum(
                                mu_KS.spikes[-1][
                                    iW : iW + 2 * spike_isolation_radius_pts + 1
                                ]
                            )
                            if num_spk_in_window == 1:
                                print(
                                    f"WARNING: Isolated spike found in KS spikes at index {iW}, or time {iW*mu_KS.bin_width}"
                                )

                    # don't forget to slice them off once done
                    if all_matched_KS_clusters is not None:
                        mu_KS.spikes[-1] = mu_KS.spikes[-1][:, 0]
                        mu_KS.spikes[-1] = mu_KS.spikes[-1][:, np.newaxis]

                    # print(
                    #     f"removed {mu_GT.removed_spike_count} isolated spikes from GT spikes with radius {spike_isolation_radius_pts} pts"
                    # )
                    # print(
                    #     f"removed {mu_KS.removed_spike_count} isolated spikes from KS spikes with radius {spike_isolation_radius_pts} pts"
                    # )

                if use_bins_for_matching:
                    # rebin the spike trains to the bin width for comparison
                    mu_GT.rebin_trials(
                        preaccuracy_rebin_width / 1000
                    )  # rebin to rebin_width ms bins

                    mu_KS.rebin_trials(
                        preaccuracy_rebin_width / 1000
                    )  # rebin to rebin_width ms bins

                    kilosort_spikes = mu_KS.spikes[-1]  # shape is (num_bins, num_units)
                    ground_truth_spikes = mu_GT.spikes[
                        -1
                    ]  # shape is (num_bins, num_units)

                    # if kilosort spike length is greater, trim it to match GT
                    if kilosort_spikes.shape[0] > ground_truth_spikes.shape[0]:
                        print(
                            f"Shape mismatch after rebinning, trimming KS spikes time dimension down to {ground_truth_spikes.shape[0]}"
                        )
                        kilosort_spikes = kilosort_spikes[
                            : ground_truth_spikes.shape[0]
                        ]

                    ground_truth_spikes_this_clust_repeat = np.tile(
                        ground_truth_spikes[:, jCluster_GT],
                        (kilosort_spikes.shape[1], 1),
                    ).T

                    # compute false positive, false negative, and true positive spikes with vectorized operations
                    # algorithm then has O(n) time complexity
                    # create a new array for each type of error (false positive, false negative, true positive)
                    true_positive_spikes = np.zeros(kilosort_spikes.shape, dtype=int)
                    false_positive_spikes = np.zeros(kilosort_spikes.shape, dtype=int)
                    false_negative_spikes = np.zeros(kilosort_spikes.shape, dtype=int)

                    # Check for NaN's once for each unit
                    nan_units = np.any(np.isnan(kilosort_spikes), axis=0)

                    # Create masks for each condition
                    true_positive_mask = (kilosort_spikes == 1) & (
                        ground_truth_spikes_this_clust_repeat == 1
                    )
                    false_positive_mask = (kilosort_spikes >= 1) & (
                        ground_truth_spikes_this_clust_repeat == 0
                    )
                    false_negative_mask = (kilosort_spikes == 0) & (
                        ground_truth_spikes_this_clust_repeat >= 1
                    )

                    # Handle cases where spike counts are larger than 1
                    true_positive_large_mask = (kilosort_spikes > 1) & (
                        ground_truth_spikes_this_clust_repeat >= 1
                    )
                    false_positive_large_mask = (
                        kilosort_spikes > ground_truth_spikes_this_clust_repeat
                    ) & true_positive_large_mask
                    false_negative_large_mask = (
                        kilosort_spikes < ground_truth_spikes_this_clust_repeat
                    ) & true_positive_large_mask

                    # Apply masks to arrays
                    true_positive_spikes[true_positive_mask] = 1
                    false_positive_spikes[false_positive_mask] = kilosort_spikes[
                        false_positive_mask
                    ]
                    false_negative_spikes[false_negative_mask] = (
                        ground_truth_spikes_this_clust_repeat[false_negative_mask]
                    )

                    # Handle large spike counts
                    true_positive_spikes[true_positive_large_mask] = np.minimum(
                        kilosort_spikes[true_positive_large_mask],
                        ground_truth_spikes_this_clust_repeat[true_positive_large_mask],
                    )
                    false_positive_spikes[false_positive_large_mask] = (
                        kilosort_spikes[false_positive_large_mask]
                        - ground_truth_spikes_this_clust_repeat[
                            false_positive_large_mask
                        ]
                    )
                    false_negative_spikes[false_negative_large_mask] = (
                        ground_truth_spikes_this_clust_repeat[false_negative_large_mask]
                        - kilosort_spikes[false_negative_large_mask]
                    )

                    # Set NaN units to 0
                    true_positive_spikes[:, nan_units] = 0
                    false_positive_spikes[:, nan_units] = 0
                    false_negative_spikes[:, nan_units] = 0

                else:
                    # the alternative approach will not rebin the second time, and instead will
                    # convert the current MUsim_obj.bin_width value to convert each spike into times
                    # GT_times is M long and KS_times is N in length
                    # a new M x N array will be created by np.tile'ing the GT_times array
                    # the KS_times array will be subracted into the GT_repeated_times array, and
                    # this result will be set into a np.ma.masked_array called "spike_dt", setting
                    # the mask to all 1's of the same shape. From this 2D array, we will first
                    # determine the number of true positives, which is found by checking each row
                    # for the minimum absolute value that is less than preaccuracy_rebin_width
                    # milliseconds, if there is ever a tie, just take the first one. The
                    # corresponding row and column will then be masked to remove those entries from
                    # future consideration. The row and column indexes represent paired GT and KS
                    # spike times but only the total number of pairs needs to be stored into the
                    # num_matches variable. Then we need to find the number of false positives and
                    # false negatives. False negatives are found by checking the columns for any
                    # masked values, and the corresponding KS spike times are marked as false

                    current_GT_bin_width = mu_GT.bin_width  # seconds
                    current_KS_bin_width = mu_KS.bin_width  # seconds

                    ground_truth_spikes = mu_GT.spikes[-1]
                    kilosort_spikes = mu_KS.spikes[-1]

                    # convert to times with np.where followed by multiplication by bin_width, then
                    # convert seconds to milliseconds by multiplying by 1000
                    GT_spike_idxs = np.where(
                        ground_truth_spikes[:, jCluster_GT, np.newaxis]
                    )[0]
                    KS_spike_idxs = np.where(kilosort_spikes)[0]
                    GT_times = GT_spike_idxs * current_GT_bin_width * 1000
                    KS_times = KS_spike_idxs * current_KS_bin_width * 1000

                    # shape of this is (M x N), if M is the number of times in the GT spikes and
                    # N is the number of times in the KS spikes
                    GT_repeated_times = np.tile(
                        GT_times,
                        (KS_times.shape[0], 1),
                    ).T

                    spike_dt = np.ma.masked_array(
                        GT_repeated_times - KS_times,
                        mask=np.zeros(GT_repeated_times.shape),
                    )

                    # initialize train arrays
                    true_positive_spikes = np.zeros(kilosort_spikes.shape, dtype=int)
                    false_positive_spikes = np.zeros(kilosort_spikes.shape, dtype=int)
                    false_negative_spikes = np.zeros(kilosort_spikes.shape, dtype=int)

                    # mask the spike_dt array to remove any spike times that are too far apart
                    spike_dt_masked = np.ma.masked_greater_equal(
                        abs(spike_dt), preaccuracy_rebin_width
                    )
                    # any columns with only masked values is a KS spike time with no GT match
                    false_positive_spikes[
                        KS_spike_idxs[np.where(spike_dt_masked.sum(0).mask)]
                    ] = 1
                    # any rows with only masked values is a GT spike time with no KS match
                    false_negative_spikes[
                        GT_spike_idxs[np.where(spike_dt_masked.sum(1).mask)]
                    ] = 1

                    match_coords = np.where(spike_dt_masked.mask == 0)
                    for iMatch in np.unique(match_coords[0]):
                        matches = np.ma.argsort(spike_dt_masked[iMatch])
                        jMatch = matches[0]
                        if np.ma.masked in [jMatch, spike_dt[iMatch, jMatch]]:
                            raise ValueError
                        spike_dt.mask[iMatch, :] = (
                            1  # mask the row to mark the matching KS spike as claimed
                        )
                        true_positive_spikes[KS_spike_idxs[jMatch]] = 1
                        # check column for remaining non-masked values (corresponding times are
                        # false positive because GT spikes never violate refractory period)
                        false_positive_spikes[
                            KS_spike_idxs[np.where(spike_dt.mask[iMatch] == 0)]
                        ] = 1

                num_matches = np.sum(true_positive_spikes, axis=0)
                num_kilosort_spikes = np.sum(kilosort_spikes, axis=0)
                num_ground_truth_spikes = np.sum(ground_truth_spikes, axis=0)

                precision = compute_precision(num_matches, num_kilosort_spikes)
                recall = compute_recall(
                    num_matches, num_ground_truth_spikes, jCluster_GT
                )
                accuracy = compute_accuracy(
                    num_matches,
                    num_kilosort_spikes,
                    num_ground_truth_spikes,
                    jCluster_GT,
                )
                precisions_for_this_GT_cluster = np.fromiter(precision, float)
                recalls_for_this_GT_cluster = np.fromiter(recall, float)
                accuracies_for_this_GT_cluster = np.fromiter(accuracy, float)
                return (
                    jCluster_GT,
                    precisions_for_this_GT_cluster,
                    recalls_for_this_GT_cluster,
                    accuracies_for_this_GT_cluster,
                    num_matches,
                    num_kilosort_spikes,
                    num_ground_truth_spikes,
                    true_positive_spikes,
                    false_positive_spikes,
                    false_negative_spikes,
                    kilosort_spikes,
                    ground_truth_spikes,
                )

            # parameters for different settings across repeats
            # only do correlation alignment during 2nd pass
            correlation_alignment = [False, True]
            precorrelation_rebin_width_ms_list = [None, 0.1]
            preaccuracy_rebin_width = [10, 1]
            num_repeats = len(correlation_alignment)
            # repeat twice to only compute the correlation alignment once
            for iRepeat in range(num_repeats):
                # make lists to house the results from each iSort iteration
                precisions_list = []
                recalls_list = []
                accuracies_list = []
                num_matches_list = []
                num_kilosort_spikes_list = []
                num_ground_truth_spikes_list = []
                true_positive_spikes_list = []
                false_positive_spikes_list = []
                false_negative_spikes_list = []
                kilosort_spikes_list = []
                ground_truth_spikes_list = []
                sort_dstr_list = []
                for iSort, sort_dstr in enumerate(sorts_from_each_path_to_load):
                    # # use MUsim object to load and rebin Kilosort data
                    # mu_KS = MUsim(random_seed_entropy)
                    # mu_KS.sample_rate = ephys_fs
                    # mu_KS.load_MUs(
                    #     # npy_file_path
                    #     list_of_paths_to_sorted_folders[0][iSort],
                    #     1 / ephys_fs,
                    #     load_as="trial",
                    #     slice=[0, 0.01],  # save memory by only loading 1% of the file
                    #     load_type="kilosort",
                    # )
                    num_KS_clusters_this_sort = (
                        np.load(
                            list_of_paths_to_sorted_folders[0][iSort]
                            / "spike_clusters.npy"
                        )
                        .flatten()
                        .max()
                        + 1
                    )

                    precisions = np.zeros(
                        (len(GT_clusters_to_use), num_KS_clusters_this_sort)
                    )
                    recalls = np.zeros(
                        (len(GT_clusters_to_use), num_KS_clusters_this_sort)
                    )
                    accuracies = np.zeros(
                        (len(GT_clusters_to_use), num_KS_clusters_this_sort)
                    )
                    # del mu_KS  # free up memory after allocating precisions, recalls, and accuracies
                    (
                        num_matches,
                        num_kilosort_spikes,
                        num_ground_truth_spikes,
                        true_positive_spikes,
                        false_positive_spikes,
                        false_negative_spikes,
                        kilosort_spikes,
                        ground_truth_spikes,
                    ) = [[] for i in range(8)]

                    if parallel:
                        precorrelation_rebin_width_ms_list_element = (
                            precorrelation_rebin_width_ms_list[iRepeat]
                        )
                        preaccuracy_rebin_width_element = preaccuracy_rebin_width[
                            iRepeat
                        ]
                        correlation_alignment_element = correlation_alignment[iRepeat]
                        if parallel == "duo":
                            gtc2u = GT_clusters_to_use
                            clust_iter = [
                                i for j in zip(gtc2u, reversed(gtc2u)) for i in j
                            ][: len(gtc2u)]
                        else:
                            clust_iter = range(len(GT_clusters_to_use))
                        with ProcessPoolExecutor(
                            max_workers=(
                                2
                                if parallel == "duo"
                                else min(mp.cpu_count() // 2, num_motor_units)
                            )
                        ) as executor:
                            futures = [
                                executor.submit(
                                    compute_accuracy_for_each_GT_cluster,
                                    ground_truth_path,
                                    jCluster_GT,
                                    (
                                        None
                                        if iRepeat == 0
                                        else KS_clusters_to_consider[iSort][jCluster_GT]
                                    ),  # choose the best candidate cluster for this GT cluster
                                    (
                                        None
                                        if iRepeat == 0
                                        else KS_clusters_to_consider[iSort]
                                    ),  # need all clusters to know which to check against for spike isolation check
                                    random_seed_entropy,
                                    correlation_alignment_element,
                                    precorrelation_rebin_width_ms_list_element,
                                    preaccuracy_rebin_width_element,
                                    ephys_fs,
                                    time_frame,
                                    list_of_paths_to_sorted_folders[0][iSort],
                                    simulation_method,
                                    use_bins_for_matching=(
                                        True if iRepeat == 0 else False
                                    ),
                                )
                                for jCluster_GT in clust_iter
                            ]
                            results = dict()
                            for future in as_completed(futures):
                                result = future.result()
                                # vvv numpy arrays need to be unpacked differently vvv
                                precisions[result[0], :] = result[1]
                                recalls[result[0], :] = result[2]
                                accuracies[result[0], :] = result[3]
                                results[result[0]] = (
                                    result  # store result for unpacking later
                                )
                                print(
                                    f"Done computing accuracies for GT cluster {result[0]} in sort {sort_dstr} ({iSort+1}/{len(sorts_from_each_path_to_load)} sorts)"
                                )
                        for iKey in sorted(results.keys()):
                            num_matches.append(results[iKey][4])
                            num_kilosort_spikes.append(results[iKey][5])
                            num_ground_truth_spikes.append(results[iKey][6])
                            true_positive_spikes.append(results[iKey][7])
                            false_positive_spikes.append(results[iKey][8])
                            false_negative_spikes.append(results[iKey][9])
                            kilosort_spikes.append(results[iKey][10])
                            ground_truth_spikes.append(results[iKey][11])

                    else:
                        for jCluster_GT in range(len(GT_clusters_to_use)):
                            (
                                _,
                                precisions[jCluster_GT, :],
                                recalls[jCluster_GT, :],
                                accuracies[jCluster_GT, :],
                                *results,
                            ) = compute_accuracy_for_each_GT_cluster(
                                ground_truth_path,
                                jCluster_GT,
                                (
                                    None
                                    if iRepeat == 0
                                    else KS_clusters_to_consider[iSort][jCluster_GT]
                                ),  # choose the best candidate cluster for this GT cluster
                                (
                                    None
                                    if iRepeat == 0
                                    else KS_clusters_to_consider[iSort]
                                ),  # need all clusters to know which to check against for spike isolation check
                                random_seed_entropy,
                                correlation_alignment,
                                precorrelation_rebin_width_ms_list[iRepeat],
                                preaccuracy_rebin_width[iRepeat],
                                ephys_fs,
                                time_frame,
                                list_of_paths_to_sorted_folders[0][iSort],
                                simulation_method,
                                use_bins_for_matching=True if iRepeat == 0 else False,
                            )
                            print(
                                f"Done computing accuracies for GT cluster {jCluster_GT} in sort {sort_dstr} ({iSort+1}/{len(sorts_from_each_path_to_load)} sorts)"
                            )
                            # append the results into each corresponding output list
                            _ = [
                                x.append(y)
                                for x, y in zip(
                                    [
                                        num_matches,
                                        num_kilosort_spikes,
                                        num_ground_truth_spikes,
                                        true_positive_spikes,
                                        false_positive_spikes,
                                        false_negative_spikes,
                                        kilosort_spikes,
                                        ground_truth_spikes,
                                    ],
                                    results,
                                )
                            ]
                    # make lists into numpy arrays
                    num_matches = np.vstack(num_matches)
                    num_kilosort_spikes = np.vstack(num_kilosort_spikes)
                    num_ground_truth_spikes = np.vstack(num_ground_truth_spikes)
                    true_positive_spikes = np.array(true_positive_spikes)
                    false_positive_spikes = np.array(false_positive_spikes)
                    false_negative_spikes = np.array(false_negative_spikes)
                    kilosort_spikes = np.array(kilosort_spikes)
                    ground_truth_spikes = np.array(ground_truth_spikes)

                    # append each stacked array to each corresponding list
                    precisions_list.append(precisions)
                    recalls_list.append(recalls)
                    accuracies_list.append(accuracies)
                    num_matches_list.append(num_matches)
                    num_kilosort_spikes_list.append(num_kilosort_spikes)
                    num_ground_truth_spikes_list.append(num_ground_truth_spikes)
                    true_positive_spikes_list.append(true_positive_spikes)
                    false_positive_spikes_list.append(false_positive_spikes)
                    false_negative_spikes_list.append(false_negative_spikes)
                    kilosort_spikes_list.append(kilosort_spikes)
                    ground_truth_spikes_list.append(ground_truth_spikes)
                    sort_dstr_list.append(sort_dstr)

                    # collect garbage to free up memory from the previous iteration
                    gc.collect()

                # now rename the lists to the original variable names
                precisions = precisions_list
                recalls = recalls_list
                accuracies = accuracies_list
                num_matches = num_matches_list
                num_kilosort_spikes = num_kilosort_spikes_list
                num_ground_truth_spikes = num_ground_truth_spikes_list
                true_positive_spikes = true_positive_spikes_list
                false_positive_spikes = false_positive_spikes_list
                false_negative_spikes = false_negative_spikes_list
                kilosort_spikes = kilosort_spikes_list
                ground_truth_spikes = ground_truth_spikes_list
                print(sort_dstr_list)

                correlations = accuracies  # use accuracies as the metric for matching
                (
                    precisions,
                    recalls,
                    accuracies,
                    num_matches,
                    num_kilosort_spikes,
                    num_ground_truth_spikes,
                    true_positive_spikes,
                    false_positive_spikes,
                    false_negative_spikes,
                    kilosort_spikes,
                    ground_truth_spikes,
                    clusters_in_sort_to_use_list,
                ) = find_best_cluster_matches(
                    correlations,
                    precisions,
                    recalls,
                    accuracies,
                    num_matches,
                    num_kilosort_spikes,
                    num_ground_truth_spikes,
                    true_positive_spikes,
                    false_positive_spikes,
                    false_negative_spikes,
                    kilosort_spikes,
                    ground_truth_spikes,
                    method_for_automatic_cluster_mapping,
                    None if iRepeat == 0 else KS_clusters_to_consider,
                )
                # now rename the lists to the original variable names
                # precisions = precisions_list
                # recalls = recalls_list
                # accuracies = accuracies_list
                # num_matches = num_matches_list
                # num_kilosort_spikes = num_kilosort_spikes_list
                # num_ground_truth_spikes = num_ground_truth_spikes_list
                # true_positive_spikes = true_positive_spikes_list
                # false_positive_spikes = false_positive_spikes_list
                # false_negative_spikes = false_negative_spikes_list
                # kilosort_spikes = kilosort_spikes_list
                # ground_truth_spikes = ground_truth_spikes_list
                KS_clusters_to_consider = clusters_in_sort_to_use_list
                clusters_in_sort_to_use = clusters_in_sort_to_use_list
        elif method_for_automatic_cluster_mapping == "waves":
            true_spike_counts_for_each_cluster = np.load(
                str(ground_truth_path)  # , mmap_mode="r"
            ).sum(axis=0)
            # load and reshape into numchans x whatever (2d array) the data.bin file
            sim_ephys_data = np.memmap(
                str(path_to_sim_dat), dtype="int16", mode="r"
            ).reshape(
                -1, 24
            )  ### WARNING HARDCODED 24 CHANNELS ### !!!
            print(
                "WARNING: HARDCODED 24 CHANNELS IN compute_ground_truth_metrics.py for method 'waves'"
            )

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
                    max_workers=(
                        2
                        if parallel == "duo"
                        else min(mp.cpu_count() // 2, num_motor_units)
                    )
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
                    correlations[jCluster_GT, :] = (
                        compute_train_correlations_for_each_GT_cluster(jCluster_GT)[0]
                    )
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
                        max_workers=(
                            2
                            if parallel == "duo"
                            else min(mp.cpu_count() // 2, len(clusters_in_sort_to_use))
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
                    max_workers=(
                        2
                        if parallel == "duo"
                        else min(mp.cpu_count() // 2, num_motor_units)
                    )
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
                    correlations[jCluster_GT, :] = (
                        compute_spike_time_correlations_for_each_GT_cluster(
                            jCluster_GT
                        )[0]
                    )

        else:
            raise Exception(
                f"method_for_automatic_cluster_mapping must be either 'waves', 'times', or 'trains', but was {method_for_automatic_cluster_mapping}"
            )

        if method_for_automatic_cluster_mapping != "accuracies":
            (
                precisions,
                recalls,
                accuracies,
                num_matches,
                num_kilosort_spikes,
                num_ground_truth_spikes,
                true_positive_spikes,
                false_positive_spikes,
                false_negative_spikes,
                kilosort_spikes,
                ground_truth_spikes,
                clusters_in_sort_to_use_list,
            ) = find_best_cluster_matches(
                correlations,
                precisions,
                recalls,
                accuracies,
                num_matches,
                num_kilosort_spikes,
                num_ground_truth_spikes,
                true_positive_spikes,
                false_positive_spikes,
                false_negative_spikes,
                kilosort_spikes,
                ground_truth_spikes,
                method_for_automatic_cluster_mapping,
            )

            # now rename the lists to the original variable names
            # precisions = precisions_list
            # recalls = recalls_list
            # accuracies = accuracies_list
            # num_matches = num_matches_list
            # num_kilosort_spikes = num_kilosort_spikes_list
            # num_ground_truth_spikes = num_ground_truth_spikes_list
            # true_positive_spikes = true_positive_spikes_list
            # false_positive_spikes = false_positive_spikes_list
            # false_negative_spikes = false_negative_spikes_list
            # kilosort_spikes = kilosort_spikes_list
            # ground_truth_spikes = ground_truth_spikes_list
            clusters_in_sort_to_use = clusters_in_sort_to_use_list
            if parallel:
                with mp.Pool(processes=len(bin_widths_for_comparison)) as pool:
                    zip_obj = zip(
                        [ground_truth_path] * len(bin_widths_for_comparison),
                        [GT_clusters_to_use] * len(bin_widths_for_comparison),
                        [random_seed_entropy] * len(bin_widths_for_comparison),
                        bin_widths_for_comparison,
                        [ephys_fs] * len(bin_widths_for_comparison),
                        [time_frame] * len(bin_widths_for_comparison),
                        [list_of_paths_to_sorted_folders[0]]
                        * len(bin_widths_for_comparison),
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
                        list_of_paths_to_sorted_folders[0],
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
                    num_ground_truth_spikes.append(
                        np.array(num_ground_truth_spikes_temp)
                    )
                    true_positive_spikes.append(np.array(true_positive_spikes_temp))
                    false_positive_spikes.append(np.array(false_positive_spikes_temp))
                    false_negative_spikes.append(np.array(false_negative_spikes_temp))

                # collapse accuracies, precisions, and recalls into a 2D array
                accuracies = np.vstack(accuracies)
                precisions = np.vstack(precisions)
                recalls = np.vstack(recalls)
                iPlot, iSort = iShow, 0
        elif method_for_automatic_cluster_mapping == "accuracies":
            iPlot = iShow
        #     # putting into list will allow indexing with 0
        #     num_ground_truth_spikes = [num_ground_truth_spikes]
        #     num_kilosort_spikes = [num_kilosort_spikes]
        #     precisions = [precisions]
        #     recalls = [recalls]
        #     accuracies = [accuracies]
        #     false_positive_spikes = [false_positive_spikes]
        #     false_negative_spikes = [false_negative_spikes]
        #     true_positive_spikes = [true_positive_spikes]
        #     kilosort_spikes = [kilosort_spikes]
        #     ground_truth_spikes = [ground_truth_spikes]
        # else:
        #     raise Exception("Unexpected error")
    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)
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
        for iSort in range(len(sorts_from_each_path_to_load)):
            plot1(
                num_ground_truth_spikes,
                num_kilosort_spikes,
                precisions,
                recalls,
                accuracies,
                bin_widths_for_comparison[0],
                clusters_in_sort_to_use[iSort],
                GT_clusters_to_use,
                sorts_from_each_path_to_load[iSort],
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

    if show_plot2 or save_png_plot2 or save_html_plot2 or save_svg_plot2:
        ### plot 2: spike trains

        # now plot the spike trains, emphasizing each type of error with a different color
        # include an event plot of the kilosort and ground truth spikes for reference
        for iSort in range(len(sorts_from_each_path_to_load)):
            plot2(
                kilosort_spikes[iSort],
                ground_truth_spikes[iSort],
                false_positive_spikes[iSort],
                false_negative_spikes[iSort],
                true_positive_spikes[iSort],
                bin_widths_for_comparison[0],
                clusters_in_sort_to_use[iSort],
                GT_clusters_to_use,
                sorts_from_each_path_to_load[iSort],
                plot_template,
                show_plot2,
                plot2_xlim,
                save_png_plot2,
                save_svg_plot2,
                save_html_plot2,
                # make figsize 1080p
                figsize=(1920, 1080),
                sort_index=iSort,
            )

    if show_plot3 or save_png_plot3 or save_html_plot3 or save_svg_plot3:
        ### plot 3: comparison across bin widths, ground truth metrics
        # the bin width for comparison is varied from 0.25 ms to 8 ms, doubling each time
        # the metrics are computed for each bin width
        for iSort in range(len(sorts_from_each_path_to_load)):
            plot3(
                bin_widths_for_comparison[0],
                precisions,
                recalls,
                accuracies,
                num_motor_units,
                clusters_in_sort_to_use[iSort],
                GT_clusters_to_use,
                sorts_from_each_path_to_load[iSort],
                plot_template,
                show_plot3,
                save_png_plot3,
                save_svg_plot3,
                save_html_plot3,
                # make figsize 1080p
                figsize=(1920, 1080),
            )

    if show_plot4 or save_png_plot4 or save_html_plot4 or save_svg_plot4:
        ### plot 4: analogous to plot 3, but for each different run, processing results from each
        # sort in the list of paths to sorted folders
        # for iSort in range(len(sorts_from_each_path_to_load)):
        plot4(
            bin_widths_for_comparison[0],
            precisions,
            recalls,
            accuracies,
            num_motor_units,
            clusters_in_sort_to_use[iSort],
            GT_clusters_to_use,
            sorts_from_each_path_to_load,
            plot_template,
            show_plot4,
            save_png_plot4,
            save_svg_plot4,
            save_html_plot4,
            # make figsize 1080p
            figsize=(1080, 1080),
        )

    if show_plot5 or save_png_plot5 or save_html_plot5 or save_svg_plot5:
        ### plot 5: examples of overlaps throughout sort to validate results
        for iSort in range(len(sorts_from_each_path_to_load)):
            plot5(
                bin_widths_for_comparison[0],
                precisions,
                recalls,
                accuracies,
                nt0,
                num_motor_units,
                clusters_in_sort_to_use[iSort],
                GT_clusters_to_use,
                sorts_from_each_path_to_load[iSort],
                plot_template,
                show_plot5,
                save_png_plot5,
                save_svg_plot5,
                save_html_plot5,
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
