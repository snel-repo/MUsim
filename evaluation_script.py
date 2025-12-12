# IMPORT packages
import copy
import tempfile
from datetime import datetime
from pathlib import Path
from pdb import set_trace

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame as df
from scipy.signal import correlate, correlation_lags

start_time = datetime.now()  # begin timer for script execution time


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


def load_npy_files_from_folder(folder_path, memmap=False):
    """
    Function loads spike_times.npy and spike_clusters.npy and returns them as numpy arrays.
    """
    folder_path = Path(folder_path)
    spike_times_path = Path(folder_path / "spike_times.npy")
    spike_clusters_path = Path(folder_path / "spike_clusters.npy")
    spike_times = (
        np.load(spike_times_path, mmap_mode="r")
        if memmap
        else np.load(spike_times_path)
    )
    spike_clusters = (
        np.load(spike_clusters_path, mmap_mode="r")
        if memmap
        else np.load(spike_clusters_path)
    )
    if 0 not in np.unique(spike_clusters):
        spike_clusters -= 1
    # add axis if not at least 2D
    if len(spike_times.shape) < 2 and len(spike_clusters.shape) < 2:
        print("correcting dim's of loaded data, adding a dimension")
        spike_times = np.expand_dims(spike_times, axis=-1)
        spike_clusters = np.expand_dims(spike_clusters, axis=-1)
        print(f"new shapes are")
        print("spike_times", spike_times.shape)
        print("spike_clusters", spike_clusters.shape)
    return spike_times, spike_clusters


class MUsim:
    def __init__(self, random_seed=False):
        self.memmap = False
        self.spikes = []
        # default number of units
        self.num_units = 10
        # default number of trials
        self.num_trials = 50
        # amount of time per trial is (num_bins_per_trial/sample_rate)
        self.num_bins_per_trial = 30000
        # Hz, default sampling rate. it's inverse results in the time alloted to each bin
        self.sample_rate = 30000
        if random_seed:
            if type(random_seed) == int:
                # print("random_seed is an integer.")
                self.MUseed = random_seed
                self.RNG = np.random.default_rng(self.MUseed)
                self.MUseed_sqn = self.RNG.bit_generator._seed_seq
            elif type(random_seed) == np.random.SeedSequence:
                # print("random_seed is a SeedSequence object.")
                self.MUseed_sqn = random_seed
                self.MUseed = self.MUseed_sqn.entropy
                self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(1))
            elif type(random_seed) == np.random.Generator:
                # print("random_seed is a Generator object.")
                self.RNG = random_seed
                self.MUseed_sqn = self.RNG.bit_generator._seed_seq
                self.MUseed = self.MUseed_sqn.entropy
            else:
                raise Exception(
                    "random_seed must be either a BitGenerator, SeedSequence or an integer."
                )
        else:
            self.MUseed_sqn = np.random.SeedSequence()
            self.MUseed = self.MUseed_sqn.entropy
            self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(1))

    def __repr__(self):
        return f"MUsim object with {self.num_units} units, {len(self.spikes)} trials across {len(self.session)} sessions."

    def copy(self):
        """
        Function returns a shallow copy of the MUsim object.
        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Function returns a deep copy of the MUsim object.
        """
        return copy.deepcopy(self)

    def _create_trial_from_kilosort_files(self, spike_times, spike_clusters):
        """
        Function loads data from a kilosort output files and creates a trial from it. It will load the
        spike times and spike clusters from spike_times.npy and spike_clusters.npy, respectively.
        It will then create a MUsim format trial from this data.
        """

        units_in_sort = np.sort(np.unique(spike_clusters).astype(int))
        largest_cluster_number = units_in_sort[-1]
        # get number of bins
        num_bins = int(
            np.round(spike_times[-1][0]) + 1
        )  # use last spike time because it is the largest in time

        # create empty trial
        if self.memmap:
            # do not load the trial into ram, as it may be very large
            # make a memmap to a temporary file using tempfile library
            trial = np.memmap(
                tempfile.TemporaryFile(),
                dtype=np.float32,
                mode="w+",
                shape=(num_bins, largest_cluster_number + 1),
            )
        else:
            trial = np.zeros((num_bins, largest_cluster_number + 1), dtype=np.float32)

        # fill trial with spikes in order of KS cluster number
        for ii in range(largest_cluster_number + 1):
            unit_spike_times = spike_times[spike_clusters == ii]
            if len(unit_spike_times) > 0:
                trial[unit_spike_times, ii] = 1
            else:
                trial[:, ii] = np.nan  # no spikes for this cluster number

        return trial

    def load_MUs(
        self,
        spike_times,
        spike_clusters,
        recording_bin_width,
        slice=[0, 1],
    ):
        """
        Function loads data and appends into MUsim().session, just like with simulated sessions.
        Data axes are transposed to make structure compatible. Transpose operation:
        (Trials x Time x MUs) --> (Time x MUs x Trials)
        """

        # check inputs
        assert isinstance(spike_times, np.ndarray), "spike_times must be a numpy array."
        assert isinstance(
            spike_clusters, np.ndarray
        ), "spike_clusters must be a numpy array."

        # use _create_trial_from_kilosort() to create a trial from kilosort output
        trial_from_KS = self._create_trial_from_kilosort_files(
            spike_times, spike_clusters
        )
        sliced_trial_from_KS = trial_from_KS[
            int(np.round(trial_from_KS.shape[0] * slice[0])) : int(
                np.round(trial_from_KS.shape[0] * slice[1])
            ),
            :,
        ]
        self.spikes.append(
            sliced_trial_from_KS.copy()
        )  # spikes in order of KS cluster number
        del trial_from_KS, sliced_trial_from_KS  # clear large variables
        self.num_trials = 1
        self.MUmode = "loaded"  # record that this session was loaded

        self.bin_width = recording_bin_width
        self.sample_rate = 1 / recording_bin_width
        self.num_bins_per_trial = self.spikes[-1].shape[0]
        self.num_units = self.spikes[-1].shape[1]
        return

    def rebin(self, new_bin_width, index=-1):
        """
        Function re-bins the data in MUsim().spikes[-1] or MUsim().session[-1] to a new bin width.
        To do this, all spikes in the original bins are indexed with new spacing and matrix
        multiplication (no for loops). For each unit, all spikes in previous bins are summed
        to create each new bin value.
        Input:
            new_bin_width: new bin width to re-bin the data to, in seconds.
            index: index of the trial or session to re-bin. Default is -1,
            which re-bins the last trial or session.

        Returns: nothing.
        """
        if new_bin_width == self.bin_width:
            return  # no need to rebin if bin width is the same
        spikes = self.spikes[index]
        new_num_bins = int(np.round(spikes.shape[0] * self.bin_width / new_bin_width))
        # get new bin edges as floats
        new_bin_edges = np.linspace(0, new_num_bins * new_bin_width, new_num_bins + 1)
        # now snap all edges to nearest integer using existing sample rate
        new_bin_indexes = np.round(new_bin_edges * self.sample_rate).astype(int)
        new_spikes = np.zeros(
            (new_num_bins, spikes.shape[1]), dtype=float
        )  # float to allow nan
        for ii in range(new_num_bins):
            new_spikes[ii, :] = np.sum(
                spikes[new_bin_indexes[ii] : new_bin_indexes[ii + 1], :], axis=0
            )
        # update bin width
        self.bin_width = new_bin_width
        # update number of bins
        self.num_bins_per_trial = new_num_bins
        # update sample rate
        self.sample_rate = 1 / new_bin_width
        # update spikes
        if self.memmap:
            self.spikes[index] = np.memmap(
                tempfile.TemporaryFile(),
                dtype=np.float32,
                mode="w+",
                shape=(new_num_bins, spikes.shape[1]),
            )
        else:
            self.spikes[index] = np.zeros(
                (new_num_bins, spikes.shape[1]), dtype=np.float32
            )
        self.spikes[index][:] = new_spikes
        return


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


def find_best_cluster_matches(
    scoring_metric_list,
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
    KS_clusters_to_consider=None,
):

    if KS_clusters_to_consider is None:
        clusters_in_sort_to_use_list = []
    else:
        clusters_in_sort_to_use_list = KS_clusters_to_consider

    for iSort in range(len(scoring_metric_list)):
        if KS_clusters_to_consider is None:
            # now find the cluster with the highest correlation
            sorted_cluster_pair_corr_idx = np.unravel_index(
                np.argsort(scoring_metric_list[iSort].ravel()),
                scoring_metric_list[iSort].shape,
            )

            # make sure to slice off coordinate pairs with a nan result
            num_nan_pairs = np.isnan(np.sort(scoring_metric_list[iSort].ravel())).sum()
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
            # [uniq_KS, uniq_KS_idx, uniq_KS_inv_idx] = np.unique(
            #     GT_mapped_idxs[:, 1], return_index=True, return_inverse=True
            # )

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
                scoring_metric_list[iSort][idx_pair[0], idx_pair[1]]
                for idx_pair in GT_mapped_idxs
            ]

            GT_mapped_precisions = [
                precisions[iSort][idx_pair[0], idx_pair[1]]
                for idx_pair in GT_mapped_idxs
            ]
            GT_mapped_recalls = [
                recalls[iSort][idx_pair[0], idx_pair[1]] for idx_pair in GT_mapped_idxs
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
                GT_mapped_num_KS_spikes = num_kilosort_spikes[iSort].flatten().tolist()
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
            num_kilosort_spikes[iSort] = np.array(GT_mapped_num_KS_spikes).astype(int)
            # they are copies, so just take the first one
            num_ground_truth_spikes[iSort] = num_ground_truth_spikes[iSort][0]
            ground_truth_spikes[iSort] = np.array(ground_truth_spikes[iSort])[0]

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

            clusters_in_sort_to_use = GT_mapped_idxs[:, 1]
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

        unit_df = df()
        unit_df["Unit"] = np.array(clusters_in_sort_to_use_list[iSort]).astype(int)
        unit_df["True Count"] = num_ground_truth_spikes[iSort][GT_clusters_to_use]
        unit_df["KS Count"] = num_kilosort_spikes[iSort]
        unit_df["Precision"] = precisions[iSort]
        unit_df["Recall"] = recalls[iSort]
        unit_df["Accuracy"] = accuracies[iSort]
        unit_df.set_index("Unit", inplace=True)

        # print metrics for each unit
        print("\n")  # add a newline for readability
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


def compute_accuracy_for_each_GT_cluster(
    GT_spike_times,
    GT_spike_clusters,
    KS_spike_times,
    KS_spike_clusters,
    iRepeat,  # iteration counter
    num_repeats,  # total number of iterations needed
    jCluster_GT,  # index of the GT cluster to compare against
    KS_clusters_to_consider,  # list of candidate clusters for this GT cluster
    all_matched_KS_clusters,  # list of all matched KS clusters
    random_seed_entropy,
    correlation_alignment,
    precorrelation_rebin_width_ms,
    preaccuracy_rebin_width,
    ephys_fs,
    time_frame,
    use_bins_for_matching=True,
):

    # use MUsim object to load and rebin ground truth data
    mu_GT = MUsim(random_seed_entropy)
    mu_GT.load_MUs(
        GT_spike_times,
        GT_spike_clusters,
        1 / ephys_fs,
        slice=time_frame,
    )
    mu_GT_bin_width_ms_orig = mu_GT.bin_width * 1000

    # use MUsim object to load and rebin Kilosort data
    mu_KS = MUsim(random_seed_entropy)
    mu_KS.load_MUs(
        KS_spike_times,
        KS_spike_clusters,
        1 / ephys_fs,
        slice=time_frame,
    )
    mu_KS_bin_width_ms_orig = mu_KS.bin_width * 1000
    if all_matched_KS_clusters is not None:
        mu_KS_other = mu_KS.deepcopy()
        all_other_KS_clusters = np.setdiff1d(
            all_matched_KS_clusters, KS_clusters_to_consider
        )
        mu_KS_other.spikes[-1] = mu_KS_other.spikes[-1][:, all_other_KS_clusters]
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
                    mu_GT.spikes[-1].shape[0] - mu_KS_other.spikes[-1].shape[0],
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

        min_delay_ms = -2  # ms
        max_delay_ms = 2  # ms

        if precorrelation_rebin_width_ms is not None:
            # precorrelation alignment rebinning
            mu_GT.rebin(
                precorrelation_rebin_width_ms / 1000
            )  # rebin to rebin_width ms bins
            mu_KS.rebin(
                precorrelation_rebin_width_ms / 1000
            )  # rebin to rebin_width ms bins
            mu_KS_other.rebin(precorrelation_rebin_width_ms / 1000)

            min_delay_samples = int(round(min_delay_ms / precorrelation_rebin_width_ms))
            max_delay_samples = int(round(max_delay_ms / precorrelation_rebin_width_ms))
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
        assert spike_isolation_radius_ms > 0, "spike_isolation_radius_ms must be >0"

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
            f"Total average overlap fraction is {GT_spike_counts_after.sum() / GT_spike_counts_before.sum()}"
        )
        # concatenate the mu_KS_other spikes into mu_KS.spikes[-1] before removing isolated spikes
        if all_matched_KS_clusters is not None:
            # apply same roll to mu_KS_other.spikes
            mu_KS_other.spikes[-1] = np.roll(mu_KS_other.spikes[-1], -shift, axis=0)
            mu_KS.spikes[-1] = np.hstack((mu_KS.spikes[-1], mu_KS_other.spikes[-1]))
        mu_KS = remove_isolated_spikes(mu_KS, spike_isolation_radius_pts)
        # make sure there are no isolated spikes remaining in mu_KS by taking the sum of a rolling 1ms window to make sure it never equals 1 (only zero or >1)
        # only take the sum if a spike is in the center time point though
        for iW in range(len(mu_KS.spikes[-1]) - 2 * spike_isolation_radius_pts - 1):
            spike_in_center_time_point = (
                mu_KS.spikes[-1][iW + spike_isolation_radius_pts + 1].sum() >= 1
            )
            if spike_in_center_time_point:
                num_spk_in_window = np.sum(
                    mu_KS.spikes[-1][iW : iW + 2 * spike_isolation_radius_pts + 1]
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
        mu_GT.rebin(preaccuracy_rebin_width / 1000)  # rebin to rebin_width ms bins

        mu_KS.rebin(preaccuracy_rebin_width / 1000)  # rebin to rebin_width ms bins

        kilosort_spikes = mu_KS.spikes[-1]  # shape is (num_bins, num_units)
        ground_truth_spikes = mu_GT.spikes[-1]  # shape is (num_bins, num_units)

        # if kilosort spike length is greater, trim it to match GT
        if kilosort_spikes.shape[0] > ground_truth_spikes.shape[0]:
            print(
                f"Shape mismatch after rebinning, trimming KS spikes time dimension down to {ground_truth_spikes.shape[0]}"
            )
            kilosort_spikes = kilosort_spikes[: ground_truth_spikes.shape[0]]

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
            - ground_truth_spikes_this_clust_repeat[false_positive_large_mask]
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
        GT_spike_idxs = np.where(ground_truth_spikes[:, jCluster_GT, np.newaxis])[0]
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
        false_positive_spikes[KS_spike_idxs[np.where(spike_dt_masked.sum(0).mask)]] = 1
        # any rows with only masked values is a GT spike time with no KS match
        false_negative_spikes[GT_spike_idxs[np.where(spike_dt_masked.sum(1).mask)]] = 1

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
    recall = compute_recall(num_matches, num_ground_truth_spikes, jCluster_GT)
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


def run_cluster_matching_and_evaluation(
    GT_spike_times,
    GT_spike_clusters,
    KS_spike_times,
    KS_spike_clusters,
    GT_clusters_to_use,
    KS_clusters_to_consider,
    num_repeats,
    correlation_alignment,
    precorrelation_rebin_width_ms_list,
    preaccuracy_rebin_width,
    ephys_fs,
    time_frame,
    random_seed_entropy,
    parallel,
):

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

        num_KS_clusters_this_sort = np.unique(KS_spike_clusters).shape[0]

        precisions = np.zeros((len(GT_clusters_to_use), num_KS_clusters_this_sort))
        recalls = np.zeros((len(GT_clusters_to_use), num_KS_clusters_this_sort))
        accuracies = np.zeros((len(GT_clusters_to_use), num_KS_clusters_this_sort))
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
            preaccuracy_rebin_width_element = preaccuracy_rebin_width[iRepeat]
            correlation_alignment_element = correlation_alignment[iRepeat]
            if parallel == "duo":
                gtc2u = GT_clusters_to_use
                clust_iter = [i for j in zip(gtc2u, reversed(gtc2u)) for i in j][
                    : len(gtc2u)
                ]
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
                        GT_spike_times,
                        GT_spike_clusters,
                        KS_spike_times,
                        KS_spike_clusters,
                        iRepeat,  # iteration counter
                        num_repeats,  # total number of iterations needed
                        jCluster_GT,
                        (
                            None
                            if iRepeat == 0
                            else KS_clusters_to_consider[0][jCluster_GT]
                        ),  # choose the best candidate cluster for this GT cluster
                        (
                            None if iRepeat == 0 else KS_clusters_to_consider[0]
                        ),  # need all clusters to know which to check against for spike isolation check
                        random_seed_entropy,
                        correlation_alignment_element,
                        precorrelation_rebin_width_ms_list_element,
                        preaccuracy_rebin_width_element,
                        ephys_fs,
                        time_frame,
                        use_bins_for_matching=(
                            True if iRepeat == 0 else False
                        ),  # do not use bins on second loop
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
                    results[result[0]] = result  # store result for unpacking later
                    print(f"Done computing accuracies for GT cluster {result[0]}")
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
                    GT_spike_times,
                    GT_spike_clusters,
                    KS_spike_times,
                    KS_spike_clusters,
                    iRepeat,  # iteration counter
                    num_repeats,  # total number of iterations needed
                    jCluster_GT,
                    (
                        None
                        if iRepeat == 0
                        else KS_clusters_to_consider[0][jCluster_GT]
                    ),  # choose the best candidate cluster for this GT cluster
                    (
                        None if iRepeat == 0 else KS_clusters_to_consider[0]
                    ),  # need all clusters to know which to check against for spike isolation check
                    random_seed_entropy,
                    correlation_alignment,
                    precorrelation_rebin_width_ms_list[iRepeat],
                    preaccuracy_rebin_width[iRepeat],
                    ephys_fs,
                    time_frame,
                    use_bins_for_matching=(
                        True if iRepeat == 0 else False
                    ),  # do not use bins on second loop
                )
                print(f"Done computing accuracies for GT cluster {jCluster_GT}")
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

        scoring_metric_list = accuracies  # use accuracies as the metric for matching
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
            scoring_metric_list,
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
            None if iRepeat == 0 else KS_clusters_to_consider,
        )
        KS_clusters_to_consider = clusters_in_sort_to_use_list
        # clusters_in_sort_to_use = clusters_in_sort_to_use_list

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
    )


def plot1(
    num_ground_truth_spikes,
    num_kilosort_spikes,
    precision,
    recall,
    accuracy,
    bin_width_for_comparison,
    GT_clusters_to_use,
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
    show_plot1c,
    save_png_plot1c,
    save_svg_plot1c,
    save_html_plot1c,
    spike_isolation_radius_ms,
    sort_type,
    KS_session_folder,
    figsize=(1920, 1080),
):
    iSort = 0
    if show_plot1a or save_png_plot1a or save_svg_plot1a or save_html_plot1a:
        fig1a = go.Figure()
        fig1a.add_trace(
            go.Scatter(
                x=np.arange(0, num_motor_units),
                y=precision[iSort],
                mode="lines+markers",
                name="Precision",
                line=dict(width=15, color="green"),
                marker=dict(size=35),
            )
        )
        fig1a.add_trace(
            go.Scatter(
                x=np.arange(0, num_motor_units),
                y=recall[iSort],
                mode="lines+markers",
                name="Recall",
                line=dict(width=15, color="crimson"),
                marker=dict(size=35),
            )
        )
        fig1a.add_trace(
            go.Scatter(
                x=np.arange(0, num_motor_units),
                y=accuracy[iSort],
                mode="lines+markers",
                name="Accuracy",
                line=dict(width=15, color="orange"),
                marker=dict(size=35),
            )
        )

        # make the title shifted higher up,
        # make text much larger
        fig1a.update_layout(
            title={
                "text": f"<b>Comparison of {sort_type} Performance to Ground Truth, {bin_width_for_comparison} ms Bins</b>",
            },
            xaxis_title="<b>GT Cluster ID,<br>True Count</b>",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template=plot_template,
            yaxis=dict(
                title="<b>Metric Score</b>",
                title_standoff=1,
                range=[i / 100 for i in plot1_ylim],
            ),
        )
        # update the x tick label of the bar graph to match the cluster ID
        fig1a.update_xaxes(
            ticktext=[
                f"<b>Unit {GT_clusters_to_use[iUnit]},<br>{str(round(num_ground_truth_spikes[iSort][iUnit]/1000,1))}k</b>"
                for iUnit in range(num_motor_units)
            ],
            tickvals=np.arange(0, num_motor_units),
            tickfont=dict(size=32, family="Open Sans", color="black", weight="bold"),
            showgrid=True,
            gridcolor="grey",
        )
        fig1a.update_yaxes(
            tickfont=dict(size=32, family="Open Sans", color="black", weight="bold"),
            showgrid=True,
            gridcolor="grey",
        )
    if show_plot1b or save_png_plot1b or save_svg_plot1b or save_html_plot1b:
        # make text larger
        fig1b = go.Figure(
            layout=go.Layout(
                yaxis=dict(
                    title_standoff=10,
                ),
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
                    y=100
                    * num_kilosort_spikes[iSort]
                    / num_ground_truth_spikes[iSort],
                    name="% True Spike Count",
                    marker_color="black",
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
            line_width=20,
            line_dash="dash",
            line_color="firebrick",
            # yref="y2",
            name="100% Spike Count",
        )
        # make all the text way larger
        fig1b.update_layout(
            title={
                "text": f"<b>True Spike Count Captured for Each Cluster Using {sort_type}, {bin_width_for_comparison} ms Bins</b><br>",
                # "y": 0.95,
            },
            xaxis_title="<b>GT Cluster ID,<br>True Count</b>",
            template=plot_template,
            yaxis=dict(
                title=bar_yaxis_title,
                showgrid=False,
            ),
        )

        # update the x tick label of the bar graph to match the cluster ID
        fig1b.update_xaxes(
            ticktext=[
                f"<b>Unit {GT_clusters_to_use[iUnit]},<br>{str(np.round(num_ground_truth_spikes[iSort][iUnit]/1000,1))}k<b>"
                for iUnit in range(num_motor_units)
            ],
            tickvals=np.arange(0, num_motor_units),
            tickfont=dict(size=32, family="Open Sans"),
        )
        fig1b.update_layout(yaxis_range=plot1_ylim)
        # make y axis bold
        fig1b.update_yaxes(
            title_font=dict(
                size=32, family="Open Sans", color="black", weight="bold"
            ),
            tickfont=dict(
                size=32, family="Open Sans", color="black", weight="bold"
            ),
        )
    if show_plot1c or save_png_plot1c or save_svg_plot1c or save_html_plot1c:
        # make text larger
        fig1c = go.Figure(
            layout=go.Layout(
                yaxis=dict(
                    # title_font=dict(size=14, family="Open Sans"),
                    title_standoff=10,
                ),
                # title_font=dict(size=18),
            )
        )

        fig1c.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=0,  # num_ground_truth_spikes[iSort].min(),
            y0=0,  # num_ground_truth_spikes[iSort].min(),
            x1=num_ground_truth_spikes[iSort].max(),
            y1=num_ground_truth_spikes[iSort].max(),
            line=dict(
                color="black",
                width=10,
                dash="dot",
            ),
        )

        fig1c.add_trace(
            go.Scatter(
                x=num_ground_truth_spikes[iSort],
                y=num_kilosort_spikes[iSort],
                mode="markers",
                name="Count",
                marker=dict(
                    size=35,
                    color=("firebrick" if sort_type == "EMUsort" else "black"),
                ),
                # yaxis="y2",
            )
        )
        axis_type = "linear"
        fig1c.update_xaxes(type=axis_type, showgrid=True)
        fig1c.update_yaxes(
            type=axis_type,
            showgrid=True,
            scaleanchor="x",
            scaleratio=1,
        )
        fig1c.update_layout(
            title={
                "text": f"<b>Spike Counts Captured for Each Cluster</b>",
                # "y": 0.95,
            },
            xaxis=dict(title="<b>True Spike Count</b>"),
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
                title="<b>Sorted Spike Count</b>",
                # title_standoff=1,
                # anchor="free",
                # autoshift=True,
                # shift=-30,
                # side="right",
                # showgrid=False,
            ),
        )

    if save_png_plot1a:
        fig1a.write_image(
            f"plot1/plot1a_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot1a:
        fig1a.write_image(
            f"plot1/plot1a_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot1a:
        fig1a.write_html(
            f"plot1/plot1a_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot1a:
        fig1a.show()

    if save_png_plot1b:
        fig1b.write_image(
            f"plot1/plot1b_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot1b:
        fig1b.write_image(
            f"plot1/plot1b_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot1b:
        fig1b.write_html(
            f"plot1/plot1b_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot1b:
        fig1b.show()

    if save_png_plot1c:
        fig1c.write_image(
            f"plot1/plot1c_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}_{axis_type}.png",
            width=figsize[0],
            height=figsize[1],
        )
    if save_svg_plot1c:
        fig1c.write_image(
            f"plot1/plot1c_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}_{axis_type}.svg",
            width=figsize[0],
            height=figsize[1],
        )
    if save_html_plot1c:
        fig1c.write_html(
            f"plot1/plot1c_spkRad_{str(spike_isolation_radius_ms)}_{bin_width_for_comparison}ms_{sort_type}_{KS_session_folder.name}_{axis_type}.html",
            include_plotlyjs="cdn",
            full_html=False,
        )
    if show_plot1c:
        fig1c.show()


if __name__ == "__main__":
    # set parameters
    parallel = True  # can set to True, False, and "duo", which executes 2 processes at a time, with a clever ordering of pairs to keep memory consumption down
    time_frame = [0, 1]  # must be between 0 and 1
    ephys_fs = 30000  # Hz
    bin_widths_for_comparison = [0.1]  # only affects plotting
    spike_isolation_radius_ms = 1  # radius of isolation of a spike for it to be removed from consideration. set to positive float, integer, or set None to disable

    nt0 = 121  # number of time bins in the template, in ms it is 3.367, only used if method_for_automatic_cluster_mapping is "waves"
    random_seed_entropy = 218530072159092100005306709809425040261  # 75092699954400878964964014863999053929  # int

    ## plot settings
    sort_type = "MUedit"
    plot_template = "plotly_white"  # ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
    plot1_bar_type = "percent"  # totals / percent
    plot1_ylim = [-10, 120]
    show_plot1a = False
    show_plot1b = False
    show_plot1c = True
    save_png_plot1a = False
    save_png_plot1b = False
    save_png_plot1c = False
    save_svg_plot1a = False
    save_svg_plot1b = False
    save_svg_plot1c = False
    save_html_plot1a = False
    save_html_plot1b = False
    save_html_plot1c = False

    ## set ground truth data folder path
    GT_folder = Path(
        "./spikes_files/spikes_20250429-202657_godzilla_20221116_10MU_8CH_SNR-1-from_data_jitter-0.2std_method-KS_templates_12-files" # test of sameness with boolean .npy file
        # "/home/smoconn/git/MUsim/spikes_files/spikes_20241203-120158_godzilla_20221117_10MU_SNR-1-from_data_jitter-2.0std_method-KS_templates_12-files"
        # "/home/smamid3/MUEdit Convert/MUEdit Convert/ground_truth_2.25"
        # "/home/smoconn/git/MUsim/spikes_files/spikes_20240607-143039_godzilla_20221117_10MU_SNR-1-from_data_jitter-0std_method-KS_templates_12-files_20250306-180027"
    )
    # if ".npy" in GT_folder:
    #     GT_folder = Path().joinpath("spikes_files", GT_folder)
    # set which ground truth clusters to compare with (a range from 0 to num_motor_units)
    num_motor_units = 10
    GT_clusters_to_use = list(range(0, num_motor_units))

    ## load Kilosort data
    # paths to the folders containing the Kilosort data
    KS_session_folder = Path(
        # "./MUedit/run1/20250429-202658" # MUedit paper results
        "./MUedit/run2" # MUedit paper results
        # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/paper_evals/CHs_8_MUs_10/sim_noise_0.2_orig_CHs/sorted_20250429_205402741549_sim_noise_0.2_orig_CHs_Th_5,2_spkTh_6,9_SCORE_0.523"
        # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/litmus2/sorted_20241203_153703193461_litmus2"
        # "/home/smamid3/MUEdit Convert/MUEdit Convert/mu_edit_2.25_sort"
        # "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/siemu_test/sim_2022-11-17_17-08-07_shape_noise_2.25/sorted_20250224_171039931888_sim_2022-11-17_17-08-07_shape_noise_2.25_Th_5,2_spkTh_6,9_SCORE_0.540"
    )
    # clusters_to_take_from = [24, 2, 3, 1, 23, 26, 0, 4, 32, 27]  # 0-indexed

    # load the spike times and clusters from the ground truth and Kilosort data folders
    GT_spike_times, GT_spike_clusters = load_npy_files_from_folder(GT_folder)
    KS_spike_times, KS_spike_clusters = load_npy_files_from_folder(KS_session_folder)
    print("updated")
    # add dummies to KS if below GT
    num_dummies = len(np.unique(GT_spike_clusters)) - len(np.unique(KS_spike_clusters))
    print(f"number of dummy clusters is {num_dummies}")
    if num_dummies > 0:
        highest_KS_cluster = np.max(np.unique(KS_spike_clusters))
        dum_clusters = (
            np.arange(highest_KS_cluster + 1, highest_KS_cluster + num_dummies + 1)
        )
        dum_times = int(60000)+np.zeros_like(
            dum_clusters
        ).astype(int)  # np.random.randint(np.max(GT_spike_times), size=dum_clusters.shape)
        KS_spike_clusters = np.expand_dims(
            np.insert(KS_spike_clusters, 0, dum_clusters), -1
        )
        KS_spike_times = np.expand_dims(np.insert(KS_spike_times, 0, dum_times), -1)
    # set_trace()
    # parameters for different settings across repeats
    # only do correlation alignment during 2nd pass
    precorrelation_rebin_width_ms_list = [None, 0.1]
    correlation_alignment = [False, True]
    preaccuracy_rebin_width = [10, 1]
    num_repeats = len(correlation_alignment)
    KS_clusters_to_consider = None

    if parallel:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
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
    ) = run_cluster_matching_and_evaluation(
        GT_spike_times,
        GT_spike_clusters,
        KS_spike_times,
        KS_spike_clusters,
        GT_clusters_to_use,
        KS_clusters_to_consider,
        num_repeats,
        correlation_alignment,
        precorrelation_rebin_width_ms_list,
        preaccuracy_rebin_width,
        ephys_fs,
        time_frame,
        random_seed_entropy,
        parallel,
    )

    if (
        show_plot1a
        or save_png_plot1a
        or save_html_plot1a
        or save_svg_plot1a
        or show_plot1b
        or save_png_plot1b
        or save_html_plot1b
        or save_svg_plot1b
        or show_plot1c
        or save_png_plot1c
        or save_html_plot1c
        or save_svg_plot1c
    ):
        ### plot 1: bar plot of spike counts
        # now create an overlay plot of the two plots above. Do not use subplots, but use two y axes
        # make bar plot of total spike counts use left y axis
        plot1(
            num_ground_truth_spikes,
            num_kilosort_spikes,
            precisions,
            recalls,
            accuracies,
            bin_widths_for_comparison[0],
            GT_clusters_to_use,
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
            show_plot1c,
            save_png_plot1c,
            save_svg_plot1c,
            save_html_plot1c,
            spike_isolation_radius_ms,
            # make figsize 1080p
            sort_type,
            KS_session_folder,
            figsize=(1000, 1000),
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
