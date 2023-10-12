import pdb
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import expit, logit


class MUsim:
    def __init__(self, random_seed=False):
        self.MUmode = "static"  # "static" for size-principle obediance, "dynamic" for yank-dependent thresholds
        self.MUactivation = "sigmoid"  # "sigmoid"/"heaviside" activation function of MUs
        self.MUthresholds_dist = "exponential"  # can be either "exponential", "uniform" or "normal". Distributes proportion of small vs large MUs
        self.MUspike_dynamics = "poisson"  # can be either "poisson", "independent", or "spike_history". Models spiking behavior as poisson process or as independent
        # fractions of MU thresholds that will flip during a dynamic mode simulation
        self.MUreversal_frac = 1
        # will force the units at these indexes to remain static during the dynamic simulations
        self.MUreversal_static_units = []
        # variable used to track which MUs reverse threshold (nan until assigned in dynamic mode)
        self.static_MU_idxs = np.nan
        # units[0] holds all MU thresholds, .units[1] holds all MU response curves, .units[2] holds all MU minimum ISI (poisson lambda)
        self.units = [[], [], []]
        # minimum ISI (refractory period) for each unit, in ms (only used when MUspike_dynamics=="poisson")
        self.refractory_period = 1
        # spiking responses appended here each time .simulate_spikes() is called
        self.spikes = []
        # matrix of spiking responses appended to this list here each time .simulate_session() is called
        self.session = []
        # current force profile appended to this list here each time .simulate_session() is called
        self.session_forces = []
        # current yank profile appended to this list here each time .simulate_session() is called
        self.session_yanks = []
        # num trials for each session
        self.session_num_trials = []
        # MU response curves for each session
        self.session_response_curves = []
        # MU activation functions for each session
        self.session_MUactivations = []
        # save random seed for each session
        self.session_MUseed = []
        # smoothed spiking responses appended here each time .convolve() is called
        self.smooth_spikes = []
        # matrix of smoothed spiking responses appended here each time .convolve(target=session) is called
        self.smooth_session = []
        # default number of units to simulate
        self.num_units = 10
        # default number of trials to simulate
        self.num_trials = 50
        # amount of time per trial is (num_bins_per_trial/sample_rate)
        self.num_bins_per_trial = 30000
        # Hz, default sampling rate. it's inverse results in the time alloted to each bin
        self.sample_rate = 30000
        self.bin_width = 1 / self.sample_rate
        self.default_max_force = 5
        # define initial force profile
        self.init_force_profile = np.linspace(0, self.default_max_force, self.num_bins_per_trial)
        # initialize force profile
        self.force_profile = self.init_force_profile
        # define initial yank profile
        self.init_yank_profile = np.round(
            self.sample_rate * np.diff(self.force_profile), decimals=10
        )
        # repeat last value to match force profile length
        self.init_yank_profile = np.append(self.init_yank_profile, self.init_yank_profile[-1])
        # initialize yank profile
        self.yank_profile = self.init_yank_profile
        # default yank value at which the thresholds flips
        self.yank_flip_thresh = 20
        # only used when MUspike_dynamics=="independent", set to achieve ~80hz (simulate realistic MU firing rates)
        self.max_spike_prob = 0.08
        # fixed minimum threshold for the generated units' response curves
        self.threshmin = 2
        # fixed maximum threshold for the generated units' response curves
        self.threshmax = 10
        if random_seed:
            if type(random_seed) == int:
                print("random_seed is an integer.")
                self.MUseed = random_seed
                self.RNG = np.random.default_rng(self.MUseed)
                self.MUseed_sqn = self.RNG.bit_generator._seed_seq
            elif type(random_seed) == np.random.SeedSequence:
                print("random_seed is a SeedSequence object.")
                self.MUseed_sqn = random_seed
                self.MUseed = self.MUseed_sqn.entropy
                self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(1))
            elif type(random_seed) == np.random.Generator:
                print("random_seed is a Generator object.")
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

        print(f"New MUsim object random seed entropy is: {self.MUseed}")
        # generate random state with generate_state() method

    def __repr__(self):
        return f"MUsim object with {self.num_units} units, {len(self.spikes)} trials across {len(self.session)} sessions."

    def copy(self):
        """
        Function returns a shallow copy of the MUsim object.
        """
        return self

    def _get_spike_history_kernel(
        self,
        path_to_kernel_csv=Path(__file__).parent.joinpath("spike_history_kernel_basis.csv"),
        weights=[-12, -0.5, -0.2, -0.4],
    ):
        kernel_basis = np.genfromtxt(path_to_kernel_csv, delimiter=",")  # load kernel basis
        kernel_ms = kernel_basis @ weights  # linear combination of basis vectors
        # interpolate kernel to match sample rate
        kernel = np.interp(
            np.linspace(0, 1, int(np.round(len(kernel_ms) / 1000 * self.sample_rate * 7))),
            np.linspace(0, 1, len(kernel_ms)),
            kernel_ms,
        )
        return kernel

    def _get_spiking_probability(self, thresholded_forces):
        # this function approximates sigmoidal relationship between motor unit firing rate and output force
        # vertical scaling and offset of each units' response curve is applied to generate probability curves
        if self.MUactivation == "sigmoid":
            unit_response_curves = 1 - expit(thresholded_forces)
        elif self.MUactivation == "heaviside":
            unit_response_curves = 1 - np.heaviside(thresholded_forces, 0)
        if self.MUspike_dynamics == "independent":
            p = self.max_spike_prob
            scaled_unit_response_curves = (unit_response_curves * p) + (1 - p)
        elif self.MUspike_dynamics == "poisson":
            # this will serve as an inhomogenous thinning function for the poisson process,
            # to simulate an inhomogenous poisson process
            # https://en.wikipedia.org/wiki/Poisson_point_process#Thinning
            # scale and offset based on threshmin and threshmax values
            # this gives headspace for the MU firing rates, and ensures no spikes with no force
            scaled_unit_response_curves = self.threshmax * unit_response_curves + self.threshmin
        elif self.MUspike_dynamics == "spike_history":
            # just use regular activation function, without flipping, since
            # probabilities are now predicting the presence of spikes, not the absence
            scaled_unit_response_curves = logit(-(unit_response_curves - 1))
        return scaled_unit_response_curves

    def _get_dynamic_thresholds(self, threshmax, threshmin, new=False):
        # self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(2)[-1])
        # create arrays from original static MU thresholds, to define MU thresholds that change with each timestep
        MUthresholds_dist = self.MUthresholds_dist
        static_MU_idxs = self.static_MU_idxs
        if new == True:
            # first call assigns which units will flip in this MU population
            all_MU_idxs = np.arange(self.num_units)
            # setting MUreversal_frac to zero overrides back to size principle, regardless of idxs in MUreversal_static_units
            if len(self.MUreversal_static_units) != 0 and self.MUreversal_frac != 0:
                # assign chosen indexed MUs to remain static
                static_MU_idxs = self.MUreversal_static_units
                num_static_units = len(static_MU_idxs)
                num_dynamic_units = int(self.num_units - num_static_units)
            else:
                # without specifically assigned static MUs in self.MUreversal_static_units, static MUs are randomly assigned
                num_dynamic_units = int(self.MUreversal_frac * self.num_units)
                num_static_units = int(self.num_units - num_dynamic_units)
                static_MU_idxs = np.sort(
                    self.RNG.choice(all_MU_idxs, num_static_units, replace=False)
                )
            dynamic_MU_idxs = np.setdiff1d(all_MU_idxs, static_MU_idxs)

            self.static_MU_idxs = static_MU_idxs  # store indexes for MUs that do not flip (i.e., that obey size principle)
            self.dynamic_MU_idxs = dynamic_MU_idxs
            self.num_static_units = num_static_units
            self.num_dynamic_units = num_dynamic_units
            # self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(3)[-1])
            if (
                MUthresholds_dist == "uniform"
            ):  # equal distribution of thresholds for large/small units
                MUthresholds_gen = (threshmax - threshmin) * self.RNG.random(
                    (self.num_units)
                ) + threshmin
            elif MUthresholds_dist == "normal":  # more small units, normal dist
                MUthresholds_gen = (
                    threshmax * abs(self.RNG.standard_normal(self.num_units) / 4) + threshmin
                )
            elif MUthresholds_dist == "exponential":  # more small units, exponential dist
                MUthresholds_gen = (
                    threshmax * self.RNG.standard_exponential(self.num_units) / 10 + threshmin
                )
            else:
                raise Exception(
                    "MUthresholds_dist input must either be 'uniform', 'exponential', or 'normal'."
                )
            MUthresholds = (
                np.repeat(MUthresholds_gen, len(self.force_profile))
                .reshape(len(MUthresholds_gen), len(self.force_profile))
                .T
            )
        elif static_MU_idxs is not np.nan:
            MUthresholds_orig = self.MUthreshold_original
            MUthresholds_flip = -(
                MUthresholds_orig - threshmax - threshmin
            )  # values flip within range
            MUthresholds = np.nan * np.zeros(MUthresholds_orig.shape)  # place holder
            MUthresholds_flip[:, self.static_MU_idxs] = MUthresholds_orig[
                :, self.static_MU_idxs
            ]  # prevent chosen static MU idxs from reversing

            yank_mat = np.repeat(self.yank_profile, self.num_units).reshape(
                len(self.yank_profile), self.num_units
            )
            yank_flip_idxs = np.where(
                yank_mat >= self.yank_flip_thresh
            )  # get idxs for all yank threshold crossings
            yank_no_flip_idxs = np.where(yank_mat < self.yank_flip_thresh)
            MUthresholds[yank_flip_idxs] = MUthresholds_flip[yank_flip_idxs]  # set flips
            MUthresholds[yank_no_flip_idxs] = MUthresholds_orig[
                yank_no_flip_idxs
            ]  # set original values
        return MUthresholds

    def _lorenz(self, x, y, z, s=10, r=28, b=2.667):
        """
        Input:
        x, y, z: a point of interest in three dimensional space
        s, r, b: parameters defining the lorenz attractor
        Returns:
        x_dot, y_dot, z_dot: values of the lorenz attractor's partial
            derivatives at the point x, y, z
        """
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return x_dot, y_dot, z_dot

    def _create_trial_from_kilsort(self, kilosort_path):
        """
        Function loads data from a kilosort output and creates a trial from it. It will load the
        spike times and spike clusters from spike_times.npy and spike_clusters.npy, respectively.
        It will then create a MUsim format trial from this data.
        """
        spike_times = np.load(kilosort_path.joinpath("spike_times.npy")).ravel()
        spike_clusters = np.load(kilosort_path.joinpath("spike_clusters.npy")).ravel()
        units_in_sort = np.sort(np.unique(spike_clusters).astype(int))
        largest_cluster_number = units_in_sort[-1]
        num_units = len(units_in_sort)
        # get number of bins
        num_bins = int(
            np.round(spike_times[-1]) + 1
        )  # use last spike time because it is the largest in time
        # create empty trial
        trial = np.zeros((num_bins, largest_cluster_number + 1))
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
        npy_file_path,
        recording_bin_width,
        load_as="session",
        slice=[0, 1],
        load_type="rat-loco",
    ):
        """
        Function loads data and appends into MUsim().session, just like with simulated sessions.
        Data axes are transposed to make structure compatible transpose operation:
        (Trials x Time x MUs) --> (Time x MUs x Trials)

        Input:
            npy_file_path: if load type is "rat-loco", then this is the path to the .npy file
                containing the binned MU data.
                If load type is "kilosort", then this is the path to the kilosort output folder
                (containing spike_times.npy and spike_clusters.npy)
            recording_bin_width: provide the time width between samples of the input data
            load_as: "session" or "trial". If "session", then the entire session is loaded into MUsim().session.
                Otherwise, all trials in the session are concatenated and placed as a single trial into MUsim().spikes[-1].
                If "session", then the data is transposed to (Time x MUs x Trials).
                If "trial", then the data is reshaped into ((Time x Trials) x MUs)
            slice: slice of the data to load, as a fraction of the total data. Default is [0, 1], which loads all data.
            load_type: "MUsim", "rat-loco" or "kilosort". If "rat-loco", then the data is loaded from the rat_loco_analysis
                repository format. If "kilosort", then the data is loaded from the kilosort output format.
        Returns: nothing.
        """
        # check inputs
        assert type(npy_file_path) in [
            str,
            Path,
            PosixPath,
        ], "npy_file_path must be a string or a Path object."
        assert recording_bin_width > 0 and type(recording_bin_width) in [
            int,
            float,
        ], "recording_bin_width must be a positive number."
        assert load_as in ["session", "trial"], "load_as must be either 'session' or 'trial'."
        assert (
            type(slice) == list
            and len(slice) == 2
            and 0 <= slice[0] <= 1
            and 0 <= slice[1] <= 1
            and slice[0] < slice[1]
        ), "slice must be a list of length 2, with values between 0 and 1, and slice[0] < slice[1]."
        assert load_type in [
            "MUsim",
            "rat-loco",
            "kilosort",
        ], "load_type must be 'MUsim', 'rat-loco' or 'kilosort'."

        if load_type in ["MUsim", "rat-loco"]:
            input_MU_data = np.load(npy_file_path)
            if load_as == "session":
                if load_type == "MUsim":
                    # if MUsim data, do not transpose
                    loaded_MU_data = input_MU_data
                elif load_type == "rat-loco":
                    # (Trials x Time x MUs) --> (Time x MUs x Trials)
                    loaded_MU_data = np.transpose(input_MU_data, (1, 2, 0))
                # remove trials according to session fraction dictated by slice
                sliced_MU_data = loaded_MU_data[
                    :,
                    :,
                    int(np.round(loaded_MU_data.shape[2] * slice[0])) : int(
                        np.round(loaded_MU_data.shape[2] * slice[1])
                    ),
                ]
                self.session.append(sliced_MU_data)
                for ii in range(sliced_MU_data.shape[2]):  # add trials
                    self.spikes.append(sliced_MU_data[:, :, ii])
                self.num_trials = sliced_MU_data.shape[2]
                self.session_num_trials.append(self.num_trials)
                self.MUmode = "loaded"  # record that this session was loaded
            elif load_as == "trial":
                if load_type == "MUsim":
                    # if MUsim data, this will be 2D (Time x MUs), so no need to transpose or reshape
                    loaded_MU_data = input_MU_data
                elif load_type == "rat-loco":
                    loaded_MU_data = input_MU_data
                    # (Trials x Time x MUs) --> ((Time x Trials) x MUs)
                    loaded_MU_data = np.reshape(
                        input_MU_data,
                        (
                            input_MU_data.shape[0] * input_MU_data.shape[1],
                            input_MU_data.shape[2],
                        ),
                    )
                # time slice according to trial fraction dictated by slice
                sliced_MU_data = loaded_MU_data[
                    int(np.round(loaded_MU_data.shape[0] * slice[0])) : int(
                        np.round(loaded_MU_data.shape[0] * slice[1])
                    ),
                    :,
                ]
                # sort rows by total count (descending)
                sorted_MU_trial = sliced_MU_data[:, np.argsort(-sliced_MU_data.sum(axis=0))]
                self.spikes.append(sorted_MU_trial)
                self.num_trials = 1
        elif load_type == "kilosort":
            if load_as == "session":
                raise Exception("load_as=session not implemented for load_type=kilosort.")
            elif load_as == "trial":
                # use _create_trial_from_kilsort() to create a trial from kilosort output
                trial_from_KS = self._create_trial_from_kilsort(npy_file_path)
                sliced_trial_from_KS = trial_from_KS[
                    int(np.round(trial_from_KS.shape[0] * slice[0])) : int(
                        np.round(trial_from_KS.shape[0] * slice[1])
                    ),
                    :,
                ]
                # sorted_MU_trial = sliced_trial_from_KS[
                #     :, np.argsort(-sliced_trial_from_KS.sum(axis=0))
                # ]  # sort rows by total count (descending)
                self.spikes.append(sliced_trial_from_KS)  # spikes in order of KS cluster number
                self.num_trials = 1

        self.sample_rate = 1 / recording_bin_width
        self.num_bins_per_trial = self.spikes[-1].shape[0]
        self.num_units = self.spikes[-1].shape[1]
        # self.session_forces.append(np.repeat(np.nan, len(self.force_profile)))
        # self.session_yanks.append(np.repeat(np.nan, len(self.yank_profile)))
        return

    def rebin(self, new_bin_width, target="trial", index=-1):
        """
        Function re-bins the data in MUsim().spikes[-1] or MUsim().session[-1] to a new bin width.
        To do this, all spikes in the original bins are indexed with new spacing and matrix
        multiplication (no for loops). For each unit, all spikes in previous bins are summed
        to create each new bin value.
        Input:
            new_bin_width: new bin width to re-bin the data to, in seconds.
            target: "trial" or "session". If "trial", then the data is re-binned in MUsim().spikes[-1].
                Otherwise, the data is re-binned in MUsim().session[-1]. For the "session" target,
                the data is reshaped from (Time x MUs x Trials) to ((Time x Trials) x MUs), re-binned,
                then reshaped back to (Time x MUs x Trials).
            index: index of the trial or session to re-bin. Default is -1,
                which re-bins the last trial or session.

        Returns: nothing.
        """
        if target == "trial":
            spikes = self.spikes[index]
            new_num_bins = int(np.round(spikes.shape[0] * self.bin_width / new_bin_width))
            # get new bin edges as floats
            new_bin_edges = np.linspace(0, new_num_bins * new_bin_width, new_num_bins + 1)
            # now snap all edges to nearest integer
            new_bin_indexes = np.round(new_bin_edges * self.sample_rate).astype(int)
            new_spikes = np.zeros((new_num_bins, spikes.shape[1]), dtype=int)
            for ii in range(new_num_bins):
                new_spikes[ii, :] = np.sum(
                    spikes[new_bin_indexes[ii] : new_bin_indexes[ii + 1], :], axis=0
                )

            # update bin width
            self.bin_width = new_bin_width
            # update number of bins
            self.num_bins_per_trial = new_num_bins
            # update spikes
            self.spikes[index] = new_spikes
        if target == "session":
            raise Exception("target=session not implemented yet.")
        return

    def rebin_trials(self, new_bin_width, trial_indexes=[-1]):
        """
        Function re-bins the data in MUsim().spikes[session_indexes] to a new bin width.
        This function is a wrapper for MUsim().rebin(), to apply the re-binning to multiple trials.
        Input:
            new_bin_width: new bin width to re-bin the data to.
            session_indexes: list of session indexes to re-bin. Default is [-1], which re-bins
            the last session. Alternatively, can be a list of indexes.
        Returns: nothing.
        """
        for ii in trial_indexes:
            self.rebin(new_bin_width, target="trial", index=ii)
        return

    def rebin_sessions(self, new_bin_width, session_indexes=[-1]):
        """
        Function re-bins the data in MUsim().session[session_indexes] to a new bin width.
        This function is a wrapper for MUsim().rebin(), to apply the re-binning to multiple sessions.
        Input:
            new_bin_width: new bin width to re-bin the data to.
            session_indexes: list of session indexes to re-bin. Default is [-1], which re-bins
            the last session. Alternatively, can be a list of indexes.
        Returns: nothing.
        """
        for ii in session_indexes:
            self.rebin(new_bin_width, target="session", index=ii)
        return

    def save(self, save_path="./MUsim.npy"):
        """
        Function saves the MUsim object to a .npy file at save_path.
        """
        np.save(save_path, self, allow_pickle=False)
        return

    def save_spikes(self, save_path="./MUsim_spikes.npy", spikes_index=-1):
        """
        Function saves the spikes from self.spikes[spikes_index] to a .npy file at save_path.
        Save shape is (Time x MUs).
        """
        np.save(save_path, self.spikes[spikes_index], allow_pickle=False)
        return

    def save_session(self, save_path="./MUsim_session.npy", session_index=-1):
        """
        Function saves the session from self.session[session_index] to a .npy file at save_path.
        Save shape is (Time x MUs x Trials).
        """
        np.save(save_path, self.session[session_index], allow_pickle=False)
        return

    def sample_MUs(self, MUmode="static"):
        """
        Input:
            MUmode: decide whether unit thresholds are "static", "dynamic", or simulate "lorenz" dynamics
            (MU thresholds will be distributed according to "uniform" or "normal" setting at self.MUthresholds_dist)
        Returns: list of lists,
                units[0] holds threshold of each unit [or an array of np.empty(self.num_units) if "lorenz" mode is chosen]
                units[1] holds response curves from each MU [or holds lorenz latents if that mode is chosen]
                units[2] holds minimum ISI (poisson lambda) for each unit
        """
        # self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(4)[-1])
        # re-initialize all force/yank profiles if new trial length is set ( i.e., num_bins_per_trial )
        if self.num_bins_per_trial != len(self.init_force_profile):
            self.num_bins_per_trial = int(self.num_bins_per_trial)
            # define initial force profile
            self.init_force_profile = np.linspace(
                0, self.default_max_force, self.num_bins_per_trial
            )
            self.force_profile = self.init_force_profile  # initialize force profile
            self.init_yank_profile = np.round(
                self.sample_rate * np.diff(self.force_profile), decimals=10
            )  # define initial yank profile
            self.init_yank_profile = np.append(
                self.init_yank_profile, self.init_yank_profile[-1]
            )  # repeat last value to match force profile length
            self.yank_profile = self.init_yank_profile  # initialize yank profile

        units = self.units
        num_units = self.num_units
        force_profile = self.force_profile
        threshmax = self.threshmax
        threshmin = self.threshmin
        MUthresholds_dist = self.MUthresholds_dist

        if MUmode == "static":
            # equal distribution of thresholds for large/small units
            if MUthresholds_dist == "uniform":
                MUthresholds = (threshmax - threshmin) * self.RNG.random(
                    (self.num_units)
                ) + threshmin
            # more small units, normal dist
            elif MUthresholds_dist == "normal":
                MUthresholds = (
                    threshmax * abs(self.RNG.standard_normal(self.num_units) / 4) + threshmin
                )
            elif MUthresholds_dist == "exponential":  # more small units, exponential dist
                MUthresholds = (
                    threshmax * self.RNG.standard_exponential(self.num_units) / 10 + threshmin
                )
            else:
                raise Exception(
                    "MUthresholds_dist input must either be 'uniform', 'exponential', or 'normal'."
                )
        elif MUmode == "dynamic":
            MUthresholds = self._get_dynamic_thresholds(threshmax, threshmin, new=True)
        elif MUmode == "lorenz":
            MUthresholds = np.empty(self.num_units)  # thresholds ignored for Lorenz simulation
        else:
            raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")
        units[0] = MUthresholds

        if MUmode in ["static", "dynamic"]:
            all_forces = np.repeat(force_profile, num_units).reshape(
                (len(force_profile), num_units)
            )
            thresholded_forces = (
                all_forces - MUthresholds
            )  # subtract each respective threshold to get unique response
            units[1] = self._get_spiking_probability(thresholded_forces)
        elif MUmode == "lorenz":  # implement the lorenz system simulation
            dt = 1 / (self.sample_rate)
            num_steps = self.num_bins_per_trial - 1
            # Need one more for the initial values
            xs = np.empty(num_steps + 1)
            ys = np.empty(num_steps + 1)
            zs = np.empty(num_steps + 1)
            # Set initial values
            xs[0], ys[0], zs[0] = 20 * self.RNG.random(
                (3, 1)
            )  # random initial condition from [0,20)
            # Step through "time", calculating the partial derivatives at the current point
            # and using them to estimate the next point
            for i in range(num_steps):
                x_dot, y_dot, z_dot = self._lorenz(xs[i], ys[i], zs[i])
                xs[i + 1] = xs[i] + (x_dot * dt)
                ys[i + 1] = ys[i] + (y_dot * dt)
                zs[i + 1] = zs[i] + (z_dot * dt)
            # Center and Scale latent variables
            xs_shift, ys_shift, zs_shift = (
                xs - xs.mean(),
                ys - ys.mean(),
                zs - zs.mean(),
            )  # mean subtract
            xs_scaled, ys_scaled, zs_scaled = (
                xs_shift / xs_shift.max(),
                ys_shift / ys_shift.max(),
                zs_shift / zs_shift.max(),
            )  # scale by max
            units[1] = np.vstack(
                (xs_scaled, ys_scaled, zs_scaled)
            ).T  # lorenz latents stored in units[1]

        if MUmode == "static":
            spike_sorted_cols = self.units[0].argsort()
            self.units[0] = units[0][spike_sorted_cols]
        elif MUmode == "dynamic":
            spike_sorted_cols = self.units[0].mean(axis=0).argsort()
            self.units[0] = units[0][:, spike_sorted_cols]
            self.MUthreshold_original = self.units[0]  # save original
        elif MUmode == "lorenz":
            spike_sorted_cols = np.arange(3)
        else:
            raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")
        if MUmode in ["static", "dynamic"]:
            self.units[1] = units[1][:, spike_sorted_cols]
        # randomize minimum ISI (poisson lambda) for each unit to be from 5-25 ms
        low_lambda = 8 * self.sample_rate / 1000
        high_lambda = 32 * self.sample_rate / 1000
        # self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(5)[-1])
        self.units[2] = self.RNG.integers(low_lambda, high_lambda, size=(self.num_units))
        # sort self.units[2] in ascending order, to match with units[0] and units[1] ordering
        self.units[2] = np.sort(self.units[2])
        self.MUmode = MUmode  # record the last recruitment mode
        return units

    def simulate_spikes(self):
        """
        simple routine to generate spikes with probabilities according to the (sigmoidal) response curve.
        Input: unit response curve (probability of seeing a 0 each timestep, otherwise its 1)
        Returns: numpy array same length as response curve with  1's and 0's indicating spikes
        """
        # self.RNG = np.random.default_rng(self.MUseed_sqn.generate_state(6)[-1])
        if len(self.units[0]) == 0:
            raise Exception(
                "unit response curve empty. run '.sample_MUs()' method to define motor unit properties."
            )
        if self.MUmode == "lorenz":
            # project latent variables to high-D space
            # don't care what the high-D space is, so use random
            proj_mat = self.RNG.random((self.num_units, 3)) - 0.5  # balance it by subtracting 0.5
            latents = self.units[1]
            hiD_proj = proj_mat @ latents.T  # projection to hiD
            hiD_rates = np.exp(hiD_proj)
            # get spikes and transpose to Time x MU
            spikes = self.RNG.poisson(hiD_rates * (1 / (self.sample_rate))).T
            self.spikes.append(np.asarray(spikes, dtype=int))
        elif self.MUspike_dynamics == "independent":
            unit_response_curves = self.units[1]
            selection = self.RNG.random(unit_response_curves.shape)
            spike_idxs = np.where(selection > unit_response_curves)
            # initialize as array of zeros, spikes assigned later at spike_idxs
            self.spikes.append(np.zeros(unit_response_curves.shape, dtype=int))
            # all spikes are now assigned value of 1
            self.spikes[-1][spike_idxs] = 1
        elif self.MUspike_dynamics == "poisson":
            # use second units descriptor as relation with force signal
            unit_response_curves = self.units[1]
            # initialize as array of zeros, spikes assigned later at spike_idxs
            self.spikes.append(np.zeros(unit_response_curves.shape, dtype=int))
            # lam of RNG.poisson is the upper bound for poisson process, which will be thinned
            # probabilistically by the sigmoidal response curves for each MU
            # https://en.wikipedia.org/wiki/Poisson_point_process#Thinning
            unit_rates = self.units[2]
            time_steps = self.RNG.poisson(
                lam=unit_rates, size=self.spikes[-1].shape
            ) + int(  # unique rates for MUs
                self.refractory_period * self.sample_rate / 1000
            )  # enforce refractory period
            spike_times = np.cumsum(time_steps, axis=0)
            for ii, iUnitTimes in enumerate(spike_times.T):
                valid_time_idxs = np.where(iUnitTimes < self.num_bins_per_trial)
                # assign spikes to 1, if sigmoidal uniform distribution was also 1
                self.spikes[-1][iUnitTimes[valid_time_idxs], ii] = 1
                # section below thins spikes from poisson process, according to sigmoidal response curves
                # probability of being set to zero is 1-sigmoidal response curve
                self.spikes[-1][:, ii] = np.where(
                    (
                        (
                            unit_response_curves[:, ii]
                            > self.threshmax * self.RNG.random(unit_response_curves.shape[0])
                        )  # thin spikes according to 1-sigmoidal relationship defined in _get_spiking_probability
                    ),
                    np.zeros(self.spikes[-1][:, ii].shape),
                    self.spikes[-1][:, ii],
                )

                # self.spikes[-1] = np.where(
                #     unit_response_curves < np.tile(self.units[0], (self.units[1].shape[0], 1)),
                #     np.zeros(unit_response_curves.shape),
                #     self.spikes[-1],
                # )
            # for iUnit in range(self.num_units):
            #     for iTimestep in range(self.num_bins_per_trial):
        elif self.MUspike_dynamics == "spike_history":
            # use second units descriptor as relation with force signal
            unit_response_curves = self.units[1]
            # initialize as array of zeros, spikes assigned later at spike_idxs
            self.spikes.append(np.zeros(unit_response_curves.shape, dtype=int))
            kernel = self._get_spike_history_kernel()
            # For each MU and time t, compute whether a spike occured using a binomial distribution,
            # and if so, add the spike_history_kernel from t:t+len(kernel) timesteps of the response
            # curve for that MU. Also record the spike at that time in self.spikes
            for iUnit in range(self.num_units):
                for iTimestep in range(self.num_bins_per_trial):
                    bounded_response_curve_at_t = expit(self.units[1][iTimestep, iUnit])
                    if self.RNG.binomial(1, bounded_response_curve_at_t):
                        try:
                            self.units[1][iTimestep : iTimestep + len(kernel), iUnit] += kernel
                        except ValueError:
                            if iTimestep + len(kernel) > self.num_bins_per_trial:
                                overflow_amount = iTimestep + len(kernel) - self.num_bins_per_trial
                                self.units[1][
                                    iTimestep : iTimestep + len(kernel) - overflow_amount, iUnit
                                ] += kernel[:-overflow_amount]
                        except:
                            raise
                        self.spikes[-1][iTimestep, iUnit] = 1

        return self.spikes[-1]

    def simulate_session(self):
        if len(self.units[0]) == 0:
            raise Exception(
                "unit response curve empty. run '.sample_MUs()' method to define motor unit properties."
            )
        trials = self.num_trials
        session_data_shape = (len(self.force_profile), self.num_units, trials)
        self.session.append(np.zeros(session_data_shape))
        for iTrial in range(trials):
            self.session[-1][:, :, iTrial] = self.simulate_spikes()
        self.session_forces.append(self.force_profile)
        self.session_yanks.append(self.yank_profile)
        self.session_num_trials.append(self.num_trials)
        self.session_response_curves.append(self.units[1])
        self.session_MUactivations.append(self.MUactivation)
        self.session_MUseed.append(self.MUseed)
        return self.session[-1]

    def apply_new_force(self, input_force_profile):
        """
        feed in a new 1D force profile to apply to all recruited motor units
        """
        assert len(input_force_profile.shape) == 1, "new force profile must be one-dimensional."
        self.num_bins_per_trial = len(input_force_profile)  # set new trial length

        if len(self.units[0]) == 0:
            raise Exception(
                "unit response curve empty. run '.sample_MUs()' method to define motor unit properties."
            )
        all_forces = np.repeat(input_force_profile, self.num_units).reshape(
            (len(input_force_profile), self.num_units)
        )
        self.yank_profile = np.round(
            self.sample_rate * np.diff(input_force_profile), decimals=10
        )  # update yank_profile
        self.yank_profile = np.append(
            self.yank_profile, self.yank_profile[-1]
        )  # repeat yank_profile[-1] value [to match len(force_profile)]
        if self.MUmode == "static":
            thresholded_forces = all_forces - self.units[0]
        elif self.MUmode == "dynamic":  # everything must be updated for dynamic
            MUthresholds = self._get_dynamic_thresholds(self.threshmax, self.threshmin)
            thresholded_forces = all_forces - MUthresholds
            self.units[0] = MUthresholds
        else:
            raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")

        self.units[1] = self._get_spiking_probability(thresholded_forces)
        self.force_profile = input_force_profile  # set new force profile value

    def reset_force(self):
        self.apply_new_force(self.init_force_profile)

    def convolve(self, sigma=40, target="spikes"):  # default smoothing value is 40 bins
        """
        Smooth spiking data from a single trial or entire session with Gaussian kernel with selected bandwidth.

        Inputs:
            - sigma: positive float sets bandwidth for Gaussian convolution.
            - target: string input specifying which data should be smoothed
                - 'spikes' : will smooth most recently generated spikes from '.simulate_spikes()' method, stored as .smooth_spikes
                - 'session': will smooth most recently generated session from '.simulate_session()' method, stored as .smooth_session

        Returns:
            - numpy array: smoothed session or smoothed spikes, dependent on target choice.
        """
        if target == "spikes":
            num_units_in_last_trial = self.spikes[-1].shape[1]
            self.smooth_spikes.append(np.zeros(self.spikes[-1].shape))  # create new list entry
            for iUnit in range(num_units_in_last_trial):
                self.smooth_spikes[-1][:, iUnit] = gaussian_filter1d(
                    self.spikes[-1][:, iUnit], sigma, mode="reflect"
                )
            return self.smooth_spikes[-1]
        elif target == "session":
            num_trials_in_last_session = self.session[-1].shape[2]
            num_units_in_last_session = self.session[-1].shape[1]
            session_data_shape = (
                self.session[-1].shape[0],
                num_units_in_last_session,
                num_trials_in_last_session,
            )
            self.smooth_session.append(np.zeros(session_data_shape))  # create new list entry
            for iUnit in range(num_units_in_last_session):
                for iTrial in range(num_trials_in_last_session):
                    self.smooth_session[-1][:, iUnit, iTrial] = gaussian_filter1d(
                        self.session[-1][:, iUnit, iTrial], sigma, mode="reflect"
                    )
            return self.smooth_session[-1]

    def see(self, target="spikes", trial=-1, session=-1, unit=0, legend=True, no_offset=False):
        """
        Main visualization method for MUsim.

        Inputs:
            target: string inputs for visualization of simulation data or properties
                - 'curves': view current MU response curves to the last force_profile
                - 'spikes': view spike outputs from the last force response simulation
                - 'smooth': view convolved outputs from the last force response simulation
                - 'unit': view selected unit activations across all simulated trials
                - 'force': view default and current force and yank profiles
            trial: select trial to view if target is 'spikes' or 'smooth' (default: last trial)
            unit: select unit to view if target is 'unit'

        Returns:
            Plots.
        """
        if target == "thresholds":
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                "color", plt.cm.jet(np.linspace(0, 1, self.num_units))
            )
            if self.MUmode == "static":
                thresholds = self.units[0]
            elif self.MUmode == "dynamic":
                thresholds = self.units[0].mean(axis=0)
            else:
                raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")
            plt.hist(thresholds, self.num_units)
            plt.title("thresholds across " + str(self.num_units) + " generated units")
            plt.ylabel("count")
            plt.xlabel("threshold values (shift applied to response curve)")
            plt.show(block=False)
        elif target == "force":
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                "color", plt.cm.seismic(np.linspace(0, 1, 4))
            )
            # plt.plot(self.init_force_profile)
            # plt.plot(self.init_yank_profile)
            try:
                plt.plot(self.session_forces[session])
            except IndexError:
                plt.plot(self.force_profile)
            except:
                raise
            # plt.plot(self.session_yanks[session])
            # plt.plot(
            #     np.repeat(self.yank_flip_thresh,len(self.yank_profile)),
            #     color='black',
            #     linestyle='dashed'
            #     )
            plt.legend(
                [
                    # "default force",
                    # "default yank",
                    "current force",
                    # "current yank",
                    # "yank flip threshold"
                ]
            )
            plt.title("force profile for simulation")
            plt.ylabel("simulated force (a.u.)")
            plt.xlabel("time (indexes)")
            plt.show(block=False)
        elif target == "curves":
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                "color", plt.cm.jet(np.linspace(0, 1, self.num_units))
            )
            if self.num_units > 10 and legend == True:
                legend = False
                print("could not plot legend with more than 10 MUs.")
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            # if self.MUspike_dynamics == "spike_history":
            #     self.units[1] = expit(self.units[1])
            if legend:  # flip legend to match data; handle dynamic vs. static (arrays vs. scalars)
                if self.MUmode == "static":
                    thresholds = self.units[0]
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:, ii], label=thresholds[ii])
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[::-1], labels[::-1], title="thresholds", loc="lower left")
                elif self.MUmode == "dynamic":
                    # take mean to reduce to 1 number
                    thresholds = self.units[0].mean(axis=0).round()
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:, ii], label=thresholds[ii])
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(
                        handles[::-1], labels[::-1], title="mean thresholds", loc="lower left"
                    )
            else:
                if self.MUmode == "static":
                    thresholds = self.units[0]
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:, ii])
                elif self.MUmode == "dynamic":
                    # take mean to reduce to 1 number
                    thresholds = self.units[0].mean(axis=0).round()
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:, ii])

            plt.title("MU response curves")
            plt.xlabel("time (indexes)")
            plt.ylabel("activation level")
            plt.show(block=False)
        elif target == "spikes":
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                "color", plt.cm.jet(np.linspace(0, 1, self.num_units))
            )
            # colorList = plt.cm.jet(np.linspace(0,1,self.num_units))
            # check whether legend should be placed outside if too many MU's
            legend = False
            if self.num_units > 10 and legend == True:
                legend = False
                print("could not plot legend with more than 10 MUs.")
                counts = np.sum(self.spikes[trial], axis=0)
            else:
                peaks = np.zeros(self.spikes[trial].shape)
                leng = len(self.force_profile)
                for iUnit in range(self.num_units):
                    peak_idxs = find_peaks(
                        self.spikes[trial].ravel()[iUnit * leng : iUnit * leng + leng],
                        height=(0.8, 1.2),
                    )
                    peaks[peak_idxs[0], iUnit] = 1
                counts = np.sum(peaks, axis=0)[::-1]

            # prepare for spike plot
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            find_spikes_idxs = np.array(
                np.nonzero(self.spikes[trial][:, :].T)
            ).T  # find spikes with nonzero, transpose for eventplot format
            list_of_spike_idxs = np.split(
                find_spikes_idxs[:, 1], np.unique(find_spikes_idxs[:, 0], return_index=True)[1][1:]
            )  # group idxs by MU identity

            # plot spike events as raster
            handles = ax.eventplot(
                list_of_spike_idxs,
                linelengths=0.8,
                linewidths=1,
                colors=["C{}".format(i) for i in range(len(list_of_spike_idxs))],
            )
            if legend:  # determine whether to plot the legend
                ax.legend(handles[::-1], list(counts)[::-1], title="approx. cnt", loc="upper left")
            # ax.set_facecolor((200/255,200/255,200/255)) # set RGB color (gray)
            # fig.set_facecolor((40/255,40/255,40/255)) # set RGB color (235,210,180) is manila
            plt.title("spikes present across MU population during trial")
            plt.xlim((0, len(self.spikes[trial])))
            plt.ylim((-1, self.num_units))
            plt.xlabel("time (indexes)")
            plt.ylabel("motor unit spikes sorted by threshold")
            plt.show(block=False)
        elif target == "smooth":  # shows smoothed unit traces, 1 trial at a time
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                "color", plt.cm.jet(np.linspace(0, 1, self.num_units))
            )
            for ii in range(self.num_units):
                if no_offset is False:
                    if len(self.smooth_session) != 0:
                        max_smooth_val = self.smooth_session[session].max() / 2
                        plt.plot(self.smooth_session[session][:, ii, trial] / max_smooth_val + ii)
                    elif len(self.smooth_spikes) != 0:
                        max_smooth_val = self.smooth_spikes[trial].max() / 2
                        plt.plot(self.smooth_spikes[trial][:, ii] / max_smooth_val + ii)
                    else:
                        raise Exception(
                            "there is no smoothed spiking data. run '.convolve()' method to smooth spikes."
                        )
                else:
                    if len(self.smooth_session) != 0:
                        plt.plot(self.smooth_session[session][:, ii, trial])
                    elif len(self.smooth_spikes) != 0:
                        plt.plot(self.smooth_spikes[trial][:, ii])
                    else:
                        raise Exception(
                            "there is no smoothed spiking data. run '.convolve()' method to smooth spikes."
                        )
            plt.title("smoothed spikes present across MU population during trial")
            plt.xlabel("time (indexes)")
            plt.ylabel("activation level (smoothed spikes)")
            plt.show(block=False)
        elif target == "unit":  # looks at 1 unit, shows all trials in a session
            if len(self.smooth_session) == 0:
                raise Exception(
                    "there is no smoothed session data. run '.convolve(target='session')' method to smooth a session."
                )
            else:
                plt.plot(self.smooth_session[session][:, unit, :], color="skyblue", alpha=0.5)
                plt.plot(
                    np.mean(self.smooth_session[session][:, unit, :], axis=1), color="darkblue"
                )
                plt.title(
                    "smoothed rates for unit #"
                    + str(unit)
                    + " across "
                    + str(self.num_trials)
                    + " trials"
                )
                plt.xlabel("time (indexes)")
                plt.ylabel("activation level (smoothed spikes)")
                plt.show(block=False)
        elif target == "rates":  # shows smoothed unit traces, 1 trial at a time
            if len(self.smooth_session) != 0:
                max_smooth_val = self.smooth_session[session].max() / 2
                plt.imshow(
                    self.smooth_session[session][:, :, trial].T / max_smooth_val,
                    aspect="auto",
                    interpolation="none",
                )
            elif len(self.smooth_spikes) != 0:
                max_smooth_val = self.smooth_spikes[trial].max() / 2
                plt.imshow(
                    self.smooth_spikes[trial].T / max_smooth_val,
                    aspect="auto",
                    interpolation="none",
                )
            else:
                raise Exception(
                    "there is no smoothed spiking data. use '.convolve()' method to smooth spikes."
                )
            clb = plt.colorbar()
            clb.set_label("rates", labelpad=-30, y=1.05, rotation=0)
            plt.title("smoothed rates for all units")
            plt.xlabel("time (indexes)")
            plt.ylabel("motor units")
            plt.show(block=False)
        elif target == "ave_rates":  # shows all units' trial-averaged rates in a session
            if len(self.smooth_session) == 0:
                raise Exception(
                    "there is no smoothed session data. run '.convolve(target='session')' method to smooth a session."
                )
            else:
                trial_ave_responses = np.mean(self.smooth_session[session], axis=2)
                max_smooth_val = trial_ave_responses.max() / 2
                plt.imshow(
                    trial_ave_responses.T / max_smooth_val, aspect="auto", interpolation="none"
                )
                clb = plt.colorbar()
                clb.set_label("rates", labelpad=-30, y=1.05, rotation=0)
                plt.title("trial-averaged smoothed rates for all units")
                plt.xlabel("time (indexes)")
                plt.ylabel("motor units")
                plt.show(block=False)
        elif target == "lorenz":
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=self.units[1][:, 0],
                        y=self.units[1][:, 1],
                        z=self.units[1][:, 2],
                        mode="lines",
                    )
                ]
            )
            fig.show(block=False)
        else:
            raise Exception("invalid keyword argument for '.see()' visualization method.")
