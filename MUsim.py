import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class MUsim():

    def __init__(self, random_seed=False):
        self.MUmode="static"    # "static" for size-principle obediance, "dynamic" for yank-dependent thresholds
        self.MUseed=random_seed # int, set seed for repeatability, if desired
        self.MUactivation="sigmoid" # "sigmoid"/"heaviside" activation function of MUs
        self.MUthresholds_dist="exponential" # can be either "exponential", "uniform" or "normal". Distributes proportion of small vs large MUs
        self.MUspike_dynamics="poisson" # can be either "poisson" or "independent". Models spiking behavior as poisson process or as independent
        self.MUreversal_frac = 1 # fractions of MU thresholds that will flip during a dynamic mode simulation
        self.MUreversal_static_units = [] # will force the units at these indexes to remain static during the dynamic simulations
        self.static_MU_idxs = np.nan # variable used to track which MUs reverse threshold (nan until assigned in dynamic mode)
        self.units = [[],[]]    # will hold all MU thresholds at .units[0], all response curves at .units[1], and .units[2]
        self.spikes = []        # spiking responses appended here each time .simulate_spikes() is called
        self.noise_level = []   # non-negative Gaussian noise level term added to spikes
        self.session = []       # matrix of spiking responses appended to this list here each time .simulate_session() is called
        self.session_forces = [] # current force profile appended to this list here each time .simulate_session() is called
        self.session_yanks = [] # current yank profile appended to this list here each time .simulate_session() is called
        self.session_num_trials = [] # num trials for each session
        self.session_response_curves = [] # MU response curves for each session
        self.session_MUactivations = [] # MU activation functions for each session
        self.session_MUseed = [] # save random seed for each session
        self.session_noise_level = 0 # noise level for session
        self.smooth_spikes = [] # smoothed spiking responses appended here each time .convolve() is called
        self.smooth_session = []# matrix of smoothed spiking responses appended here each time .convolve(target=session) is called
        self.num_units = 10     # default number of units to simulate
        self.num_trials = 50    # default number of trials to simulate
        self.num_bins_per_trial = 1000 # amount of time per trial is (num_bins_per_trial/sample_rate)
        self.sample_rate = 1000 # Hz, default sampling rate. it's inverse results in the time alloted to each bin
        self.default_max_force = 5
        self.init_force_profile = np.linspace(0,self.default_max_force,self.num_bins_per_trial)  # define initial force profile
        self.force_profile = self.init_force_profile  # initialize force profile
        self.init_yank_profile = np.round(self.sample_rate*np.diff(self.force_profile),decimals=10)  # define initial yank profile
        self.init_yank_profile = np.append(self.init_yank_profile,self.init_yank_profile[-1])  # repeat last value to match force profile length
        self.yank_profile = self.init_yank_profile  # initialize yank profile
        self.yank_flip_thresh = 20  # default yank value at which the thresholds flips 
        self.max_spike_prob = 0.08  # set to achieve ~80hz (simulate realistic MU firing rates)
        self.threshmin = 2      # fixed minimum threshold for the generated units' response curves
        self.threshmax = 10      # fixed maximum threshold for the generated units' response curves
        if random_seed:
            assert type(random_seed) is int,"`random_seed` variable must be `int`"
            np.random.seed(random_seed)

    def _get_spiking_probability(self,thresholded_forces):
        # this function approximates sigmoidal relationship between motor unit firing rate and output force
        # vertical scaling and offset of each units' response curve is applied to generate probability curves
        p = self.max_spike_prob
        if self.MUactivation=="sigmoid":
            unit_response_curves = 1- expit(thresholded_forces) 
        elif self.MUactivation=="heaviside":
            unit_response_curves = 1- np.heaviside(thresholded_forces,0) 
        if self.MUspike_dynamics=="independent": 
            scaled_unit_response_curves = (unit_response_curves*p)+(1-p)
        elif self.MUspike_dynamics=="poisson":
            scaled_unit_response_curves = self.threshmax//2*(1-unit_response_curves) # flip to regular sigmoid, scale by range to allow MUs headspace
        return scaled_unit_response_curves

    def _get_dynamic_thresholds(self,threshmax,threshmin,new=False):
        # create arrays from original static MU thresholds, to define MU thresholds that change with each timestep 
        MUthresholds_dist = self.MUthresholds_dist
        static_MU_idxs = self.static_MU_idxs
        if new == True:
            # first call assigns which units will flip in this MU population
            all_MU_idxs = np.arange(self.num_units)
            # setting MUreversal_frac to zero overrides back to size principle, regardless of idxs in MUreversal_static_units
            if len(self.MUreversal_static_units)!=0 and self.MUreversal_frac!=0: 
                # assign chosen indexed MUs to remain static
                static_MU_idxs = self.MUreversal_static_units
                num_static_units = len(static_MU_idxs)
                num_dynamic_units = int(self.num_units-num_static_units)
            else:
                # without specifically assigned static MUs in self.MUreversal_static_units, static MUs are randomly assigned
                num_dynamic_units = int(self.MUreversal_frac*self.num_units)
                num_static_units = int(self.num_units-num_dynamic_units)
                static_MU_idxs = np.sort(np.random.choice(all_MU_idxs,num_static_units,replace=False))
            dynamic_MU_idxs = np.setdiff1d(all_MU_idxs,static_MU_idxs)
            
            self.static_MU_idxs = static_MU_idxs # store indexes for MUs that do not flip (i.e., that obey size principle)
            self.dynamic_MU_idxs = dynamic_MU_idxs
            self.num_static_units = num_static_units
            self.num_dynamic_units = num_dynamic_units

            if MUthresholds_dist == "uniform": # equal distribution of thresholds for large/small units
                MUthresholds_gen = (threshmax-threshmin)*np.random.random_sample((self.num_units))+threshmin
            elif MUthresholds_dist == "normal": # more small units, normal dist
                MUthresholds_gen = threshmax*abs(np.random.standard_normal(self.num_units)/4)+threshmin
            elif MUthresholds_dist == "exponential": # more small units, exponential dist
                MUthresholds_gen = threshmax*np.random.standard_exponential(self.num_units)/10+threshmin
            else:
                raise Exception("MUthresholds_dist input must either be 'uniform', 'exponential', or 'normal'.")
            MUthresholds = np.repeat(MUthresholds_gen,len(self.force_profile)).reshape(len(MUthresholds_gen),len(self.force_profile)).T
        elif static_MU_idxs is not np.nan:
            MUthresholds_orig = self.MUthreshold_original
            MUthresholds_flip = -(MUthresholds_orig-threshmax-threshmin) # values flip within range
            MUthresholds = np.nan*np.zeros(MUthresholds_orig.shape) # place holder
            MUthresholds_flip[:,self.static_MU_idxs] = MUthresholds_orig[:,self.static_MU_idxs] # prevent chosen static MU idxs from reversing

            yank_mat = np.repeat(self.yank_profile,self.num_units).reshape(len(self.yank_profile),self.num_units)
            yank_flip_idxs = np.where(yank_mat>=self.yank_flip_thresh) # get idxs for all yank threshold crossings
            yank_no_flip_idxs = np.where(yank_mat<self.yank_flip_thresh)
            MUthresholds[yank_flip_idxs] = MUthresholds_flip[yank_flip_idxs] # set flips
            MUthresholds[yank_no_flip_idxs] = MUthresholds_orig[yank_no_flip_idxs] # set original values
        return MUthresholds
    
    def _lorenz(self, x, y, z, s=10, r=28, b=2.667):
        '''
        Given:
        x, y, z: a point of interest in three dimensional space
        s, r, b: parameters defining the lorenz attractor
        Returns:
        x_dot, y_dot, z_dot: values of the lorenz attractor's partial
            derivatives at the point x, y, z
        '''
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot

    def load_MUs(self, npy_file_path, bin_width):
        """
        Function loads data and appends into MUsim().session, just like with simulated sessions.
        
        Input:
            npy_file_path: path to load real data from rat_loco_analysis, transposes the data axes to make structure compatible
                transpose operation: (Trials x Time x MUs) --> (Time x MUs x Trials)
            bin_width: provide the time width of the bins of the input data
        
        Returns: nothing.
        """
        binned_MU_session = np.load(npy_file_path)
        # (Trials x Time x MUs) --> (Time x MUs x Trials)
        transposed_binned_MU_session = np.transpose(binned_MU_session, (1,2,0))
        self.session.append(transposed_binned_MU_session)
        for ii in range(transposed_binned_MU_session.shape[2]): # add trials
            self.spikes.append(transposed_binned_MU_session[:,:,ii])
        self.num_bins_per_trial = transposed_binned_MU_session.shape[0]
        self.num_units = transposed_binned_MU_session.shape[1]
        self.num_trials = transposed_binned_MU_session.shape[2]
        self.session_num_trials.append(self.num_trials)
        self.MUmode = "loaded" # record that this session was loaded
        self.session_forces.append(np.repeat(np.nan,len(self.force_profile)))
        self.session_yanks.append(np.repeat(np.nan,len(self.yank_profile)))
        self.noise_level = np.zeros(transposed_binned_MU_session.shape[0])
        return
    
    def sample_MUs(self, MUmode="static"):
        """ 
        Input:
            MUmode: decide whether unit thresholds are "static", "dynamic", or simulate "lorenz" dynamics
            (MU thresholds will be distributed according to "uniform" or "normal" setting at self.MUthresholds_dist) 
        Returns: list of lists,
                units[0] holds threshold of each unit [or an array of np.empty(self.num_units) if "lorenz" mode is chosen]
                units[1] holds response curves from each MU [or holds lorenz latents if that mode is chosen]
        """ 

        # re-initialize all force/yank profiles if new trial length is set ( i.e., num_bins_per_trial )
        if self.num_bins_per_trial != len(self.init_force_profile):
            self.init_force_profile = np.linspace(0,self.default_max_force,self.num_bins_per_trial)  # define initial force profile
            self.force_profile = self.init_force_profile  # initialize force profile
            self.init_yank_profile = np.round(self.sample_rate*np.diff(self.force_profile),decimals=10)  # define initial yank profile
            self.init_yank_profile = np.append(self.init_yank_profile,self.init_yank_profile[-1])  # repeat last value to match force profile length
            self.yank_profile = self.init_yank_profile  # initialize yank profile

        units = self.units
        num_units = self.num_units
        force_profile = self.force_profile
        threshmax = self.threshmax
        threshmin = self.threshmin
        MUthresholds_dist = self.MUthresholds_dist

        if MUmode == "static":
            if MUthresholds_dist == "uniform": # equal distribution of thresholds for large/small units
                MUthresholds = (threshmax-threshmin)*np.random.random_sample((self.num_units))+threshmin
            elif MUthresholds_dist == "normal": # more small units, normal dist
                MUthresholds = threshmax*abs(np.random.standard_normal(self.num_units)/4)+threshmin
            elif MUthresholds_dist == "exponential": # more small units, exponential dist
                MUthresholds = threshmax*np.random.standard_exponential(self.num_units)/10+threshmin
            else:
                raise Exception("MUthresholds_dist input must either be 'uniform', 'exponential', or 'normal'.")
        elif MUmode == "dynamic":
            MUthresholds = self._get_dynamic_thresholds(threshmax,threshmin,new=True)
        elif MUmode == "lorenz":
            MUthresholds = np.empty(self.num_units) # thresholds ignored for Lorenz simulation
        else:
            raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")
        units[0] = MUthresholds
        
        if MUmode in ["static","dynamic"]:
            all_forces = np.repeat(force_profile,num_units).reshape((len(force_profile),num_units))
            thresholded_forces = all_forces - MUthresholds # subtract each respective threshold to get unique response
            units[1] = self._get_spiking_probability(thresholded_forces)
        elif MUmode == "lorenz": # implement the lorenz system simulation
            dt = 1/(self.sample_rate)
            num_steps = self.num_bins_per_trial-1
            # Need one more for the initial values
            xs = np.empty(num_steps + 1)
            ys = np.empty(num_steps + 1)
            zs = np.empty(num_steps + 1)
            # Set initial values
            xs[0], ys[0], zs[0] = 20*np.random.rand(3,1) # random initial condition from [0,20)
            # Step through "time", calculating the partial derivatives at the current point
            # and using them to estimate the next point
            for i in range(num_steps):
                x_dot, y_dot, z_dot = self._lorenz(xs[i], ys[i], zs[i])
                xs[i + 1] = xs[i] + (x_dot * dt)
                ys[i + 1] = ys[i] + (y_dot * dt)
                zs[i + 1] = zs[i] + (z_dot * dt)
            # Center and Scale latent variables
            xs_shift, ys_shift, zs_shift = xs-xs.mean(), ys-ys.mean(), zs-zs.mean() # mean subtract
            xs_scaled, ys_scaled, zs_scaled = xs_shift/xs_shift.max(), ys_shift/ys_shift.max(), zs_shift/zs_shift.max() # scale by max
            units[1] = np.vstack((xs_scaled,ys_scaled,zs_scaled)).T # lorenz latents stored in units[1]
        
        if MUmode == "static":
            spike_sorted_cols = self.units[0].argsort()
            self.units[0] = units[0][spike_sorted_cols]
        elif MUmode == "dynamic":
            spike_sorted_cols = self.units[0].mean(axis=0).argsort()
            self.units[0] = units[0][:,spike_sorted_cols]
            self.MUthreshold_original = self.units[0] # save original
        elif MUmode == "lorenz":
            spike_sorted_cols = np.arange(3)
        else:
            raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")
        if MUmode in ["static","dynamic"]:
            self.units[1] = units[1][:,spike_sorted_cols]
        self.MUmode = MUmode # record the last recruitment mode
        return units

    def simulate_spikes(self,noise_level=0):
        """
            simple routine to generate spikes with probabilities according to the (sigmoidal) response curve.
            Input: unit response curve (probability of seeing a 0 each timestep, otherwise its 1)
            Returns: numpy array same length as response curve with  1's and 0's indicating spikes
        """
        if len(self.units[0])==0:
            raise Exception("unit response curve empty. run '.sample_MUs()' method to define motor unit properties.")
        if self.MUmode == "lorenz":
            # project latent variables to high-D space
            # don't care what the high-D space is, so use random
            proj_mat = np.random.rand(self.num_units, 3)-0.5 # balance it by subtracting 0.5
            latents = self.units[1]
            hiD_proj = proj_mat @ latents.T # projection to hiD
            hiD_rates = np.exp(hiD_proj)
            spikes = np.random.poisson(hiD_rates*(1/(self.sample_rate))).T # get spikes and transpose to Time x MU
            self.spikes.append(np.asarray(spikes,dtype=np.float))
            self.noise_level.append(noise_level)
        elif self.MUspike_dynamics=="independent":
            unit_response_curves = self.units[1]
            selection = np.random.random(unit_response_curves.shape)
            spike_idxs = np.where(selection>unit_response_curves)
            self.spikes.append(np.zeros(unit_response_curves.shape)) # initialize as array of zeros, spikes assigned later at spike_idxs 
            if noise_level != 0: # add noise before spikes
                assert noise_level>0 and noise_level<=1, "noise required to be between 0 and 1."
                self.spikes[-1] = self.spikes[-1]+noise_level*np.random.standard_normal(self.spikes[-1].shape).__abs__()
            self.noise_level.append(noise_level)
            self.spikes[-1][spike_idxs] = 1 # all spikes are now assigned value of 1, regardless of noise
        elif self.MUspike_dynamics=="poisson": 
            unit_response_curves = self.units[1] # use second units descriptor as relation with force signal
            # selection = np.random.random(unit_response_curves.shape)
            # force_gate = np.where(
            #     selection>unit_response_curves,
            #     selection,
            #     np.zeros_like(unit_response_curves)) # gate units with this sigmoidal uniform distribution, to reflect force influence
            self.spikes.append(np.zeros(unit_response_curves.shape)) # initialize as array of zeros, spikes assigned later at spike_idxs 
            time_steps = np.random.poisson(lam=5,size=self.spikes[-1].shape)
            spike_times = np.cumsum(time_steps, axis=0)
            for ii, iUnitTimes in enumerate(spike_times.T):
                valid_time_idxs = np.where(iUnitTimes<self.num_bins_per_trial)
                # force_gated_spikes = np.intersect1d(iUnitTimes[valid_time_idxs],force_gate[:,ii].nonzero())
                self.spikes[-1][iUnitTimes[valid_time_idxs],ii] = 1 # assign spikes to 1, if sigmoidal uniform distribution was also 1
                self.spikes[-1] = np.where(
                    unit_response_curves<np.tile(self.units[0],(self.units[1].shape[0],1)),
                    np.zeros(unit_response_curves.shape),
                    self.spikes[-1])
            self.noise_level.append(noise_level)
            # for iUnit in range(self.num_units):
            #     for iTimestep in range(self.num_bins_per_trial):
        return self.spikes[-1]

    def simulate_session(self):
        if len(self.units[0])==0:
            raise Exception("unit response curve empty. run '.sample_MUs()' method to define motor unit properties.")
        trials = self.num_trials
        session_data_shape = (len(self.force_profile),self.num_units,trials)
        self.session.append(np.zeros(session_data_shape))
        for iTrial in range(trials):
            self.session[-1][:,:,iTrial] = self.simulate_spikes(self.session_noise_level)
        self.session_forces.append(self.force_profile)
        self.session_yanks.append(self.yank_profile)
        self.session_num_trials.append(self.num_trials)
        self.session_response_curves.append(self.units[1])
        self.session_MUactivations.append(self.MUactivation)
        self.session_MUseed.append(self.MUseed)
        return self.session[-1]

    def apply_new_force(self,input_force_profile):
        """
        feed in a new 1D force profile to apply to all recruited motor units
        """
        assert len(input_force_profile.shape) == 1, "new force profile must be one-dimensional."
        self.num_bins_per_trial = len(input_force_profile) # set new trial length
        
        if len(self.units[0])==0:
            raise Exception("unit response curve empty. run '.sample_MUs()' method to define motor unit properties.")
        all_forces = np.repeat(input_force_profile,self.num_units).reshape((len(input_force_profile),self.num_units))
        self.yank_profile = np.round(self.sample_rate*np.diff(input_force_profile),decimals=10) # update yank_profile
        self.yank_profile = np.append(self.yank_profile,self.yank_profile[-1]) # repeat yank_profile[-1] value [to match len(force_profile)]
        if self.MUmode == "static":
            thresholded_forces = all_forces - self.units[0]
        elif self.MUmode == "dynamic": # everything must be updated for dynamic
            MUthresholds = self._get_dynamic_thresholds(self.threshmax,self.threshmin)
            thresholded_forces = all_forces - MUthresholds
            self.units[0] = MUthresholds
        else:
            raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")

        self.units[1] = self._get_spiking_probability(thresholded_forces)
        self.force_profile = input_force_profile # set new force profile value

    def reset_force(self):
        self.apply_new_force(self.init_force_profile)

    def convolve(self,sigma=40,target='spikes'): # default smoothing value is 40 bins
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
        if target == 'spikes':
            num_units_in_last_trial = self.spikes[-1].shape[1]
            self.smooth_spikes.append(np.zeros(self.spikes[-1].shape)) # create new list entry
            for iUnit in range(num_units_in_last_trial):
               self.smooth_spikes[-1][:,iUnit] = gaussian_filter1d(self.spikes[-1][:,iUnit],sigma,mode="reflect")
            return self.smooth_spikes[-1]
        elif target == 'session':
            num_trials_in_last_session = self.session[-1].shape[2]
            num_units_in_last_session = self.session[-1].shape[1]
            session_data_shape = (self.session[-1].shape[0],num_units_in_last_session,num_trials_in_last_session)
            self.smooth_session.append(np.zeros(session_data_shape)) # create new list entry
            for iUnit in range(num_units_in_last_session):
                for iTrial in range(num_trials_in_last_session):
                    self.smooth_session[-1][:,iUnit,iTrial] = gaussian_filter1d(self.session[-1][:,iUnit,iTrial],sigma,mode="reflect")
            return self.smooth_session[-1]

    def see(self,target='spikes',trial=-1,session=-1,unit=0,legend=True, no_offset=False):
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
        if target == 'thresholds':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            if self.MUmode == "static":
                thresholds = self.units[0]
            elif self.MUmode == "dynamic":
                thresholds = self.units[0].mean(axis=0)
            else:
                raise Exception("MUmode must be either 'static', 'dynamic', or 'lorenz'.")
            plt.hist(thresholds,self.num_units)
            plt.title('thresholds across '+str(self.num_units)+' generated units')
            plt.ylabel("count")
            plt.xlabel("threshold values (shift applied to response curve)")
            plt.show()
        elif target == 'force':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.seismic(np.linspace(0,1,4)))
            # plt.plot(self.init_force_profile)
            # plt.plot(self.init_yank_profile)
            plt.plot(self.session_forces[session])
            # plt.plot(self.session_yanks[session])
            # plt.plot(
            #     np.repeat(self.yank_flip_thresh,len(self.yank_profile)),
            #     color='black',
            #     linestyle='dashed'
            #     )
            plt.legend([
                # "default force",
                # "default yank",
                "current force",
                # "current yank",
                # "yank flip threshold"
                ])
            plt.title("force profile for simulation")
            plt.ylabel("simulated force (a.u.)")
            plt.xlabel("time (ms)")
            plt.show()
        elif target == 'curves':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            if self.num_units > 10 and legend == True: 
                legend=False
                print("could not plot legend with more than 10 MUs.")
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if legend: # flip legend to match data; handle dynamic vs. static (arrays vs. scalars)
                if self.MUmode == "static":
                    thresholds = self.units[0]
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:,ii],label=thresholds[ii])
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[::-1], labels[::-1], title="thresholds",loc="lower left")
                elif self.MUmode == "dynamic":
                    # take mean to reduce to 1 number
                    thresholds = self.units[0].mean(axis=0).round()
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:,ii],label=thresholds[ii])
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[::-1], labels[::-1], title="mean thresholds",loc="lower left")
            else:
                if self.MUmode == "static":
                    thresholds = self.units[0]
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:,ii])
                elif self.MUmode == "dynamic":
                    # take mean to reduce to 1 number
                    thresholds = self.units[0].mean(axis=0).round()
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:,ii])

            plt.title("MU response curves")
            plt.xlabel("time (ms)")
            plt.ylabel("probability of zero spikes in each bin")
            plt.show()
        elif target == 'spikes':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            # colorList = plt.cm.jet(np.linspace(0,1,self.num_units))
            # check whether legend should be placed outside if too many MU's
            legend=False
            if self.num_units > 10 and legend == True: 
                legend=False
                print("could not plot legend with more than 10 MUs.")
            if self.noise_level[-1]==0:
                counts = np.sum(self.spikes[trial],axis=0)
            else:
                peaks = np.zeros(self.spikes[trial].shape)
                leng = len(self.force_profile)
                for iUnit in range(self.num_units):
                    peak_idxs = find_peaks(self.spikes[trial].ravel()[iUnit*leng:iUnit*leng+leng],height=(0.8,1.2))
                    peaks[peak_idxs[0],iUnit]=1
                counts = np.sum(peaks,axis=0)[::-1]
            
            # prepare for spike plot
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            find_spikes_idxs = np.array(np.nonzero(self.spikes[trial][:,:].T)).T # find spikes with nonzero, transpose for eventplot format
            list_of_spike_idxs = np.split(find_spikes_idxs[:,1], np.unique(find_spikes_idxs[:, 0], return_index=True)[1][1:]) # group idxs by MU identity
            
            # plot spike events as raster
            handles = ax.eventplot(
                        list_of_spike_idxs,
                        linelengths=0.8,
                        colors=['C{}'.format(i) for i in range(len(list_of_spike_idxs))]
                    )
            if legend: # determine whether to plot the legend
                ax.legend(handles[::-1],list(counts)[::-1], title="approx. cnt",loc="upper left")
            # ax.set_facecolor((200/255,200/255,200/255)) # set RGB color (gray)
            # fig.set_facecolor((40/255,40/255,40/255)) # set RGB color (235,210,180) is manila
            plt.title("spikes present across MU population during trial")
            plt.xlim((0,1000*(self.num_bins_per_trial/self.sample_rate)))
            plt.ylim((-1,self.num_units))
            plt.xlabel("time (ms)")
            plt.ylabel("motor unit spikes sorted by threshold")
            plt.show()
        elif target == 'smooth': # shows smoothed unit traces, 1 trial at a time
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            for ii in range(self.num_units): 
                if no_offset is False:    
                    if len(self.smooth_session)!=0:
                        max_smooth_val = self.smooth_session[session].max()/2
                        plt.plot(self.smooth_session[session][:,ii,trial]/max_smooth_val+ii)
                    elif len(self.smooth_spikes)!=0:
                        max_smooth_val = self.smooth_spikes[trial].max()/2
                        plt.plot(self.smooth_spikes[trial][:,ii]/max_smooth_val+ii)
                    else:
                        raise Exception("there is no smoothed spiking data. run '.convolve()' method to smooth spikes.")
                else:
                    if len(self.smooth_session)!=0:
                        plt.plot(self.smooth_session[session][:,ii,trial])
                    elif len(self.smooth_spikes)!=0:
                        plt.plot(self.smooth_spikes[trial][:,ii])
                    else:
                        raise Exception("there is no smoothed spiking data. run '.convolve()' method to smooth spikes.")
            plt.title("smoothed spikes present across MU population during trial")
            plt.xlabel("time (ms)")
            plt.ylabel("activation level (smoothed spikes)")
            plt.show()
        elif target == 'unit': # looks at 1 unit, shows all trials in a session
            if len(self.smooth_session)==0:
                raise Exception("there is no smoothed session data. run '.convolve(target='session')' method to smooth a session.")
            else:
                plt.plot(self.smooth_session[session][:,unit,:],color='skyblue',alpha=.5)
                plt.plot(np.mean(self.smooth_session[session][:,unit,:],axis=1),color='darkblue')
                plt.title("smoothed rates for unit #"+str(unit)+" across "+str(self.num_trials)+" trials")
                plt.xlabel("time (ms)")
                plt.ylabel("activation level (smoothed spikes)")
                plt.show()
        elif target == 'rates': # shows smoothed unit traces, 1 trial at a time
            if len(self.smooth_session)!=0:
                max_smooth_val = self.smooth_session[session].max()/2
                plt.imshow(self.smooth_session[session][:,:,trial].T/max_smooth_val,aspect='auto',interpolation='none')
            elif len(self.smooth_spikes)!=0:
                max_smooth_val = self.smooth_spikes[trial].max()/2
                plt.imshow(self.smooth_spikes[trial].T/max_smooth_val,aspect='auto',interpolation='none')
            else:
                raise Exception("there is no smoothed spiking data. use '.convolve()' method to smooth spikes.")
            clb = plt.colorbar()
            clb.set_label('rates', labelpad=-30, y=1.05, rotation=0)
            plt.title("smoothed rates for all units")
            plt.xlabel("time (ms)")
            plt.ylabel("motor units")
            plt.show()
        elif target == 'ave_rates': # shows all units' trial-averaged rates in a session
            if len(self.smooth_session)==0:
                raise Exception("there is no smoothed session data. run '.convolve(target='session')' method to smooth a session.")
            else:
                trial_ave_responses = np.mean(self.smooth_session[session],axis=2)
                max_smooth_val = trial_ave_responses.max()/2
                plt.imshow(trial_ave_responses.T/max_smooth_val,aspect='auto',interpolation='none')
                clb = plt.colorbar()
                clb.set_label('rates', labelpad=-30, y=1.05, rotation=0)
                plt.title("trial-averaged smoothed rates for all units")
                plt.xlabel("time (ms)")
                plt.ylabel("motor units")
                plt.show()
        elif target == 'lorenz':
            fig = go.Figure(data=[go.Scatter3d(
                x=self.units[1][:,0],
                y=self.units[1][:,1],
                z=self.units[1][:,2],
                mode='lines')])
            fig.show()
        else:
            raise Exception("invalid keyword argument for '.see()' visualization method.")