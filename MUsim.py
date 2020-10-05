import numpy.random
import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class MUsim():

    def __init__(self):
        self.MUmode="static"    # "static" for size-principle obediance, "dynamic" for yank-dependent thresholds
        self.units = [[],[]]
        self.spikes = []
        self.session = []
        self.smooth_spikes = []
        self.smooth_session = []
        self.num_units = 10     # default number of units to simulate
        self.num_trials = 50    # default number of trials to simulate
        self.sample_rate = 1000 # Hz, default sampling rate
        self.init_force_profile = np.linspace(0,5,self.sample_rate)  # define initial force profile
        self.force_profile = self.init_force_profile  # initialize force profile
        self.init_yank_profile = np.round(self.sample_rate*np.diff(self.force_profile),decimals=10)  # define initial yank profile
        self.init_yank_profile = np.append(self.init_yank_profile,self.init_yank_profile[-1])  # repeat last value to match force profile length
        self.yank_profile = self.init_yank_profile  # initialize yank profile
        self.yank_flip_thresh = 10  # yank value at which the thresholds flips 
        self.spike_prob = 0.08  # set to achieve ~50hz (simulate realistic MU firing rates)
        self.threshmax = 7      # fixed maximum threshold for the generated units' response curves
        self.threshmin = 1      # fixed minimum threshold for the generated units' response curves

    def _get_spiking_probability(self,thresholded_forces):
        # vertical scaling and offset of each units' sigmoid response curve, to simulate realistic MU firing rates
        p = self.spike_prob
        unit_response_curves = 1-expit(thresholded_forces)
        scaled_unit_response_curves = (unit_response_curves*p)+(1-p)
        return scaled_unit_response_curves

    def _get_dynamic_thresholds(self,threshmax,threshmin,new=False):
        # create arrays from original static MU thresholds, to define MU thresholds that change with each timestep 
        if new is True:
            MUthresholds_gen = np.clip((np.round(threshmax*abs(np.random.randn(self.num_units)/2),decimals=4)),threshmin,threshmax)
            MUthresholds = np.repeat(MUthresholds_gen,len(self.force_profile)).reshape(len(MUthresholds_gen),len(self.force_profile)).T
        else:
            MUthresholds_orig = self.MUthreshold_original
            MUthresholds_flip = -(MUthresholds_orig-threshmax-threshmin) # values flip within range
            MUthresholds = np.nan*np.zeros(MUthresholds_orig.shape) # place holder

            yank_mat = np.repeat(self.yank_profile,self.num_units).reshape(len(self.yank_profile),self.num_units)
            yank_flip_idxs = np.where(yank_mat>=self.yank_flip_thresh)
            yank_no_flip_idxs = np.where(yank_mat<self.yank_flip_thresh)
            MUthresholds[yank_flip_idxs] = MUthresholds_flip[yank_flip_idxs] # set flips
            MUthresholds[yank_no_flip_idxs] = MUthresholds_orig[yank_no_flip_idxs] # set original values
        return MUthresholds

    def recruit(self,MUmode="static"):
        """ 
        Input:
            MUmode: decide whether unit thresholds are fixed or dynamic

            MU thresholds will be distributed from one-tailed Gaussian, to simulate more small units, low threshold units
        Returns: list of lists,
                units[0] holds threshold of each unit,
                units[1] holds response curves from each neuron
        """ 
        units = self.units
        num_units = self.num_units
        force_profile = self.init_force_profile
        threshmax = self.threshmax
        threshmin = self.threshmin

        if MUmode is "static":
            MUthresholds = np.clip((np.round(threshmax*abs(np.random.randn(num_units)/2),decimals=4)),threshmin,threshmax)
        elif MUmode is "dynamic":
            MUthresholds = self._get_dynamic_thresholds(threshmax,threshmin,new=True)
        units[0] = MUthresholds
        all_forces = np.repeat(force_profile,num_units).reshape((len(force_profile),num_units))
        thresholded_forces = all_forces - MUthresholds
        # subtract each respective threshold to get unique response
        units[1] = self._get_spiking_probability(thresholded_forces)
        if MUmode is "static":
            spike_sorted_cols = self.units[0].argsort()
            self.units[0] = units[0][spike_sorted_cols]
        elif MUmode is "dynamic":
            spike_sorted_cols = self.units[0].mean(axis=0).argsort()
            self.units[0] = units[0][:,spike_sorted_cols]
            self.MUthreshold_original = self.units[0] # save original
        self.units[1] = units[1][:,spike_sorted_cols]
        self.MUmode = MUmode # record the last recruitment mode
        return units

    def simulate_spikes(self):
        """
            simple routine to approximate an inhomogenous Poisson process over short time intervals (bins).
            Input: unit response curve (probability of seeing a 0 each timestep, otherwise its 1)
            Returns: numpy array same length as response curve with  1's and 0's indicating spikes
        """
        if len(self.units[0])==0:
            raise Exception("unit response curve empty. run '.recruit()' method to define motor unit properties.")
        unit_response_curves = self.units[1]
        selection = np.random.random(unit_response_curves.shape)
        spike_idxs = np.where(selection>unit_response_curves)
        self.spikes.append(np.zeros(unit_response_curves.shape))
        self.spikes[-1][spike_idxs] = 1 # assign spikes
        return self.spikes[-1]

    def simulate_session(self):
        if len(self.units[0])==0:
            raise Exception("unit response curve empty. run '.recruit()' method to define motor unit properties.")
        trials = self.num_trials
        session_data_shape = (len(self.force_profile),self.num_units,trials)
        self.session.append(np.zeros(session_data_shape))
        for iTrial in range(trials):
            self.session[-1][:,:,iTrial] = self.simulate_spikes()
        return self.session[-1]

    def apply_new_force(self,input_force_profile):
        """
        feed in a new 1D force profile to apply to all recruited motor units
        """
        if len(self.units[0])==0:
            self.simulate_spikes()
        all_forces = np.repeat(input_force_profile,self.num_units).reshape((len(input_force_profile),self.num_units))
        if self.MUmode is "static":
            thresholded_forces = all_forces - self.units[0]
        elif self.MUmode is "dynamic": # everything must be updated for dynamic
            self.yank_profile = np.round(self.sample_rate*np.diff(input_force_profile),decimals=10) # update yank_profile
            self.yank_profile = np.append(self.yank_profile,self.yank_profile[-1]) # repeat yank_profile[-1] value [to match len(force_profile)]
            MUthresholds = self._get_dynamic_thresholds(self.threshmax,self.threshmin)
            thresholded_forces = all_forces - MUthresholds
            self.units[0] = MUthresholds

        self.units[1] = self._get_spiking_probability(thresholded_forces)
        self.force_profile = input_force_profile # set new force profile value

    def reset_force(self):
        self.force_profile = self.init_force_profile

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
        if target is 'spikes':
            num_units_in_last_trial = self.spikes[-1].shape[1]
            self.smooth_spikes.append(np.zeros(self.spikes[-1].shape)) # create new list entry
            for iUnit in range(num_units_in_last_trial):
               self.smooth_spikes[-1][:,iUnit] = gaussian_filter1d(self.spikes[-1][:,iUnit],sigma)
            return self.smooth_spikes[-1]
        elif target is 'session':
            num_trials_in_last_session = self.session[-1].shape[2]
            num_units_in_last_session = self.session[-1].shape[1]
            session_data_shape = (len(self.force_profile),num_units_in_last_session,num_trials_in_last_session)
            self.smooth_session.append(np.zeros(session_data_shape)) # create new list entry
            for iUnit in range(num_units_in_last_session):
                for iTrial in range(num_trials_in_last_session):
                    self.smooth_session[-1][:,iUnit,iTrial] = gaussian_filter1d(self.session[-1][:,iUnit,iTrial],sigma)
            return self.smooth_session[-1]

    def see(self,target='curves',trial=-1,unit=0,legend=True):
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
        if target is 'thresholds':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            if self.MUmode is "static":
                thresholds = self.units[0]
            elif self.MUmode is "dynamic":
                thresholds = self.units[0].mean(axis=0)
            plt.hist(thresholds,self.num_units)
            plt.title('thresholds across '+str(self.num_units)+' generated units')
            plt.ylabel("count")
            plt.xlabel("threshold values (shift applied to response curve)")
            plt.show()
        elif target is 'force':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.seismic(np.linspace(0,1,4)))
            plt.plot(self.init_force_profile)
            plt.plot(self.init_yank_profile)
            plt.plot(self.force_profile)
            plt.plot(self.yank_profile)
            plt.legend(["default force","default yank","current force","current yank"])
            plt.title("force and yank profiles for simulation")
            plt.ylabel("simulated force and yank (a.u.)")
            plt.xlabel("time (ms)")
            plt.show()
        elif target is 'curves':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            if self.num_units > 10 and legend is True: 
                legend=False
                print("could not plot legend with more than 10 MUs.")
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if legend: # flip legend to match data; handle dynamic vs. static (arrays vs. scalars)
                if self.MUmode is "static":
                    thresholds = self.units[0]
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:,ii],label=thresholds[ii])
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[::-1], labels[::-1], title="thresholds",loc="lower left")
                elif self.MUmode is "dynamic":
                    # take mean to reduce to 1 number
                    thresholds = self.units[0].mean(axis=0).round()
                    for ii in range(self.num_units):
                        ax.plot(self.units[1][:,ii],label=thresholds[ii])
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[::-1], labels[::-1], title="mean thresholds",loc="lower left")
            plt.title("randomly generated unit response curves")
            plt.xlabel("time (ms)")
            plt.ylabel("probability of zero spikes in each bin")
            plt.show()
        elif target is 'spikes':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            # check whether legend should be placed outside if too many MU's
            if self.num_units > 10 and legend is True: 
                legend=False
                print("could not plot legend with more than 10 MUs.")
            counts = np.sum(self.spikes[trial],axis=0)/len(self.force_profile)*self.sample_rate
            # plot spikes and space them out by integer offsets (with -ii)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for ii in range(self.num_units):
                ax.plot(self.spikes[trial][:,ii]+ii,label=counts[ii])
            if legend: # determine where to plot the legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], title="counts",loc="lower left")
            plt.title("spikes present across population during trial")
            plt.xlabel("spikes present over time (ms)")
            plt.ylabel("motor unit activities sorted by threshold")
            plt.show()
        elif target is 'smooth':
            for ii in range(self.num_units):
                if len(self.smooth_session)!=0:
                    max_smooth_val = self.smooth_session[-1].max()/2
                    plt.plot(self.smooth_session[-1][:,ii,trial]/max_smooth_val+ii)
                elif len(self.smooth_spikes)!=0:
                    max_smooth_val = self.smooth_spikes[-1].max()/2
                    plt.plot(self.smooth_spikes[trial][:,ii]/max_smooth_val+ii)
                else:
                    raise Exception("there is no smoothed spiking data. run '.convolve()' method to smooth spikes.")
            plt.title("smoothed spikes present across population during trial")
            plt.xlabel("time (ms)")
            plt.ylabel("activation level (smoothed spikes)")
            plt.show()
        elif target is 'unit':
            if len(self.smooth_session)==0:
                raise Exception("there is no smoothed session data. run '.convolve(target='session')' method to smooth a session.")
            else:
                plt.plot(self.smooth_session[-1][:,unit,:],color='skyblue',alpha=.5)
                plt.plot(np.mean(self.smooth_session[-1][:,unit,:],axis=1),color='darkblue')
                plt.title("smoothed rates for unit #"+str(unit)+" across "+str(self.num_trials)+" trials")
                plt.xlabel("time (ms)")
                plt.ylabel("activation level (smoothed spikes)")
                plt.show()