import numpy.random
import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class MUsim():

    def __init__(self):
        self.MUmode="traditional" # "traditional" for size-principle obediance, "dynamic" for yank-dependent thresholds
        self.units = [[],[]]
        self.num_units = 10
        self.sample_rate = 1000 # Hz
        self.init_force_profile = np.linspace(0,5,self.sample_rate)
        self.force_profile = self.init_force_profile
        self.yank_profile = np.round(self.sample_rate*np.diff(self.force_profile),decimals=12)
        # repeat last yank_profile value to make same length as force_profile
        self.yank_profile = np.append(self.yank_profile,self.yank_profile[-1]) 
        self.yank_effect = .1 # scales the effect of yank on MU thresholds
        self.spike_prob = 0.08 # set to achieve ~50hz (for typical MU rates)
        self.threshmax = 7
        self.threshmin = 1

    def set_spiking_probability(self,thresholded_forces):
        p = self.spike_prob
        unit_response_curves = 1-expit(thresholded_forces)
        scaled_unit_response_curves = (unit_response_curves*p)+(1-p)
        return scaled_unit_response_curves

    def get_dynamic_thresholds(self,threshmax,threshmin):
        MUthresholds_orig = np.clip((np.round(threshmax*abs(np.random.randn(self.num_units)/2),decimals=4)),threshmin,threshmax)
        MUsubmin = MUthresholds_orig-threshmin
        threshrange = threshmax-threshmin
        MUthresholds_flip = (1-(MUsubmin/threshrange))*threshrange*threshmin # values flip within range
        MUthresholds_flip_rep = np.repeat(MUthresholds_flip,len(self.force_profile)).reshape(len(MUthresholds_flip),len(self.force_profile))
        MUthresholds_orig_rep = np.repeat(MUthresholds_orig,len(self.force_profile)).reshape(len(MUthresholds_orig),len(self.force_profile))
        clipped_yank = np.clip(self.yank_effect*self.yank_profile,0,10) # average below, weighted by the force slope
        # weighted average of flip with yank, elementwise
        MUthresholds = np.mean([MUthresholds_orig_rep.T,(MUthresholds_flip_rep*clipped_yank).T],axis=0)
        # import pdb; pdb.set_trace()
        return MUthresholds

    def recruit(self,MUmode="traditional"):
        """ 
            threshmax: fixed maximum threshold for the generated units' response curves
            threshmin: second argument is  fixed minimum threshold for the generated units' response curves
            MUmode: decide whether unit thresholds are fixed or dynamic

            MU thresholds will be distributed from one-tailed Gaussian, to simulate more small units, low threshold units
            Returns: list of lists,
                units[0] holds thresh of each unit,
                units[1] holds response curves from each neuron
        """ 
        units = self.units
        num_units = self.num_units
        force_profile = self.force_profile
        yank_effect = self.yank_effect # scales effect of yank on threshold
        threshmax = self.threshmax
        threshmin = self.threshmin

        if MUmode is "traditional":
            MUthresholds = np.clip((np.round(threshmax*abs(np.random.randn(num_units)/2),decimals=4)),threshmin,threshmax)
        elif MUmode is "dynamic":
            MUthresholds = self.get_dynamic_thresholds(threshmax,threshmin)
        units[0] = MUthresholds
        all_forces = np.repeat(force_profile,num_units).reshape((len(force_profile),num_units))
        thresholded_forces = all_forces - MUthresholds
        # subtract each respective threshold to get unique response
        units[1] = self.set_spiking_probability(thresholded_forces)
        if MUmode is "traditional":
            spike_sorted_cols = self.units[0].argsort()
            self.units[0] = units[0][spike_sorted_cols]
        elif MUmode is "dynamic":
            spike_sorted_cols = self.units[0].mean(axis=0).argsort()
            self.units[0] = units[0][:,spike_sorted_cols]
        self.units[1] = units[1][:,spike_sorted_cols]
        self.MUmode = MUmode # record the last recruitment mode
        return units

    def simulate_trial(self):
        # simple routine to simulate an inhomogenous
        # Poisson process over short time intervals
        # Input:
        # probability of seeing a 0 in each step
        # Returns:
        # numpy array of len(prob_0) with 0s and 1s indicating a spike or not
        unit_response_curves = self.units[1]
        try:
            selection = np.random.random(unit_response_curves.shape)
            spike_idxs = np.where(selection>unit_response_curves)
            spikes = np.zeros(unit_response_curves.shape)
            spikes[spike_idxs] = 1 # assign spikes
        except:
            if len(self.units[0])==0:
                raise Exception("unit response curve empty. run '.recruit()' method to define motor units.")
        self.spikes = spikes
        return self.spikes

    def simulate_session(self,trials=50):
        # NEED TO ADD ABILITY TO SAVE SESSIONS, (e.g., sess1,sess2)
        data_shape = (len(self.force_profile),self.num_units,trials)
        self.session = np.zeros(data_shape)
        for iTrial in range(trials):
            self.session[:,:,iTrial] = self.simulate_trial()
        return self.session

    def apply_new_force(self,input_force_profile):
        """
        feed in a new 1D force profile to apply to all recruited motor units
        """
        if len(self.units[0])==0:
            self.simulate_trial()
        all_forces = np.repeat(input_force_profile,self.num_units).reshape((len(input_force_profile),self.num_units))
        if self.MUmode is "traditional":
            thresholded_forces = all_forces - self.units[0]
        elif self.MUmode is "dynamic": # everything must be updated for dynamic
            self.yank_profile = np.round(self.sample_rate*np.diff(input_force_profile),decimals=12) # update yank_profile
            self.yank_profile = np.append(self.yank_profile,self.yank_profile[-1]) # repeat yank_profile[-1] value            
            MUthresholds = self.get_dynamic_thresholds(self.threshmax,self.threshmin)
            thresholded_forces = all_forces - MUthresholds
            self.units[1] = self.set_spiking_probability(thresholded_forces)
            self.units[0] = MUthresholds

        # subtract each respective threshold to get unique response
        self.units[1] = self.set_spiking_probability(thresholded_forces)
        self.force_profile = input_force_profile # set new force profile value

    def reset_force(self):
        self.force_profile = self.init_force_profile

    def convolve(self,sigma=40,target='spikes'): # default smoothing value of 40 bins
        if target is 'spikes':
            self.smooth_spikes = np.zeros(self.spikes.shape)
            for iUnit in range(self.num_units):
               self.smooth_spikes[:,iUnit] = gaussian_filter1d(self.spikes[:,iUnit],sigma)
            return self.smooth_spikes
        elif target is 'session':
            trials = self.session.shape[2]
            data_shape = (len(self.force_profile),self.num_units,trials)
            self.smooth_session = np.zeros(data_shape)
            for iUnit in range(self.num_units):
                for iTrial in range(trials):
                    self.smooth_session[:,iUnit,iTrial] = gaussian_filter1d(self.session[:,iUnit,iTrial],sigma)
            return self.smooth_session

    def vis(self,target='curves',legend=False):
        """
        target string inputs available for visualization:
            - 'curves': for the current MU response curves to the last force_profile
            - 'spikes': for the spike outputs from the last force response simulation
            - 'smooth': for the convolved outputs from the last force response simulation
        """
        if target is 'curves':
            # plot unit response curves 
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            plt.plot(self.units[1])
            if legend:
                if self.MUmode is "traditional":
                    plt.legend(self.units[0],title='thresholds')
                elif self.MUmode is "dynamic":
                    plt.legend(self.units[0].mean(axis=0).round(decimals=4),title='thresholds')
            plt.title("randomly generated unit response curves")
            plt.show()
        elif target is 'spikes':
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,self.num_units)))
            for ii in range(self.num_units):
                plt.plot(self.spikes[:,ii]-ii)
            plt.title("spikes present across population")
            rates = np.sum(self.spikes,axis=0)/len(self.force_profile)*self.sample_rate
            plt.xlabel("spikes present over time (ms)")
            plt.ylabel("motor unit activities sorted by threshold")
            if legend: plt.legend(rates,title="rate (Hz)",loc="lower left")
            plt.show()
        elif target is 'smooth':
            for ii in range(self.num_units):
                try:
                    plt.plot(self.smooth_spikes[:,ii]-ii)
                except:
                    plt.plot(self.smooth_session[:,ii,0]-ii)
                plt.title("smoothed spikes present across population")