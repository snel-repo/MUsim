import numpy.random
import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class MUsim():

    def __init__(self):
        self.units = [[],[]]
        self.num_units = 10
        self.sample_rate = 1000 # Hz
        self.init_force_profile = np.linspace(0,5,1000)
        self.force_profile = self.init_force_profile
        self.spike_prob = 0.08 # set to achieve ~50hz (for typical MU rates)

    def set_spiking_probability(self,thresholded_forces):
        p = self.spike_prob
        unit_response_curves = 1-expit(thresholded_forces)
        scaled_unit_response_curves = (unit_response_curves*p)+(1-p)
        return scaled_unit_response_curves

    def recruit(self,threshmax=7,threshmin=1):
        """ first argument is number of units,
            second argument gives the force array.
            MU thresholds will be distributed from one-tailed Gaussian

            Returns: list of lists,
                units[0] holds thresh of each unit,
                units[1] holds response curves from each neuron
        """ 
        units = self.units
        num_units = self.num_units
        force_profile = self.force_profile
        # thresholds from one-tailed normal dist (produces more small, low threshold units)
        MUthresholds = np.clip((np.round(threshmax*abs(np.random.randn(num_units)/2),decimals=4)),threshmin,threshmax)
      
        # units[0] holds thresh of each unit,
        # units[1] holds response curves from each neuron
        units[0] = MUthresholds
        all_forces = np.repeat(force_profile,num_units).reshape((len(force_profile),num_units))
        thresholded_forces = all_forces - MUthresholds
        # subtract each respective threshold to get unique response
        units[1] = self.set_spiking_probability(thresholded_forces)
        
        spike_sorted_cols = self.units[0].argsort()
        self.units[1] = units[1][:,spike_sorted_cols]
        self.units[0] = units[0][spike_sorted_cols]
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
            spikes[spike_idxs] = 1 # assigne spikes
        except:
            if len(self.units[0])==0:
                raise Exception("unit response curve empty. run '.recruit()' method to define motor units.")
        # generate spike frequencies according to poisson process
        # for ii,iUnit_curve in enumerate( unit_curves ):
        #     for jj,jSpike in enumerate( iUnit_curve ):
        #         selection = np.random.random()
        #         if spikes[ii,jj-1] is 1:
        #             spikes[ii,jj] = 0 # prevent 2 spikes in a row
        #         else:
        #             spikes[ii,jj] = 1 if selection > iUnit_curve[ jj ] else 0 # get spikes
        # spike_sorted_cols = self.units[0].argsort()
        self.spikes = spikes #[:,spike_sorted_cols]
        return self.spikes

    def simulate_session(self,trials=50):
        # NEED TO ADD ABILITY TO SAVE SESSIONS, (e.g., sess1,sess2)
        data_shape = (len(self.force_profile),self.num_units,trials)
        self.session = np.zeros(data_shape)
        for iTrial in range(trials):
            self.session[:,:,iTrial] = self.simulate_trial()
        return self.session

    def apply_new_force(self,force_profile):
        if len(self.units[0])==0:
            self.simulate_trial()
        all_forces = np.repeat(force_profile,self.num_units).reshape((len(force_profile),self.num_units))
        thresholded_forces = all_forces - self.units[0]
        # subtract each respective threshold to get unique response
        self.units[1] = self.set_spiking_probability(thresholded_forces)
        self.force_profile = force_profile # set new force profile value

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
            if legend: plt.legend(self.units[0],title='thresholds')
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