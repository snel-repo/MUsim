from MUsim import MUsim
import numpy as np
from pdb import set_trace

ephys_fs = 3000
random_seed_entropy = 75092699954400878964964014863999053929  # None
num_motor_units = 6
units_to_delay = [3, 4, 5]  # index values
i = 0
mu = MUsim(random_seed_entropy)
mu.num_units = num_motor_units  # set same number of motor units as in the Kilosort data
mu.MUthresholds_dist = "exponential"  # set the distribution of motor unit thresholds
mu.MUspike_dynamics = "spike_history"
mu.sample_rate = ephys_fs  # 30000 Hz
mu.threshmax = 10  # default 10
mu.threshmin = 2  # default 2
force_profile = 10 * np.sin(np.linspace(0, 10 * np.pi, 10000))
# sine wave for anipose, line 172 in generate_syn_data for later
##
# 60 seconds * ephys_fs
mu.sample_MUs()
mu.apply_new_force(force_profile)
mu.see("force")
# spikes = mu.simulate_spikes()
spikes = mu.simulate_spikes()
#     spike_delay=[1000, 1000, 2000) #, units_to_delay=units_to_delay
# )
mu.convolve(sigma=5)
mu.see("thresholds")
mu.see("curves")
mu.see("spikes")
mu.see("smooth")
