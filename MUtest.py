# %% IMPORT NECESSARY PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from MUsim import MUsim
#########################################################
#########################################################
#########################################################
# TRADITIONAL MODE (SIZE PRINCIPLE)
# INITIALIZE SIMULATION OBJECT, mu_test_static
mu_test_static = MUsim()
# RECRUIT NEW MOTOR UNITS
num_units = 10
mu_test_static.num_units = num_units
static_units = mu_test_static.recruit()
# %% PLOT THRESHOLD DISTRIBUTION, FORCE PROFILE, AND INDIVIDUAL UNIT RESPONSES
mu_test_static.see('thresholds') # plot binned thresholds across all units
mu_test_static.see('force') # plot default applied force
mu_test_static.see() # plot unit response curves 

# %% SIMULATE MOTOR UNITS SPIKE RESPONSE TO DEFAULT FORCE
spikes1 = mu_test_static.simulate_spikes()
mu_test_static.see('spikes') # plot spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth = mu_test_static.convolve()
mu_test_static.see('activity') # plot smoothed spike response
# %% APPLY NEW FORCE, VIEW RESPONSE
new_force_profile = 3*mu_test_static.init_force_profile
mu_test_static.apply_new_force(new_force_profile)
spikes2 = mu_test_static.simulate_spikes()
mu_test_static.see('force') # plot new applied force
mu_test_static.see() # plot unit response curves
mu_test_static.see('spikes') # plot spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth = mu_test_static.convolve()
mu_test_static.see('activity')

# %% SIMULATE SESSION (MANY TRIALS)
num_trials_to_simulate = 20
mu_test_static.num_trials = num_trials_to_simulate
results = mu_test_static.simulate_session()
# CONVOLVE ENTIRE SESSION
smooth_results = mu_test_static.convolve(target='session')
num_units_to_view = 4
select_units = np.linspace(0,mu_test_static.num_units-1,num_units_to_view).astype(int)
mu_test_static.see('unit',unit=select_units[0])
mu_test_static.see('unit',unit=select_units[1])
mu_test_static.see('unit',unit=select_units[2])
mu_test_static.see('unit',unit=select_units[3])

# %% IMPORT NECESSARY PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from MUsim import MUsim
#########################################################
#########################################################
#########################################################
# DYNAMIC MODE (THRESHOLD REVERSAL)
# INITIALIZE SIMULATION OBJECT, mu_test_dyn
mu_test_dyn = MUsim()
# RECRUIT DYNAMIC UNITS
num_units = 10
mu_test_dyn.num_units = num_units
# units = mu_test_dyn.recruit(tmax,tmin)
dyn_units = mu_test_dyn.recruit(MUmode="dynamic")
# %% PLOT THRESHOLD DISTRIBUTION, FORCE PROFILE, AND INDIVIDUAL UNIT RESPONSES
mu_test_dyn.see('thresholds') # plot binned thresholds across all units
mu_test_dyn.see('force') # plot default applied force
mu_test_dyn.see() # plot unit response curves 

# %% SIMULATE MOTOR UNITS SPIKE RESPONSE TO DEFAULT FORCE
spikes1 = mu_test_dyn.simulate_spikes()
mu_test_dyn.see('spikes') # plot spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth = mu_test_dyn.convolve()
mu_test_dyn.see('activity') # plot smoothed spike response
# %% APPLY NEW FORCE, VIEW RESPONSE
new_force_profile = 3*(mu_test_dyn.init_force_profile)
mu_test_dyn.apply_new_force(new_force_profile)
spikes2 = mu_test_dyn.simulate_spikes()
mu_test_dyn.see('force') # plot new applied force
mu_test_dyn.see() # plot unit response curves
mu_test_dyn.see('spikes') # plot spike response

# %% APPLY NON-LINEAR FORCE, VIEW RESPONSE
new_force_profile = -3*np.cos(mu_test_dyn.init_force_profile)
mu_test_dyn.apply_new_force(new_force_profile)
spikes2 = mu_test_dyn.simulate_spikes()
mu_test_dyn.see('force') # plot new applied force
mu_test_dyn.see() # plot unit response curves
mu_test_dyn.see('spikes') # plot spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth = mu_test_dyn.convolve()
mu_test_dyn.see('activity')

# %% SIMULATE SESSION (MANY TRIALS)
num_trials_to_simulate = 20
mu_test_dyn.num_trials = num_trials_to_simulate
results = mu_test_dyn.simulate_session()
# CONVOLVE ENTIRE SESSION
smooth_results = mu_test_dyn.convolve(target='session')
num_units_to_view = 4
select_units = np.linspace(0,mu_test_dyn.num_units-1,num_units_to_view).astype(int)
mu_test_dyn.see('unit',unit=select_units[0])
mu_test_dyn.see('unit',unit=select_units[1])
mu_test_dyn.see('unit',unit=select_units[2])
mu_test_dyn.see('unit',unit=select_units[3])

# %%