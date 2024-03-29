# %% IMPORT NECESSARY PACKAGES
import numpy as np

from MUsim import MUsim

#########################################################
#########################################################
#########################################################
# %%
# # TRADITIONAL MODE (SIZE PRINCIPLE)
# INITIALIZE SIMULATION OBJECT, mu_stat
random_seed = None  # set a random seed for reproducibility, None for completely random
if random_seed is None:
    random_seed_sqn = np.random.SeedSequence()
else:
    random_seed_sqn = np.random.SeedSequence(random_seed)
mu_stat = MUsim(random_seed_sqn)
# GET STATIC MOTOR UNIT THRESHOLDS
mu_stat.num_units = 32
mu_stat.MUthresholds_dist = "normal"
static_units = mu_stat.sample_MUs()
# %% PLOT THRESHOLD DISTRIBUTION, FORCE PROFILE, AND INDIVIDUAL UNIT RESPONSES
mu_stat.see("thresholds")  # plot binned thresholds across all units
mu_stat.see("force")  # plot default applied force
mu_stat.see("curves")  # plot unit response curves

# %% SIMULATE MOTOR UNITS SPIKE RESPONSE TO DEFAULT FORCE
spikes = mu_stat.simulate_spikes(noise_level=0)
mu_stat.see("spikes")  # plot spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth = mu_stat.convolve()
mu_stat.see("smooth")  # plot smoothed spike response
# %% APPLY NEW FORCE, VIEW RESPONSE
new_force_profile = 3 * mu_stat.init_force_profile
mu_stat.apply_new_force(new_force_profile)
spikes2 = mu_stat.simulate_spikes()
mu_stat.see("force")  # plot new applied force
mu_stat.see("curves")  # plot unit response curves
mu_stat.see("spikes")  # plot spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth = mu_stat.convolve()
mu_stat.see("smooth")

# %% SIMULATE SESSION (MANY TRIALS)
num_trials_to_simulate = 20
mu_stat.num_trials = num_trials_to_simulate
results = mu_stat.simulate_session()
# CONVOLVE ENTIRE SESSION
smooth_results = mu_stat.convolve(target="session")
num_units_to_view = 4
select_units = np.linspace(0, mu_stat.num_units - 1, num_units_to_view).astype(int)
mu_stat.see("unit", unit=select_units[0])
mu_stat.see("unit", unit=select_units[1])
mu_stat.see("unit", unit=select_units[2])
mu_stat.see("unit", unit=select_units[3])
# %% ###################################################
mu_lorenz = MUsim(random_seed_sqn)
# GET LORENZ SIMULATED MOTOR UNITS
mu_lorenz.num_units = 30
mu_lorenz.sample_rate = 1 / (0.006)  # 166.7 Hz
mu_lorenz.MUthresholds_dist = "uniform"
lorenz_units = mu_lorenz.sample_MUs(MUmode="lorenz")
# %% SIMULATE MOTOR UNITS SPIKING RULED BY LORENZ DYNAMICS
spikes = mu_lorenz.simulate_spikes(noise_level=0)
mu_lorenz.see("spikes")  # plot spike response
# %% CONVOLVE AND PLOT SMOOTHED SPIKES
smooth = mu_lorenz.convolve()
mu_lorenz.see("smooth")  # plot smoothed spike response
# %% VIEW LORENZ ATTRACTOR
mu_lorenz.see("lorenz")
import matplotlib.pyplot as plt

#########################################################
# %% IMPORT NECESSARY PACKAGES
import numpy as np

from MUsim import MUsim

#########################################################
#########################################################
#########################################################
# %% # DYNAMIC MODE (THRESHOLD REVERSAL)
# INITIALIZE SIMULATION OBJECT, mu_dyn
mu_dyn = MUsim(random_seed_sqn)
# # GET DYNAMIC MOTOR UNIT THRESHOLDS
mu_dyn.num_units = 10
mu_dyn.MUthresholds_dist = "uniform"
mu_dyn.MUreversal_frac = 1  # set fraction of MU population that will reverse
mu_dyn.MUreversal_static_units = list(range((mu_dyn.num_units - 1)))
dyn_units = mu_dyn.sample_MUs(MUmode="dynamic")
# %% PLOT THRESHOLD DISTRIBUTION, FORCE PROFILE, AND INDIVIDUAL UNIT RESPONSES
mu_dyn.see("thresholds")  # plot binned thresholds across all units
mu_dyn.see("force")  # plot default applied force
mu_dyn.see("curves")  # plot unit response curves

# %% SIMULATE MOTOR UNITS SPIKE RESPONSE TO DEFAULT FORCE
spikes1 = mu_dyn.simulate_spikes()
mu_dyn.see("spikes")  # plot spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth1 = mu_dyn.convolve()
mu_dyn.see("smooth")  # plot smoothed spike response
# %% APPLY NEW FORCE, VIEW RESPONSE
new_force_profile = 5 * (mu_dyn.init_force_profile)
mu_dyn.apply_new_force(new_force_profile)
spikes2 = mu_dyn.simulate_spikes()
smooth2 = mu_dyn.convolve()
mu_dyn.see("force")  # plot new applied force
mu_dyn.see("curves")  # plot unit response curves
mu_dyn.see("spikes")  # plot spike response
mu_dyn.see("smooth")  # plot smoothed spike response

# %% APPLY NON-LINEAR FORCE, VIEW RESPONSE
new_force_profile = -3 * np.cos(mu_dyn.init_force_profile)
mu_dyn.apply_new_force(new_force_profile)
spikes3 = mu_dyn.simulate_spikes()
smooth3 = mu_dyn.convolve()
mu_dyn.see("force")  # plot new applied force
mu_dyn.see("curves")  # plot unit response curves
mu_dyn.see("spikes")  # plot spike response
mu_dyn.see("smooth")  # plot smoothed spike response

# %% CONVOLVE AND PLOT SMOOTHED RESPONSE
smooth = mu_dyn.convolve()
mu_dyn.see("smooth")

# %% SIMULATE SESSION (MANY TRIALS)
num_trials_to_simulate = 20
mu_dyn.num_trials = num_trials_to_simulate
results = mu_dyn.simulate_session()
# CONVOLVE ENTIRE SESSION
smooth_results = mu_dyn.convolve(target="session")
num_units_to_view = 4
select_units = np.linspace(0, mu_dyn.num_units - 1, num_units_to_view).astype(int)
mu_dyn.see("unit", unit=select_units[0])
mu_dyn.see("unit", unit=select_units[1])
mu_dyn.see("unit", unit=select_units[2])
mu_dyn.see("unit", unit=select_units[3])

# %%
