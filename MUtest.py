# %% import
from MUsim import MUsim
import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
# initialize simulation object, mu_test
mu_test = MUsim()

#########################################################
#########################################################
#########################################################
# TRADITIONAL MODE (SIZE PRINCIPLE)
# %% RECRUIT NEW MOTOR UNITS
num_units = 10
mu_test.num_units = num_units
# units = mu_test.recruit(tmax,tmin)
units = mu_test.recruit()
plt.hist(units[0],2*num_units)
plt.title('thresholds across '+str(num_units)+' generated units')
plt.show()
# plot unit response curves 
if num_units<15:
    mu_test.vis(legend=True)
else:
    mu_test.vis()
# %% SIMULATE MOTOR UNITS FOR TRIAL
spikes = mu_test.simulate_trial()
plt.imshow(spikes.T,aspect=len(mu_test.force_profile)/mu_test.num_units)
plt.colorbar()
plt.title("spikes from each motor unit")
plt.xlabel("spikes present over time (ms)")
plt.ylabel("motor unit activities sorted by threshold")
plt.show()

# %% APPLY NEW FORCE
new_force = 1.5*mu_test.force_profile
mu_test.apply_new_force(new_force)
spikes = mu_test.simulate_trial()
plt.imshow(spikes.T,aspect=len(mu_test.force_profile)/mu_test.num_units)
plt.colorbar()
plt.title("spikes from each motor unit")
plt.xlabel("spikes present over time (ms)")
plt.ylabel("motor unit activities sorted by threshold")
plt.show()
# plot unit response curves 
mu_test.vis(legend=True)
# %% PLOT SPIKES AND COUNT
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,num_units)))
for ii in range(mu_test.num_units):
    plt.plot(mu_test.spikes[:,ii]-ii)

plt.title("spikes present across population")
rates = np.sum(mu_test.spikes,axis=0)/len(mu_test.force_profile)*mu_test.sample_rate
plt.xlabel("spikes present over time (ms)")
plt.ylabel("motor unit activities sorted by threshold")
plt.legend(rates,title="rate (Hz)",loc="lower left")
plt.show()
# # %%
# real_spike_mat = loadmat("W:\\datasets\\real_spike.mat")
# real_spike = real_spike_mat['spike'].T
# # %%
# plt.plot(real_spike)
# %%
mu_test.convolve(20)
for ii in range(mu_test.num_units):
    plt.plot(mu_test.smooth_spikes[:,ii]-ii/mu_test.num_units)
plt.title("smoothed spikes present across population")
plt.show()
# %% SIMULATE SESSION (MANY TRIALS)
results = mu_test.simulate_session(100)
# %% CONVOLVE ENTIRE SESSION
smooth_results = mu_test.convolve(target='session')

plt.plot(smooth_results[:,0,:],color='blue',alpha=.2)
plt.plot(np.mean(smooth_results[:,0,:],axis=1),color='k')
plt.title("'small' unit smoothed rates across 100 trials")
plt.show()

plt.plot(smooth_results[:,4,:],color='green',alpha=.2)
plt.plot(np.mean(smooth_results[:,4,:],axis=1),color='k')
plt.title("'medium' unit smoothed rates across 100 trials")
plt.show()

plt.plot(smooth_results[:,8,:],color='red',alpha=.2)
plt.plot(np.mean(smooth_results[:,8,:],axis=1),color='k')
plt.title("'large' unit smoothed rates across 100 trials")
plt.show()

plt.plot(smooth_results[:,9,:],color='maroon',alpha=.2)
plt.plot(np.mean(smooth_results[:,9,:],axis=1),color='k')
plt.title("'XL' unit smoothed rates across 100 trials")
plt.show()
# %% PLOT FORCE PROFILES

plt.plot(mu_test.force_profile/2)
plt.plot(mu_test.force_profile)
plt.title("Force Profiles for the 2 Simulations")
plt.legend(["yank=1","yank=2"])
plt.ylabel("Simulated Force (a.u.)")
plt.xlabel("Time (ms)")

# %%
#######################################################
#######################################################
#  DYNAMIC MODE
#################
# %% import
from MUsim import MUsim
import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
# initialize simulation object, mu_test
mu_test = MUsim()
# RECRUIT DYNAMIC UNITS
num_units = 10
mu_test.num_units = num_units
# units = mu_test.recruit(tmax,tmin)
units = mu_test.recruit(MUmode="dynamic")
#%%
plt.hist(units[0],2*num_units)
plt.title('thresholds across '+str(num_units)+' generated units')
plt.show()
# plot unit response curves 
if num_units<15:
    mu_test.vis(legend=True)
else:
    mu_test.vis()

# %% APPLY NEW FORCE
mu_test.reset_force()
new_force = 1.25*(mu_test.init_force_profile)
mu_test.apply_new_force(new_force)
spikes = mu_test.simulate_trial()
# plt.imshow(spikes.T,aspect=len(mu_test.force_profile)/mu_test.num_units)
# plt.colorbar()
# plt.title("spikes from each motor unit")
# plt.xlabel("spikes present over time (ms)")
# plt.ylabel("motor unit activities sorted by threshold")
# plt.show()

# plot unit response curves 
mu_test.vis(legend=True)
mu_test.vis('spikes')
# %% PLOT SPIKES AND COUNT
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,num_units)))
for ii in range(mu_test.num_units):
    plt.plot(mu_test.spikes[:,ii]-ii)

plt.title("spikes present across population")
rates = np.sum(mu_test.spikes,axis=0)/len(mu_test.force_profile)*mu_test.sample_rate
plt.xlabel("spikes present over time (ms)")
plt.ylabel("motor unit activities sorted by threshold")
plt.legend(rates,title="rate (Hz)",loc="lower left")
plt.show()

# %%
