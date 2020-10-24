# %% IMPORT packages
import numpy as np
from scipy.stats import gaussian_kde
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from MUsim import MUsim
import plotly.express as px
import plotly.graph_objects as go

# Define confidence interval calculation
def get_confidence(normalized_KDE_densities,confidence_value):
    CI = np.zeros( normalized_KDE_densities.shape )
    # sort indexes to find max density locations
    sorted_idxs = np.argsort(normalized_KDE_densities,axis=None)[::-1]

    # cumulative sum to capture development of dist density across all idxs
    cumsums = np.cumsum(normalized_KDE_densities.ravel()[sorted_idxs])

    # get all idx's below the threshold probability level
    idxs_in_chosen_CI = np.where(cumsums < confidence_value)[0] # CI from 0 to 1

    # create 2D bit mask to draw CI in 1's against background of 0's
    bit_mask = np.zeros( CI.shape ).ravel()
    bit_mask[ sorted_idxs[idxs_in_chosen_CI] ] = 1
    bit_mask = bit_mask.reshape( CI.shape )

    CI = bit_mask
    return CI

# define functions and start pool for multiprocessing KDE fits
def kde_fit(data):
    kde_obj = gaussian_kde(data)
    return kde_obj
pool = Pool(processes=2)

# %% SIMULATE MOTOR UNIT RESPONSES TO SESSION 1 AND SESSION 2
########################################################
# Define Simulation Parameters 
num_trials_to_simulate = 50
num_units_to_simulate = 20
trial_length = 500 # bins
noise_level = 0
max_firing_rate = 20
gaussian_bw = 40                # choose smoothing bandwidth
max_force1 = 5; max_force2 = 10 # choose max force to analyze, default is 5
# want to shuffle the second session's thresholds?
# if not, set False below
shuffle_second_MU_thresholds=False
#############################################################################################
# RUN 2 DIFFERENT SESSIONS
mu = MUsim()                            # INSTANTIATE SIMULATION OBJECT
mu.num_units = num_units_to_simulate    # SET NUMBER OF UNITS TO SIMULATE
mu.num_trials = num_trials_to_simulate  # SET NUMBER OF TRIALS TO SIMULATE
mu.num_bins_per_trial = trial_length    # SET NUMBER OF BINS PER TRIAL
mu.max_spike_prob = max_firing_rate/mu.num_bins_per_trial # SET SPIKING PROBABILITY
mu.session_noise_level = noise_level    # SET NOISE LEVEL FOR SESSION
units = mu.recruit(MUmode='dynamic')    # RECRUIT
# FIRST SESSION
force_profile = max_force1/mu.init_force_profile.max()*mu.init_force_profile  # SCALE DEFAULT FORCE
mu.apply_new_force(force_profile)       # SET SCALED LINEAR FORCE PROFILE
session1 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
session1_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 1
# SECOND SESSION
mu.reset_force()                          # RESET FORCE BACK TO DEFAULT
force_profile = max_force2/mu.init_force_profile.max()*mu.init_force_profile  # SCALE DEFAULT FORCE
mu.apply_new_force(force_profile)       # SET SCALED LINEAR FORCE PROFILE
session2 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
session2_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 2
#############################################################################################
# %% COMPUTE PCA OBJECT on all MU data
session1_smooth_stack = np.hstack(session1_smooth)
session2_smooth_stack = np.hstack(session2_smooth)
if shuffle_second_MU_thresholds is True:
    np.random.shuffle(session2_smooth_stack) #shuffle in place
session12_smooth_stack = np.hstack((session1_smooth_stack,session2_smooth_stack)).T

# standardize all unit activities
scaler = StandardScaler(with_std=False)
session12_smooth_stack = scaler.fit_transform(session12_smooth_stack)

# run for top 10 components to see VAF development
try:
    num_comp_test = 10
    pca = PCA(n_components=num_comp_test)
    fit = pca.fit(session12_smooth_stack)
except:
    num_comp_test = num_units_to_simulate
    pca = PCA(n_components=num_comp_test)
    fit = pca.fit(session12_smooth_stack)
print("explained variance: "+str(fit.explained_variance_ratio_*100))
plt.scatter(range(num_comp_test),fit.explained_variance_ratio_*100)
plt.plot(np.cumsum(fit.explained_variance_ratio_*100),c='darkorange')
plt.hlines(70,0,num_comp_test,colors='k',linestyles="dashed") # show 70% VAF line
plt.title("explained variance for each additional PC")
plt.legend(["cumulative","individual","70% e.v."])
plt.xlabel("principal components")
plt.ylabel("explained variance (% e.v.)")
plt.show()

# run for only 2 components, that capture most variance
num_comp_proj = 2
pca = PCA(n_components=num_comp_proj)
fit = pca.fit(session12_smooth_stack)

# %% PROJECT FROM FIT
proj12_session1 = pca.transform(session1_smooth_stack.T)
proj12_session2 = pca.transform(session2_smooth_stack.T)
print('pca done.')
# %% COMPUTE TRIAL AVERAGES
# reshape into trials
proj12_session1_trials = proj12_session1.T.reshape((len(force_profile),num_comp_proj,num_trials_to_simulate))
proj12_session2_trials = proj12_session2.T.reshape((len(force_profile),num_comp_proj,num_trials_to_simulate))

# get condition-averages for each
proj12_session1_ave_x = proj12_session1[:,0].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
proj12_session1_ave_y = proj12_session1[:,1].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
proj12_session2_ave_x = proj12_session2[:,0].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
proj12_session2_ave_y = proj12_session2[:,1].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)

## Format data vectors into D x N shape
proj12_session12 = np.vstack((proj12_session1,proj12_session2))

# get mins, maxes for both datasets
x_both_min, y_both_min = proj12_session12[:,0].min(), proj12_session12[:,1].min()
x_both_max, y_both_max = proj12_session12[:,0].max(), proj12_session12[:,1].max()

# %% GET KDE OBJECTS, for each matrix
proj_list = [proj12_session1.T,proj12_session2.T]
kde_objs_list = pool.map(kde_fit,proj_list)
kde1, kde2 = kde_objs_list

# Evaluate kde on a grid
grid_margin = 10 # percent
gm = grid_margin/100 # grid margin value to extend grid beyond all edges
xi, yi = np.mgrid[(x_both_min-gm):(x_both_max+gm):100j, (y_both_min-gm):(y_both_max+gm):100j]
coords = np.vstack([item.ravel() for item in [xi, yi]]) 
density_session1 = kde1(coords).reshape(xi.shape)
density_session2 = kde2(coords).reshape(xi.shape)

density_session1_pts = kde1(proj12_session1.T)
density_session2_pts = kde2(proj12_session2.T)
print('kde done.')

# normalize these to get probabilities
d_session1_norm = density_session1/np.sum(density_session1)
d_session2_norm = density_session2/np.sum(density_session2)

d_session1_norm_pts = density_session1_pts/np.sum(density_session1_pts)
d_session2_norm_pts = density_session2_pts/np.sum(density_session2_pts)

# %% PLOT SIMULATED DATA
fig = go.Figure()
# data session 2
fig.add_trace(go.Scatter(
    x = proj12_session2[:,0],
    y = proj12_session2[:,1],
    mode="markers",
    marker=dict(
        size=.75,
        opacity=.5,
        color="skyblue"
        ),
    name = "yank="+str(max_force2*mu.sample_rate/mu.num_bins_per_trial)
))

# data session 1
fig.add_trace(go.Scatter(
    x = proj12_session1[:,0],
    y = proj12_session1[:,1],
    mode="markers",
    marker=dict(
        size=.75,
        opacity=.5,
        color="gray"
        ),
    name = "yank="+str(max_force1*mu.sample_rate/mu.num_bins_per_trial)
))
# trial average session 2
fig.add_trace(go.Scatter(
    x = proj12_session2_ave_x,
    y = proj12_session2_ave_y,
    mode="lines",
    line=dict(
        width=1,
        color="blue"
        ),
    name = "yank="+str(max_force2*mu.sample_rate/mu.num_bins_per_trial)+" mean"
))
# trial average session 1
fig.add_trace(go.Scatter(
    x = proj12_session1_ave_x,
    y = proj12_session1_ave_y,
    mode="lines",
    line=dict(
        width=1,
        color="black"
        ),
    name = "yank="+str(max_force1*mu.sample_rate/mu.num_bins_per_trial)+" mean"
))
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.update_layout(
    legend=dict(title="linear force profiles:"),
    title="simulated population trajectories in 2D PCA space",
    xaxis=dict(title=dict(text="First PC")),
    yaxis=dict(title=dict(text="Second PC")),
    width=600,
    height=500
)
fig.show(config=dict(staticPlot=True)) # not interactive to avoid slowness
# %%
fig10 = px.imshow(d_session1_norm.T,title="2D PDF, yank="+str(max_force1*mu.sample_rate/mu.num_bins_per_trial),width=500,height=500,origin='lower')
fig20 = px.imshow(d_session2_norm.T,title="2D PDF, yank="+str(max_force2*mu.sample_rate/mu.num_bins_per_trial),width=500,height=500,origin='lower')
fig10.show(); fig20.show()
# %%
CI_10 = get_confidence(d_session1_norm,.95)
CI_20 = get_confidence(d_session2_norm,.95)
# %% plot Computed CI's for each dataset
figCI10 = px.imshow(CI_10.T,title="yank="+str(max_force1*mu.sample_rate/mu.num_bins_per_trial)+", 95% CI",width=500,height=500,origin='lower')
figCI20 = px.imshow(CI_20.T,title="yank="+str(max_force2*mu.sample_rate/mu.num_bins_per_trial)+", 95% CI",width=500,height=500,origin='lower')
figCI10.show(); figCI20.show()
# %%
O_1in2 = np.sum(CI_20*d_session1_norm)
O_2in1 = np.sum(CI_10*d_session2_norm)

fig_O_1in2 = px.imshow((CI_20*d_session1_norm).T,title="yank="+str(max_force1*mu.sample_rate/mu.num_bins_per_trial)+" density inside 95%CI of yank="+str(max_force2*mu.sample_rate/mu.num_bins_per_trial)+": "+str(np.round(O_1in2,decimals=4)),width=500,height=500,origin='lower')
fig_O_2in1 = px.imshow((CI_10*d_session2_norm).T,title="yank="+str(max_force2*mu.sample_rate/mu.num_bins_per_trial)+" density inside 95%CI of yank="+str(max_force1*mu.sample_rate/mu.num_bins_per_trial)+": "+str(np.round(O_2in1,decimals=4)),width=500,height=500,origin='lower')
fig_O_1in2.show(); fig_O_2in1.show()
# %% 