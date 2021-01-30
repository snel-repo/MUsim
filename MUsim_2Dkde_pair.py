# %% IMPORT packages
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from MUsim import MUsim

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

# %% SIMULATE MOTOR UNIT RESPONSES TO PROFILE 1 AND PROFILE 2
########################################################
# Define Simulation Parameters 
num_trials_to_simulate = 20
num_units_to_simulate = 10
gaussian_bw = 40                # choose smoothing bandwidth
unit1 = 0; unit2 = -1           # choose units to analyze
maxforce1 = 5; maxforce2 = 15   # choose max force to analyze, default is 5
# want to shuffle the second session's thresholds?
# if not, set False below
shuffle_second_MU_thresholds=False
#############################################################################################
# RUN 2 DIFFERENT SESSIONS
mu = MUsim()                            # INSTANTIATE SIMULATION OBJECT
mu.num_units = num_units_to_simulate    # SET NUMBER OF UNITS TO SIMULATE
mu.num_trials = num_trials_to_simulate  # SET NUMBER OF TRIALS TO SIMULATE
units = mu.sample_MUs(MUmode='static')  # SAMPLE MUs
# FIRST SESSION
force_profile = maxforce1/mu.init_force_profile.max()*mu.force_profile  # SCALE DEFAULT FORCE
mu.apply_new_force(force_profile)       # SET SCALED LINEAR FORCE PROFILE
session1 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
session1_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 1
# SECOND SESSION
mu.reset_force                          # RESET FORCE BACK TO DEFAULT
force_profile = maxforce2/mu.init_force_profile.max()*mu.force_profile  # SCALE DEFAULT FORCE
mu.apply_new_force(force_profile)       # SET SCALED LINEAR FORCE PROFILE
session2 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
session2_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 2
#############################################################################################
# %% COMPUTE UNIT DATA MATRICES
# Get 2 aligned channels of data
session1_smooth_stack = np.hstack(session1_smooth)
session2_smooth_stack = np.hstack(session2_smooth)
if shuffle_second_MU_thresholds is True:
    np.random.shuffle(session2_smooth_stack) #shuffle in place

mu1_session1 = session1_smooth_stack[unit1,:]
mu2_session1 = session1_smooth_stack[unit2,:]
mu1_session2 = session2_smooth_stack[unit1,:]
mu2_session2 = session2_smooth_stack[unit2,:]

# get condition-averages for each
mu1_session1_ave = mu1_session1.reshape((len(mu.force_profile),num_trials_to_simulate)).mean(axis=1)
mu2_session1_ave = mu2_session1.reshape((len(mu.force_profile),num_trials_to_simulate)).mean(axis=1)
mu1_session2_ave = mu1_session2.reshape((len(mu.force_profile),num_trials_to_simulate)).mean(axis=1)
mu2_session2_ave = mu2_session2.reshape((len(mu.force_profile),num_trials_to_simulate)).mean(axis=1)

## Format data vectors into D x N shape
mu12_session1 = np.vstack([mu1_session1,mu2_session1])
mu12_session2 = np.vstack([mu1_session2,mu2_session2])
mu12_session12 = np.hstack((mu12_session1,mu12_session2))

# %% GET KDE OBJECTS, fit on each matrix
kde10 = gaussian_kde(mu12_session1)
kde20 = gaussian_kde(mu12_session2)

# get mins, maxes for both datasets
x_both_min, y_both_min = mu12_session12[0,:].min(), mu12_session12[1,:].min()
x_both_max, y_both_max = mu12_session12[0,:].max(), mu12_session12[1,:].max()

# Evaluate kde on a grid
grid_margin = 20 # percent
gm_coef = (grid_margin/100)+1 # grid margin coefficient to extend grid beyond all edges
xi, yi = np.mgrid[(gm_coef*x_both_min):(gm_coef*x_both_max):100j, (gm_coef*y_both_min):(gm_coef*y_both_max):100j]
coords = np.vstack([item.ravel() for item in [xi, yi]]) 
density_session1 = kde10(coords).reshape(xi.shape)
density_session2 = kde20(coords).reshape(xi.shape)

density_session1_pts = kde10(mu12_session1)
density_session2_pts = kde20(mu12_session2)

# normalize these to get probabilities
d_session1_norm = density_session1/np.sum(density_session1)
d_session2_norm = density_session2/np.sum(density_session2)

d_session1_norm_pts = density_session1_pts/np.sum(density_session1_pts)
d_session2_norm_pts = density_session2_pts/np.sum(density_session2_pts)

# %% PLOT SIMULATED DATA
fig = go.Figure()
# data session 2
fig.add_trace(go.Scatter(
    x = mu12_session2[0],
    y = mu12_session2[1],
    mode="markers",
    marker=dict(
        size=.75,
        opacity=.5,
        color="skyblue"
        ),
    name = "yank="+str(maxforce2)
))
# data session 1
fig.add_trace(go.Scatter(
    x = mu12_session1[0],
    y = mu12_session1[1],
    mode="markers",
    marker=dict(
        size=.75,
        opacity=.5,
        color="gray"
        ),
    name = "yank="+str(maxforce1)
))
# trial average session 2
fig.add_trace(go.Scatter(
    x = mu1_session2_ave,
    y = mu2_session2_ave,
    mode="lines",
    line=dict(
        width=1,
        color="blue"
        ),
    name = "yank="+str(maxforce2)+" mean"
))
# trial average session 1
fig.add_trace(go.Scatter(
    x = mu1_session1_ave,
    y = mu2_session1_ave,
    mode="lines",
    line=dict(
        width=1,
        color="black"
        ),
    name = "yank="+str(maxforce1)+" mean"
))
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.update_layout(
    legend=dict(title="linear force profiles:"),
    title="simulated trajectories in 2-unit state-space",
    xaxis=dict(title=dict(text="'small' unit")),
    yaxis=dict(title=dict(text="'large' unit")),
    width=600,
    height=500
)
fig.show()
# %%
fig10 = px.imshow(d_session1_norm.T,title="2D PDF, yank="+str(maxforce1),width=500,height=500,origin='lower')
fig20 = px.imshow(d_session2_norm.T,title="2D PDF, yank="+str(maxforce2),width=500,height=500,origin='lower')
fig10.show(); fig20.show()
# %%
CI_10 = get_confidence(d_session1_norm,.95)
CI_20 = get_confidence(d_session2_norm,.95)
# %% plot Computed CI's for each dataset
figCI10 = px.imshow(CI_10.T,title="yank="+str(maxforce1)+", 95% CI",width=500,height=500,origin='lower')
figCI20 = px.imshow(CI_20.T,title="yank="+str(maxforce2)+", 95% CI",width=500,height=500,origin='lower')
figCI10.show(); figCI20.show()
# %%
O_10in20 = np.sum(CI_20*d_session1_norm)
O_20in10 = np.sum(CI_10*d_session2_norm)

fig_O_10in20 = px.imshow((CI_20*d_session1_norm).T,title="yank="+str(maxforce1)+" density inside 95%CI of yank="+str(maxforce2)+": "+str(np.round(O_10in20,decimals=4)),width=500,height=500,origin='lower')
fig_O_20in10 = px.imshow((CI_10*d_session2_norm).T,title="yank="+str(maxforce2)+" density inside 95%CI of yank="+str(maxforce1)+": "+str(np.round(O_20in10,decimals=4)),width=500,height=500,origin='lower')
fig_O_10in20.show(); fig_O_20in10.show()
# %% 
