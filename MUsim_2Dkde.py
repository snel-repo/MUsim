# %% IMPORT packages
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats
import jupyter
import IPython
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psub
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

# %% SIMULATE MOTOR UNIT RESPONSES TO YANK 1 AND YANK 2
########################################################
# Define Simulation Parameters 
num_trials_to_simulate = 50
num_units_to_simulate = 10
gaussian_bw = 40        # choose smoothing bandwidth
unit1 = 0; unit2 = -1           # choose unit to analyze
yankval1 = 1; yankval2 = 3    # choose yank to analyze
########################################################
mu = MUsim()            # INSTANTIATE SIMULATION OBJECT
mu.num_units = num_units_to_simulate
units = mu.recruit(MUmode='static')    # RECRUIT
force_profile = yankval1*mu.force_profile # APPLY FORCE PROFILE (NON-DEFAULT)
mu.apply_new_force(force_profile)
session1 = mu.simulate_session(num_trials_to_simulate) # APPLY DEFAULT FORCE PROFILE
yank1 = mu.convolve(gaussian_bw, target="session") # SMOOTH SPIKES FOR SESSION 1

mu.reset_force() # RESET FORCE BACK TO DEFAULT
force_profile = yankval2*mu.force_profile # APPLY FORCE PROFILE (NON-DEFAULT)
mu.apply_new_force(force_profile)
session2 = mu.simulate_session(num_trials_to_simulate)
yank2 = mu.convolve(gaussian_bw, target="session") # SMOOTH SPIKES FOR SESSION 2
# %% COMPUTE UNIT DATA MATRICES
# Get 2 aligned channels of data
mu1_y1 = np.hstack(yank1[:,unit1,:])
mu2_y1 = np.hstack(yank1[:,unit2,:])
mu1_y2 = np.hstack(yank2[:,unit1,:])
mu2_y2 = np.hstack(yank2[:,unit2,:])

# get condition-averaged traces for each
mu1_y1_ave = np.mean(mu1_y1.reshape((len(mu.force_profile),num_trials_to_simulate)),axis=1)
mu2_y1_ave = np.mean(mu2_y1.reshape((len(mu.force_profile),num_trials_to_simulate)),axis=1)
mu1_y2_ave = np.mean(mu1_y2.reshape((len(mu.force_profile),num_trials_to_simulate)),axis=1)
mu2_y2_ave = np.mean(mu2_y2.reshape((len(mu.force_profile),num_trials_to_simulate)),axis=1)

## Format data vectors into D x N shape
mu12_y1 = np.vstack([mu1_y1,mu2_y1])
mu12_y2 = np.vstack([mu1_y2,mu2_y2])
mu12_y12 = np.hstack((mu12_y1,mu12_y2))

# %% GET KDE OBJECTS, fit on each matrix
kde10 = scipy.stats.gaussian_kde(mu12_y1)
kde20 = scipy.stats.gaussian_kde(mu12_y2)

# get mins, maxes for both datasets
x_both_min, y_both_min = mu12_y12[0,:].min(), mu12_y12[1,:].min()
x_both_max, y_both_max = mu12_y12[0,:].max(), mu12_y12[1,:].max()

# Evaluate kde on a grid
grid_margin = 20 # percent
gm_coef = (grid_margin/100)+1 # grid margin coefficient to extend grid beyond all edges
xi, yi = np.mgrid[(gm_coef*x_both_min):(gm_coef*x_both_max):100j, (gm_coef*y_both_min):(gm_coef*y_both_max):100j]
coords = np.vstack([item.ravel() for item in [xi, yi]]) 
density_y1 = kde10(coords).reshape(xi.shape)
density_y2 = kde20(coords).reshape(xi.shape)

density_y1_pts = kde10(mu12_y1)
density_y2_pts = kde20(mu12_y2)

# normalize these to get probabilities
d_y1_norm = density_y1/np.sum(density_y1)
d_y2_norm = density_y2/np.sum(density_y2)

d_y1_norm_pts = density_y1_pts/np.sum(density_y1_pts)
d_y2_norm_pts = density_y2_pts/np.sum(density_y2_pts)

# %% PLOT SIMULATED DATA
fig = go.Figure()
# data yank 2
fig.add_trace(go.Scatter(
    x = mu12_y2[0],
    y = mu12_y2[1],
    mode="markers",
    marker=dict(
        size=.75,
        opacity=.5,
        color="skyblue"
        ),
    name = "yank="+str(yankval2)
))
# data yank 1
fig.add_trace(go.Scatter(
    x = mu12_y1[0],
    y = mu12_y1[1],
    mode="markers",
    marker=dict(
        size=.75,
        opacity=.5,
        color="gray"
        ),
    name = "yank="+str(yankval1)
))
# trial average yank 2
fig.add_trace(go.Scatter(
    x = mu1_y2_ave,
    y = mu2_y2_ave,
    mode="lines",
    line=dict(
        width=1,
        color="blue"
        ),
    name = "yank="+str(yankval2)+" mean"
))
# trial average yank 1
fig.add_trace(go.Scatter(
    x = mu1_y1_ave,
    y = mu2_y1_ave,
    mode="lines",
    line=dict(
        width=1,
        color="black"
        ),
    name = "yank="+str(yankval1)+" mean"
))
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.update_layout(
    legend=dict(title="Force Profiles:"),
    title="Simulated trajectories in 2-unit State Space",
    xaxis=dict(title=dict(text="'small' unit")),
    yaxis=dict(title=dict(text="'large' unit")),
    width=600,
    height=500
)
fig.show()
# %%
fig10 = px.imshow(d_y1_norm.T,title="2D PDF, Yank="+str(yankval1),width=500,height=500,origin='lower')
fig20 = px.imshow(d_y2_norm.T,title="2D PDF, Yank="+str(yankval2),width=500,height=500,origin='lower')
fig10.show(); fig20.show()
# %%
CI_10 = get_confidence(d_y1_norm,.95)
CI_20 = get_confidence(d_y2_norm,.95)
# %% plot Computed CI's for each dataset
figCI10 = px.imshow(CI_10.T,title="Yank="+str(yankval1)+", 95% CI",width=500,height=500,origin='lower')
figCI20 = px.imshow(CI_20.T,title="Yank="+str(yankval2)+", 95% CI",width=500,height=500,origin='lower')
figCI10.show(); figCI20.show()
# %%
O_10in20 = np.sum(CI_20*d_y1_norm)
O_20in10 = np.sum(CI_10*d_y2_norm)

fig_O_10in20 = px.imshow((CI_20*d_y1_norm).T,title="Yank="+str(yankval1)+" density inside 95%CI of Yank="+str(yankval2)+": "+str(np.round(O_10in20,decimals=4)),width=500,height=500,origin='lower')
fig_O_20in10 = px.imshow((CI_10*d_y2_norm).T,title="Yank="+str(yankval2)+" density inside 95%CI of Yank="+str(yankval1)+": "+str(np.round(O_20in10,decimals=4)),width=500,height=500,origin='lower')
fig_O_10in20.show(); fig_O_20in10.show()
# %% 
