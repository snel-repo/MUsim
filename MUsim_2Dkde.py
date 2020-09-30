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

# %% SIMULATE MOTOR UNIT RESPONSES TO YANK 1 AND YANK 2
##################################
mu = MUsim()            # INSTANTIATE
units = mu.recruit()    # RECRUIT
num_trials_to_simulate = 40
gaussian_bw = 20
session1 = mu.simulate_session(num_trials_to_simulate) # APPLY DEFAULT FORCE PROFILE
yank1 = mu.convolve(gaussian_bw, target="session") # SMOOTH SPIKES FOR SESSION 1

new_force = 2*mu.force_profile # APPLY NEW FORCE PROFILE (DOUBLE THE DEFAULT)
mu.apply_new_force(new_force)
session2 = mu.simulate_session(num_trials_to_simulate)
yank2 = mu.convolve(gaussian_bw, target="session") # SMOOTH SPIKES FOR SESSION 2
# %% Get 2 aligned channels of data
mu0_y1 = np.hstack(yank1[:,0,:])
mu8_y1 = np.hstack(yank1[:,8,:])

mu0_y2 = np.hstack(yank2[:,0,:])
mu8_y2 = np.hstack(yank2[:,8,:])

## Format data vectors into D x N shape
mu08_y1 = np.vstack([mu0_y1,mu8_y1])
mu08_y2 = np.vstack([mu0_y2,mu8_y2])
mu08_y12 = np.hstack((mu08_y1,mu08_y2))

# %% Get KDE objects, fit on each matrix
kde10 = scipy.stats.gaussian_kde(mu08_y1)
kde20 = scipy.stats.gaussian_kde(mu08_y2)

# get mins, maxes for each dataset
x_both_min, y_both_min = mu08_y12[0,:].min(), mu08_y12[1,:].min()
x_both_max, y_both_max = mu08_y12[0,:].max(), mu08_y12[1,:].max()

# Evaluate kde on a grid
xi, yi = np.mgrid[x_both_min:x_both_max:100j, y_both_min:y_both_max:100j]
coords = np.vstack([item.ravel() for item in [xi, yi]]) 
density_y1 = kde10(coords).reshape(xi.shape)
density_y2 = kde20(coords).reshape(xi.shape)

density_y1_pts = kde10(mu08_y1)
density_y2_pts = kde20(mu08_y2)

# normalize these to get probabilities
d_y1_norm = density_y1/np.sum(density_y1)
d_y2_norm = density_y2/np.sum(density_y2)

d_y1_norm_pts = density_y1_pts/np.sum(density_y1_pts)
d_y2_norm_pts = density_y2_pts/np.sum(density_y2_pts)

#%% Define confidence interval calculation
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

# %% Plot actual data
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = mu08_y1[0],
    y = mu08_y1[1],
    mode="markers",
    marker=dict(
        size=.5,
        opacity=.5
        ),
    name = "10"
))
fig.add_trace(go.Scatter(
    x = mu08_y2[0],
    y = mu08_y2[1],
    mode="markers",
    marker=dict(
        size=.5,
        opacity=.5
        ),
    name = "20"
))
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.update_layout(
    legend=dict(title="Treadmill speed:"),
    title="Bulk EMG During Locomotion, 2-Muscle State Space",
    xaxis=dict(title=dict(text="Gastrocnemius")),
    yaxis=dict(title=dict(text="Vastus Intermedius")),
    width=600,
    height=500
)
fig.show()
# %%
fig10 = px.imshow(d_y1_norm.T,title="2D PDF, Speed=10",width=500,height=500,origin='lower')
fig20 = px.imshow(d_y2_norm.T,title="2D PDF, Speed=20",width=500,height=500,origin='lower')
fig10.show(); fig20.show()
# %%
CI_10 = get_confidence(d_y1_norm,.95)
CI_20 = get_confidence(d_y2_norm,.95)
# %% plot Computed CI's for each dataset
figCI10 = px.imshow(CI_10.T,title="Speed 10, 95% CI",width=500,height=500,origin='lower')
figCI20 = px.imshow(CI_20.T,title="Speed 20, 95% CI",width=500,height=500,origin='lower')
figCI10.show(); figCI20.show()
# %%
O_10in20 = np.sum(CI_20*d_y1_norm)
O_20in10 = np.sum(CI_10*d_y2_norm)

fig_O_10in20 = px.imshow((CI_20*d_y1_norm).T,title="S10 density inside 95%CI of S20: "+str(np.round(O_10in20,decimals=4)),width=500,height=500,origin='lower')
fig_O_20in10 = px.imshow((CI_10*d_y2_norm).T,title="S20 density inside 95%CI of S10: "+str(np.round(O_20in10,decimals=4)),width=500,height=500,origin='lower')
fig_O_10in20.show(); fig_O_20in10.show()
# %% 
