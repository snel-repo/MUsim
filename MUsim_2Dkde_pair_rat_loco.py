# %% IMPORT packages
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import plotly.io as pio
import colorlover as cl
from plotly.offline import iplot
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
# num_trials_to_simulate = 20
# num_units_to_simulate = 10
gaussian_bw = 10                # choose smoothing bandwidth
unit1 = 0; unit2 = 1           # choose units to analyze
treadmill_speed1 = '20'; treadmill_speed2 = '20'
# dogerat
session_date = '220603'
rat_name = 'dogerat'
treadmill_incline1 = '00'; treadmill_incline2 = '10'
# cleopatra
# session_date = '220715'
# rat_name = 'cleopatra'
# treadmill_incline1 = '00'; treadmill_incline2 = '10'

general_session_info = f"{session_date}_{rat_name}" 
session1_parameters = f"{session_date}_{rat_name}_speed{treadmill_speed1}_incline{treadmill_incline1}"
session2_parameters = f"{session_date}_{rat_name}_speed{treadmill_speed2}_incline{treadmill_incline2}"
#############################################################################################
# RUN 2 DIFFERENT SESSIONS
mu = MUsim()                            # INSTANTIATE SIMULATION OBJECT
# mu.num_units = num_units_to_simulate    # SET NUMBER OF UNITS TO SIMULATE
# mu.num_trials = num_trials_to_simulate  # SET NUMBER OF TRIALS TO SIMULATE
# units = mu.sample_MUs(MUmode='static')  # SAMPLE MUs
# FIRST SESSION
mu.load_MUs('../rat-loco/'+f'{session_date}_{rat_name}_speed{treadmill_speed1}_incline{treadmill_incline1}_time.npy',bin_width=1)
# session1 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
session1_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 1
# SECOND SESSION
mu.load_MUs('../rat-loco/'+f'{session_date}_{rat_name}_speed{treadmill_speed2}_incline{treadmill_incline2}_time.npy',bin_width=1)
# session2 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
session2_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 2
#############################################################################################
# %% COMPUTE UNIT DATA MATRICES
# Get 2 aligned channels of data
session1_smooth_stack = np.hstack(session1_smooth)
session2_smooth_stack = np.hstack(session2_smooth)

mu1_session1 = session1_smooth_stack[unit1,:]
mu2_session1 = session1_smooth_stack[unit2,:]
mu1_session2 = session2_smooth_stack[unit1,:]
mu2_session2 = session2_smooth_stack[unit2,:]

# get condition-averages for each
mu1_session1_ave = mu1_session1.reshape((mu.session[0].shape[0],mu.session_num_trials[0])).mean(axis=1)
mu2_session1_ave = mu2_session1.reshape((mu.session[0].shape[0],mu.session_num_trials[0])).mean(axis=1)
mu1_session2_ave = mu1_session2.reshape((mu.session[1].shape[0],mu.session_num_trials[1])).mean(axis=1)
mu2_session2_ave = mu2_session2.reshape((mu.session[1].shape[0],mu.session_num_trials[1])).mean(axis=1)

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
x_range = x_both_max-x_both_min
y_range = y_both_max-y_both_min
grid_margin = 20 # percent
gm = grid_margin/100 # grid margin value to extend grid beyond all edges
xi, yi = np.mgrid[(x_both_min-gm*x_range):(x_both_max+gm*x_range):1000j, (y_both_min-gm*y_range):(y_both_max+gm*y_range):1000j]
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

# %% PLOT MOTOR UNIT DATA
pio.templates.default = 'plotly_white' 
N_colors = 24#len(MU_spike_amplitudes_list)*len(ephys_channel_idxs_list)+len(bodyparts_list)
CH_colors = cl.to_rgb(cl.interp(cl.scales['6']['seq']['Greys'],N_colors))[-1:-N_colors:-1] # black to grey, 16
MU_colors = cl.to_rgb(cl.interp(cl.scales['10']['div']['Spectral'],N_colors)) # rainbow scale, 32
# rotate or reverse colors palettes, if needed
from collections import deque
color_list_len = len(MU_colors)
MU_colors_deque = deque(MU_colors)
MU_colors_deque.rotate(0)
MU_colors = list(MU_colors_deque)
MU_colors.reverse()

fig = go.Figure()
# data session 1
fig.add_trace(go.Scatter(
    x = mu12_session1[0],
    y = mu12_session1[1],
    mode="markers",
    marker=dict(
        size=3,
        opacity=.2,
        color='green'
        ),
    name = "Incline"+str(treadmill_incline1)
))
# data session 2
fig.add_trace(go.Scatter(
    x = mu12_session2[0],
    y = mu12_session2[1],
    mode="markers",
    marker=dict(
        size=3,
        opacity=.15,
        color=MU_colors[22]
        ),
    name = "Incline"+str(treadmill_incline2)
))
# trial average session 2
fig.add_trace(go.Scatter(
    x = mu1_session2_ave,
    y = mu2_session2_ave,
    mode="lines",
    opacity=.8,
    line=dict(
        width=5,
        color=MU_colors[22]
        ),
    name = "Incline"+str(treadmill_incline2)+" mean"
))
# trial average session 1
fig.add_trace(go.Scatter(
    x = mu1_session1_ave,
    y = mu2_session1_ave,
    mode="lines",
    opacity=.8,
    line=dict(
        width=5,
        color='green'
        ),
    name = "Incline"+str(treadmill_incline1)+" mean"
))

# fig.add_trace(go.Scatter(x=[x_both_min, x_both_max],
#              y=[y_both_min, y_both_max],
#              mode='markers',
#              marker=dict(
#                  size=(y_both_max-y_both_min)/100, 
#                  color=[y_both_min, y_both_max], 
#                  colorscale=MU_colors, 
#                  colorbar=dict(thickness=10), 
#                  showscale=True
#              ),
#              hoverinfo='none'
#             ))

fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_layout(
    legend=dict(title="<b>Incline Conditions:</b>"),
    title=f"<b>Motor Unit Trajectories in 2-Unit State Space</b><br><sup>Session Info: <b>{general_session_info}</b></sup>",
    xaxis=dict(title=dict(text="<b>Low-Threshold Unit Activity</b>")),
    yaxis=dict(title=dict(text="<b>High-Threshold Unit Activity</b>")),
    width=500,
    height=500,
)
iplot(fig)
# %%
fig10 = px.imshow(d_session1_norm.T,title="2D PDF, incline"+str(treadmill_incline1),width=500,height=500,origin='lower')
fig20 = px.imshow(d_session2_norm.T,title="2D PDF, incline"+str(treadmill_incline2),width=500,height=500,origin='lower')
iplot(fig10); iplot(fig20)
# %%
CI_1 = get_confidence(d_session1_norm,.95)
CI_2 = get_confidence(d_session2_norm,.95)
OVL = np.minimum(d_session1_norm,d_session2_norm)
OVL_norm = OVL/OVL.sum()
CI_OVL = get_confidence(OVL_norm,.95)
# %% plot Computed CI's for each dataset
figCI1 = px.imshow(CI_1.T,title="<b>Incline"+str(treadmill_incline1)+", 95%CI</b>",width=500,height=500,origin='lower')
figCI2 = px.imshow(CI_2.T,title="<b>Incline"+str(treadmill_incline2)+", 95%CI</b>",width=500,height=500,origin='lower')
figCI_OVL = px.imshow(CI_OVL.T,title="<b>95% Confidence Interval of OVL</b><br><sup>between incline "+str(treadmill_incline1)+" and "+str(treadmill_incline2),width=500,height=500,origin='lower')
iplot(figCI1); iplot(figCI2); iplot(figCI_OVL)
# %%
fig_OVL = px.imshow(OVL.T,title="<b>Overlap of Trajectory Distributions: OVL="+str(np.round(OVL.sum(),decimals=4))+"</b><br><sup>For inclines "+str(treadmill_incline1)+' and '+str(treadmill_incline2)+"</sup>",width=500,height=500,origin='lower')
iplot(fig_OVL)
# # %%
# O_10in20 = np.sum(CI_20*d_session1_norm)
# O_20in10 = np.sum(CI_10*d_session2_norm)

# fig_O_10in20 = px.imshow((CI_20*d_session1_norm).T,title="<b>95% Confidence Interval Overlap: "+str(np.round(O_10in20,decimals=4))+"</b><br><sup>For incline"+str(treadmill_incline1)+' within incline'+str(treadmill_incline2)+"</sup>",width=500,height=500,origin='lower')
# fig_O_20in10 = px.imshow((CI_10*d_session2_norm).T,title="<b>95% Confidence Interval Overlap: "+str(np.round(O_20in10,decimals=4))+"</b><br><sup>For incline"+str(treadmill_incline2)+' within incline'+str(treadmill_incline1)+"</sup>",width=500,height=500,origin='lower')
# iplot(fig_O_10in20); iplot(fig_O_20in10)
# # %% 

# %%
