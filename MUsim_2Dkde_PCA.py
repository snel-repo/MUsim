# %% IMPORT packages
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
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

# %% SIMULATE MOTOR UNIT RESPONSES TO YANK 1 AND YANK 2
########################################################
# Define Simulation Parameters 
num_trials_to_simulate = 50
num_units_to_simulate = 10
gaussian_bw = 40            # choose smoothing bandwidth
yankval1 = 1; yankval2 = 3  # choose yank to analyze
########################################################
mu = MUsim()            # INSTANTIATE SIMULATION OBJECT
mu.num_units = num_units_to_simulate
units = mu.recruit(MUmode='dynamic')    # RECRUIT
force_profile = yankval1*mu.force_profile # APPLY FORCE PROFILE (NON-DEFAULT)
mu.apply_new_force(force_profile)
session1 = mu.simulate_session(num_trials_to_simulate) # APPLY DEFAULT FORCE PROFILE
yank1 = mu.convolve(gaussian_bw, target="session") # SMOOTH SPIKES FOR SESSION 1

mu.reset_force() # RESET FORCE BACK TO DEFAULT
force_profile = yankval2*mu.force_profile # APPLY FORCE PROFILE (NON-DEFAULT)
mu.apply_new_force(force_profile)
session2 = mu.simulate_session(num_trials_to_simulate)
yank2 = mu.convolve(gaussian_bw, target="session") # SMOOTH SPIKES FOR SESSION 2

# %% COMPUTE PCA OBJECT on all MU data
yank1_stack = np.hstack(yank1)
yank2_stack = np.hstack(yank2)
yank12_stack = np.hstack((yank1_stack,yank2_stack)).T

# run for all components to see VAF
num_comp_test = 10
pca = PCA(n_components=num_comp_test)
fit = pca.fit(yank12_stack)
print("explained variance: "+str(fit.explained_variance_ratio_))
plt.scatter(range(num_comp_test),fit.explained_variance_ratio_)
plt.plot(np.cumsum(fit.explained_variance_ratio_),c='darkorange')
plt.hlines(0.7,0,num_comp_test,colors='k',linestyles="dashed")
plt.title("explained variance for each additional unit")
plt.legend(["cumulative","individual"])
plt.xlabel("units")

# run for only 2 components, that capture most variance
num_comp_proj = 2
pca = PCA(n_components=num_comp_proj)
fit = pca.fit(yank12_stack)

# %% PROJECT FROM FIT
proj12_y1 = pca.transform(yank1_stack.T)
proj12_y2 = pca.transform(yank2_stack.T)
# %% COMPUTE TRIAL AVERAGES
# reshape into trials
proj12_y1_trials = proj12_y1.T.reshape((len(force_profile),num_comp_proj,num_trials_to_simulate))
proj12_y2_trials = proj12_y2.T.reshape((len(force_profile),num_comp_proj,num_trials_to_simulate))

# get condition-averaged traces for each
proj12_y1_ave_x = proj12_y1[:,0].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
proj12_y1_ave_y = proj12_y1[:,1].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
proj12_y2_ave_x = proj12_y2[:,0].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
proj12_y2_ave_y = proj12_y2[:,1].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)

## Format data vectors into D x N shape
proj12_y12 = np.hstack((proj12_y1,proj12_y2))

# get mins, maxes for both datasets
x_both_min, y_both_min = proj12_y12[(0,2),:].min(), proj12_y12[(1,3),:].min()
x_both_max, y_both_max = proj12_y12[(0,2),:].max(), proj12_y12[(1,3),:].max()

# %% GET KDE OBJECTS, fit on each matrix
kde10 = scipy.stats.gaussian_kde(proj12_y1.T)
kde20 = scipy.stats.gaussian_kde(proj12_y2.T)

# Evaluate kde on a grid
xi, yi = np.mgrid[x_both_min:x_both_max:100j, y_both_min:y_both_max:100j]
coords = np.vstack([item.ravel() for item in [xi, yi]]) 
density_y1 = kde10(coords).reshape(xi.shape)
density_y2 = kde20(coords).reshape(xi.shape)

density_y1_pts = kde10(proj12_y1.T)
density_y2_pts = kde20(proj12_y2.T)

# normalize these to get probabilities
d_y1_norm = density_y1/np.sum(density_y1)
d_y2_norm = density_y2/np.sum(density_y2)

d_y1_norm_pts = density_y1_pts/np.sum(density_y1_pts)
d_y2_norm_pts = density_y2_pts/np.sum(density_y2_pts)

# %% PLOT SIMULATED DATA
fig = go.Figure()
# data yank 2
fig.add_trace(go.Scatter(
    x = proj12_y2[:,0],
    y = proj12_y2[:,1],
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
    x = proj12_y1[:,0],
    y = proj12_y1[:,1],
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
    x = proj12_y2_ave_x,
    y = proj12_y2_ave_y,
    mode="lines",
    line=dict(
        width=1,
        color="blue"
        ),
    name = "yank="+str(yankval2)+" mean"
))
# trial average yank 1
fig.add_trace(go.Scatter(
    x = proj12_y1_ave_x,
    y = proj12_y1_ave_y,
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
    title="Simulated population trajectories in 2D PCA Space",
    xaxis=dict(title=dict(text="First PC")),
    yaxis=dict(title=dict(text="Second PC")),
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
