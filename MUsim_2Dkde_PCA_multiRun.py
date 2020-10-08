# %% IMPORT packages
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
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

# start pool and define function for multiprocessing KDE fits
pool = Pool(processes=2)
def kde_crunch(kde_obj,coords_in,grid_shape):
    density = kde_obj(coords_in).reshape(grid_shape)
    return density


# %% SIMULATE MOTOR UNIT RESPONSES TO SESSION 1 AND SESSION 2
#############################################################################################
# Define Simulation Parameters 
num_sessions_to_simulate = 5
num_trials_to_simulate = 20
num_units_to_simulate = 10
noise_level = 0
gaussian_bw = 40   # choose smoothing bandwidth
explore_vals = range(2,11,2)    # this allows you to test a range of variables, 
vals_iter = iter(explore_vals)  # when calling next(vals_iter) in each loop
maxforce1 = 5; maxforce2 = vals_iter  # choose max force to analyze, default is 5
# want to shuffle the second session's thresholds?
# if not, set False below
shuffle_second_MU_thresholds=False
plot_results = False
overlap_results = []
##############################################################################################
# %% SIMULATE MULTIPLE RUNS WITHOUT PLOTTING
while len(overlap_results)<num_sessions_to_simulate:
    #############################################################################################
    # RUN 2 DIFFERENT SESSIONS
    mu = MUsim()                            # INSTANTIATE SIMULATION OBJECT
    mu.num_units = num_units_to_simulate    # SET NUMBER OF UNITS TO SIMULATE
    mu.num_trials = num_trials_to_simulate  # SET NUMBER OF TRIALS TO SIMULATE
    mu.session_noise_level = noise_level    # SET NOISE LEVEL FOR SESSION
    units = mu.recruit(MUmode='static')     # RECRUIT
    # FIRST SESSION
    force_profile = maxforce1/mu.init_force_profile.max()*mu.force_profile  # SCALE DEFAULT FORCE
    mu.apply_new_force(force_profile)       # SET SCALED LINEAR FORCE PROFILE
    session1 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
    session1_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 1
    # SECOND SESSION
    mu.reset_force                          # RESET FORCE BACK TO DEFAULT
    force_profile = next(maxforce2)/mu.init_force_profile.max()*mu.force_profile  # SCALE DEFAULT FORCE
    mu.apply_new_force(force_profile)       # SET SCALED LINEAR FORCE PROFILE
    session2 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
    session2_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 2
    #############################################################################################
    
    # COMPUTE PCA OBJECT on all MU data
    session1_smooth_stack = np.hstack(session1_smooth)
    session2_smooth_stack = np.hstack(session2_smooth)
    if shuffle_second_MU_thresholds is True:
        np.random.shuffle(session2_smooth_stack) #shuffle in place
    session12_smooth_stack = np.hstack((session1_smooth_stack,session2_smooth_stack)).T

    # standardize all unit activities
    scaler = StandardScaler()
    session12_smooth_stack = StandardScaler().fit_transform(session12_smooth_stack)

    # run for only 2 components, that capture most variance
    num_comp_proj = 2
    pca = PCA(n_components=num_comp_proj)
    fit = pca.fit(session12_smooth_stack)

    # PROJECT FROM FIT
    proj12_session1 = pca.transform(session1_smooth_stack.T)
    proj12_session2 = pca.transform(session2_smooth_stack.T)
    print('pca done.')
    # COMPUTE TRIAL AVERAGES
    # reshape into trials
    proj12_session1_trials = proj12_session1.T.reshape((len(force_profile),num_comp_proj,num_trials_to_simulate))
    proj12_session2_trials = proj12_session2.T.reshape((len(force_profile),num_comp_proj,num_trials_to_simulate))

    # get condition-averages for each
    proj12_session1_ave_x = proj12_session1[:,0].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
    proj12_session1_ave_y = proj12_session1[:,1].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
    proj12_session2_ave_x = proj12_session2[:,0].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)
    proj12_session2_ave_y = proj12_session2[:,1].reshape((len(force_profile),num_trials_to_simulate)).mean(axis=1)

    # Format data vectors into D x N shape
    proj12_session12 = np.vstack((proj12_session1,proj12_session2))

    # get mins, maxes for both datasets
    x_both_min, y_both_min = proj12_session12[:,0].min(), proj12_session12[:,1].min()
    x_both_max, y_both_max = proj12_session12[:,0].max(), proj12_session12[:,1].max()

    # GET KDE OBJECTS, fit on each matrix
    kde1 = gaussian_kde(proj12_session1.T)
    kde2 = gaussian_kde(proj12_session2.T)

    # Evaluate kde on a grid
    grid_margin = 20 # percent
    gm = grid_margin/100 # grid margin value to extend grid beyond all edges
    xi, yi = np.mgrid[(x_both_min-gm):(x_both_max+gm):100j, (y_both_min-gm):(y_both_max+gm):100j]
    coords = np.vstack([item.ravel() for item in [xi, yi]])
    density_session1 = kde1(coords).reshape(xi.shape)
    density_session2 = kde2(coords).reshape(xi.shape)

    # fit each KDE object as a separate process, for ~double speed
    kde_fix = partial(kde_crunch,coords_in=coords,grid_shape=xi.shape) # set constants
    kde_objs_list = [kde1,kde2]
    kde_out_list = pool.map(kde_fix, kde_objs_list)
    density_y1 = kde_out_list[0]
    density_y2 = kde_out_list[1]

    # density_session1_pts = kde1(proj12_session1.T)
    # density_session2_pts = kde2(proj12_session2.T)
    print('kde done.')

    # normalize these to get probabilities
    d_session1_norm = density_session1/np.sum(density_session1)
    d_session2_norm = density_session2/np.sum(density_session2)

    # d_session1_norm_pts = density_session1_pts/np.sum(density_session1_pts)
    # d_session2_norm_pts = density_session2_pts/np.sum(density_session2_pts)

    CI_10 = get_confidence(d_session1_norm,.95)
    CI_20 = get_confidence(d_session2_norm,.95)

    O_1in2 = np.sum(CI_20*d_session1_norm)
    O_2in1 = np.sum(CI_10*d_session2_norm)
    
    overlap_results.append(np.hstack((O_1in2,O_2in1)))
    print("loop: "+str(len(overlap_results))+". Results are: "+str(overlap_results[-1]))

pool.close()
overlap_results = np.vstack(overlap_results)

 # just show final results
if plot_results is True:
    px.line(overlap_results)
########################################################################################
# %% saving results:
# f = open("overlap_session1-y3-stat-noshuff_save0.txt", "w")
# f = open("overlap_session1-y3-dyn-noshuff_save0.txt", "w")
# f = open("overlap_session1-y3-stat-shuff_save0.txt", "w")
f = open("test_file.txt","w")
f.write("#           OneInTwo            TwoInOne\n") # column names
np.savetxt(f, overlap_results)
f.close()
# %% loading results:
# stat_noshuf = np.loadtxt("overlap_session1-y3-stat-noshuff_save0.txt")
# dyn_noshuf = np.loadtxt("overlap_session1-y3-dyn-noshuff_save0.txt")
# stat_shuf = np.loadtxt("overlap_session1-y3-stat-shuff_save0.txt")
# %%
