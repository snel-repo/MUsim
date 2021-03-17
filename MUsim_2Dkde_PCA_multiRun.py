# %% IMPORT packages
import numpy as np
from scipy.stats import gaussian_kde
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from MUsim import MUsim
from functools import partial

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
def kde_grid(kde_obj,coords_in,grid_shape):
    density = kde_obj(coords_in).reshape(grid_shape)
    return density
pool = Pool(processes=2)


# %% SIMULATE MOTOR UNIT RESPONSES TO SESSION 1 AND SESSION 2
#############################################################################################
# Define Simulation Parameters 
search_param = np.repeat([0,1],20) # set values to test a range of some chosen variable
search_iter = iter(search_param)  # when calling next(vals_iter) in each loop
num_sessions_to_simulate = 10
num_trials_to_simulate = 50
num_units_to_simulate = 32
trial_length = 500 # bins
noise_level = 0
max_firing_rate = 50
gaussian_bw = 40   # choose smoothing bandwidth
max_force1 = 5; max_force2 = 10  # choose max force to analyze, default is 5
# want to shuffle the second session's thresholds?
# if not, set False below
MUmode = 'dynamic'
MUreversal_frac = 1 # set fraction of MU population that will reverse
yank_flip_thresh = 15
MUreversal_static_units = list(range(num_units_to_simulate-10))
# del MUreversal_static_units[ ((len(MUreversal_static_units)//2)+1):((len(MUreversal_static_units)//2)+11)]
shuffle_second_MU_thresholds=False
plot_results = True
overlap_results = []
##############################################################################################
# %% SIMULATE MULTIPLE RUNS WITHOUT PLOTTING
mu = MUsim()                            # INSTANTIATE SIMULATION OBJECT
while len(overlap_results)<len(search_param):
    #############################################################################################
    # RUN 2 DIFFERENT SESSIONS
    mu.num_units = num_units_to_simulate    # SET NUMBER OF UNITS TO SIMULATE
    mu.num_trials = num_trials_to_simulate  # SET NUMBER OF TRIALS TO SIMULATE
    mu.max_spike_prob = max_firing_rate/mu.num_bins_per_trial # SET SPIKING PROBABILITY
    mu.num_bins_per_trial = trial_length    # SET NUMBER OF BINS PER TRIAL
    mu.session_noise_level = noise_level    # SET NOISE LEVEL FOR SESSION
    mu.yank_flip_thresh = yank_flip_thresh
    mu.MUreversal_frac = next(search_iter) #MUreversal_frac # SET NUMBER OF UNITS THAT WILL FLIP THRESHOLD DYNAMICALLY
    mu.MUreversal_static_units = MUreversal_static_units # SET WHICH UNITS ARE FORCED TO BE STATIC DURING DYNAMIC SIMULATION, PROVIDE LIST OF IDXS
    units = mu.sample_MUs(MUmode='dynamic')  # SAMPLE MUs
    # FIRST SESSION
    force_profile1 = max_force1/mu.init_force_profile.max()*mu.init_force_profile  # SCALE DEFAULT FORCE
    force_profile1 = np.roll(force_profile1,100) # shift by 100ms
    force_profile1[:100] = 0 # set first 100ms to zero force
    mu.apply_new_force(force_profile1)       # SET SCALED LINEAR FORCE PROFILE
    session1 = mu.simulate_session()        # GENERATE SPIKE RESPONSES FOR EACH UNIT
    session1_smooth = mu.convolve(gaussian_bw, target="session")  # SMOOTH SPIKES FOR SESSION 1
    # SECOND SESSION
    mu.reset_force()                      # RESET FORCE BACK TO DEFAULT
    force_profile2 = max_force2/mu.init_force_profile.max()*mu.init_force_profile  # SCALE DEFAULT FORCE
    force_profile2 = np.roll(force_profile2,100) # shift by 100ms
    force_profile2[:100] = 0 # set first 100ms to zero force
    mu.apply_new_force(force_profile2)       # SET SCALED LINEAR FORCE PROFILE
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
    scaler = StandardScaler(with_std=False)
    session12_smooth_stack = scaler.fit_transform(session12_smooth_stack)

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
    proj12_session1_trials = proj12_session1.T.reshape((mu.num_bins_per_trial,num_comp_proj,num_trials_to_simulate))
    proj12_session2_trials = proj12_session2.T.reshape((mu.num_bins_per_trial,num_comp_proj,num_trials_to_simulate))

    # get condition-averages for each
    proj12_session1_ave_x = proj12_session1[:,0].reshape((mu.num_bins_per_trial,num_trials_to_simulate)).mean(axis=1)
    proj12_session1_ave_y = proj12_session1[:,1].reshape((mu.num_bins_per_trial,num_trials_to_simulate)).mean(axis=1)
    proj12_session2_ave_x = proj12_session2[:,0].reshape((mu.num_bins_per_trial,num_trials_to_simulate)).mean(axis=1)
    proj12_session2_ave_y = proj12_session2[:,1].reshape((mu.num_bins_per_trial,num_trials_to_simulate)).mean(axis=1)

    # Format data vectors into D x N shape
    proj12_session12 = np.vstack((proj12_session1,proj12_session2))

    # get mins, maxes for both datasets
    x_both_min, y_both_min = proj12_session12[:,0].min(), proj12_session12[:,1].min()
    x_both_max, y_both_max = proj12_session12[:,0].max(), proj12_session12[:,1].max()

    # DEFINE GRID FOR KDE
    x_range = x_both_max-x_both_min
    y_range = y_both_max-y_both_min
    grid_margin = 20 # percent
    gm = grid_margin/100 # grid margin value to extend grid beyond all edges
    xi, yi = np.mgrid[(x_both_min-gm*x_range):(x_both_max+gm*x_range):100j, (y_both_min-gm*y_range):(y_both_max+gm*y_range):100j]
    coords = np.vstack([item.ravel() for item in [xi, yi]]) 
    # grid_margin = 10 # percent
    # gm = grid_margin/100 # grid margin value to extend grid beyond all edges
    # xi, yi = np.mgrid[(x_both_min-gm):(x_both_max+gm):100j, (y_both_min-gm):(y_both_max+gm):100j]
    # coords = np.vstack([item.ravel() for item in [xi, yi]])

    # GET KDE OBJECTS, for each matrix
    proj_list = [proj12_session1.T,proj12_session2.T]
    kde_objs_list = pool.map(kde_fit,proj_list)

    # TRANSFORM DATA 
    kde_grid_fix = partial(kde_grid,coords_in=coords,grid_shape=xi.shape) # set constants
    kde_out_list = pool.map(kde_grid_fix, kde_objs_list)
    density_session1 = kde_out_list[0]
    density_session2 = kde_out_list[1]

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
    plt.plot(search_param,overlap_results)
    plt.title("overlap pair values across conditions")
    plt.legend(["1st dist in 2nd","2nd dist in 1st"],title="fraction of:")
    plt.ylim((0,1))
    plt.xlabel('search parameter values')
    plt.ylabel('fraction of overlap')
    plt.show()
    simulation_type_range = round(len(search_param)/2)
    plt.scatter(overlap_results[:simulation_type_range,1],overlap_results[:simulation_type_range,0],color='silver')
    plt.scatter(overlap_results[simulation_type_range:,1],overlap_results[simulation_type_range:,0],color='crimson')
    plt.title("overlap pair scatter")
    plt.legend(["size principle","10/32 Dynamic MUs,\nlarge threshold changes"])
    plt.xlabel("fraction of 1st dist in 2nd")
    plt.ylabel("fraction of 2nd dist in 1st")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()

means = overlap_results.mean(axis=0)
stds = overlap_results.std(axis=0)
print("means are: "+str(means))
print("std. devs. are: "+str(stds))
########################################################################################
# %% saving results:
print("writing to file.")
# f = open("overlap_session1-y3-stat-noshuff_save0.txt", "w")
# f = open("overlap_session1-y3-dyn-noshuff_save0.txt", "w")
# f = open("overlap_session1-y3-stat-shuff_save0.txt", "w")
f = open("MU_Reversal_Test_-10_unit_large.txt","w")
# f = open("test_file.txt","w")
f.write("#           OneInTwo            TwoInOne\n") # column names
np.savetxt(f, overlap_results)
f.close()
# %% loading results:
# stat_noshuf = np.loadtxt("overlap_session1-y3-stat-noshuff_save0.txt")
# dyn_noshuf = np.loadtxt("overlap_session1-y3-dyn-noshuff_save0.txt")
# stat_shuf = np.loadtxt("overlap_session1-y3-stat-shuff_save0.txt")

overlap_results = np.loadtxt("MU_Reversal_Test_-3_unit_large.txt")
# results40 = np.loadtxt("40Hz-5-15-dynamic.txt")
# %%