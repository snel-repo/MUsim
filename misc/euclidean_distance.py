import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from MUsim import MUsim

# params
session_date = ['20221116-3', '20221116-7', '20221116-5', '20221116-8', '20221116-9']
rat_name = ['godzilla','godzilla','godzilla','godzilla','godzilla']
treadmill_speed = ['05','05','10','10','10']
treadmill_incline = ['00','00','00','00','00']
combo_params = zip(session_date, rat_name, treadmill_speed, treadmill_incline)
# for i in combo_params:
#     print(i)
    
# Create list of sessions to be compared
session_list = [
    f"/home/tony/git/rat-loco/{sess}_{rat}_speed{spd}_incline{inc}_phase.npy" for (sess,rat,spd,inc) in combo_params
    ]
# session_list = [
#     '/home/tony/git/rat-loco/20221116-3_godzilla_speed05_incline00_phase.npy',
#     '/home/tony/git/rat-loco/20221116-5_godzilla_speed10_incline00_phase.npy',
#     '/home/tony/git/rat-loco/20221116-5_godzilla_speed10_incline00_phase.npy',
#     '/home/tony/git/rat-loco/20221116-5_godzilla_speed10_incline00_phase.npy',
#     '/home/tony/git/rat-loco/20221116-5_godzilla_speed10_incline00_phase.npy',
# ]

mu = MUsim()
for ((iSess1, iSess2),(iSpeed1, iSpeed2)) in zip(combinations(session_list, 2),combinations(treadmill_speed, 2)):
    mu.load_MUs(iSess1, bin_width=2)
    session1_smooth = mu.convolve(sigma = 10, target="session") 
    # mu.load_MUs('/home/tony/git/rat-loco/20221116-5_godzilla_speed10_incline00_phase.npy', bin_width=2)
    mu.load_MUs(iSess2, bin_width=2)
    session2_smooth = mu.convolve(sigma = 10, target="session") 

    session1_smooth = np.transpose(session1_smooth, (2,1,0))
    session2_smooth = np.transpose(session2_smooth, (2,1,0))

    # Get the number of trajectories in each dataset
    num_trajectories_session1 = session1_smooth.shape[0]
    num_trajectories_session2 = session2_smooth.shape[0]
    num_trajectories_product = num_trajectories_session1 * num_trajectories_session2

    # Get length of all trajectories
    len_trajectories_session1 = session1_smooth.shape[2]
    len_trajectories_session2 = session2_smooth.shape[2]
    assert len_trajectories_session1 == len_trajectories_session2, "Length of arrays should be the same!"
    len_trajectories_session = len_trajectories_session1

    # Get length of all trajectories
    num_units_session1 = session1_smooth.shape[1]
    num_units_session2 = session2_smooth.shape[1]
    assert num_units_session1 == num_units_session2, "Length of arrays should be the same!"
    num_units_session = num_units_session1

    # Create a matrix to store the results
    Euclid_pair = np.zeros((num_trajectories_session1, num_trajectories_session2))

    all_distances = np.zeros((num_trajectories_product, len_trajectories_session, num_units_session))
    # Loop over all pairs of trajectories
    loop_counter = 0
    for i in range(num_trajectories_session1):
        for j in range(num_trajectories_session2):
            # for iPoint in range(len_trajectories_session):
                # Calculate the Euclidean distance between the two trajectories
            pointwise_distance = session1_smooth[i] - session2_smooth[j]
            all_distances[loop_counter, :, :] = pointwise_distance.T
            loop_counter += 1

    # Euclid_pair = all_distances.sum(-1).sum(-1)
    distance_metric = np.linalg.norm(all_distances,axis=2).sum(0)
    print((iSess1, iSess2))
    
    # Print the results
    print(distance_metric.sum())
    if iSpeed1==iSpeed2 and iSpeed1=='05':
        color = 'blue'
    elif iSpeed1==iSpeed2 and iSpeed2=='10':
        color = 'red'
    else:
        color = 'orange'
    plt.plot(distance_metric, c=color)

speeds = []
for (speed1, speed2) in combinations(treadmill_speed, 2):
    speeds.append((speed1, speed2))

plt.legend(speeds)

plt.show()

#print(session1_smooth[i] - session2_smooth[j])
# print(session1_smooth[i])

# Get the number of points in each dataset
#num_points_session1 = session1_smooth.shape[0] * session1_smooth.shape[1]
#num_points_session2 = session2_smooth.shape[0] * session2_smooth.shape[1]

# Reshape the datasets into 2D arrays with shape (num_points, 3)
#session1_smooth_2d = session1_smooth.reshape((num_points_session1, 3))
#session2_smooth_2d = session2_smooth.reshape((num_points_session2, 3))

# Calculate the Euclidean distance between all pairs of points
#Euclid_all = 0.0
# for i in range(num_points_session1):
#     for j in range(num_points_session2):
#         # Calculate the Euclidean distance between the two points
#         distance = np.linalg.norm(session1_smooth_2d[i] - session2_smooth_2d[j])
#         # Add the distance to the sum
#         Euclid_all += distance

# # Print the sum of distances
# print(Euclid_all)





