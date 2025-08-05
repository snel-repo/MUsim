import h5py
import numpy as np

filename = "/snel/share/data/rodent-ephys/open-ephys/treadmill/sean-pipeline/godzilla/paper_evals/CHs_8_MUs_10/sim_noise_0.2_orig_CHs/Record Node 101/experiment1/recording1/structure.oebin_decomp_1_skew.mat"


input_data = []

spike_data = []


with h5py.File(filename, "r") as file:

    print("Keys: ", list(file.keys()))

    signal = file["signal"]

    print("Signal Subfields: ", list(signal.keys()))

    signal_dischargetimes = signal["Dischargetimes"][:]

    print()

    input_data = signal_dischargetimes

    print("------------------------------")

    print(type(input_data))
    print(input_data)

    for cluster in range(0, len(input_data)):

        cluster_ref = input_data[cluster][0]
        cluster_data = file[cluster_ref][:]

        for spike in cluster_data:

            spike = int(spike[0])
            spike_data.append((spike, (cluster + 1)))

    print(len(spike_data))

    spike_data.sort(key=lambda x: x[0])

    spike_times = []
    spike_clusters = []

    for entry in spike_data:
        spike_times.append(entry[0])
        spike_clusters.append(entry[1])

    # lets convert them into numpy arrays

    spike_times = np.array(spike_times)
    spike_clusters = np.array(spike_clusters)

    # lets save them to a file

    np.save("./MUedit/run4/spike_times.npy", spike_times)
    np.save("./MUedit/run4/spike_clusters.npy", spike_clusters)

    # cluster_1_ref = input_data[0][0]

    # cluster_1_data = file[cluster_1_ref][:]

    # print()

    # print(len(cluster_1_data))
    # print(cluster_1_data[0:20])

    # print(type(cluster_1_data[0][0]))
    # print(cluster_1_data[0][0])
