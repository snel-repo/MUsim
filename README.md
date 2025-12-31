# `MUsim`

### A Python class for arbitrary motor unit simulations

This class can be used to produce firing rates for any number of simulated motor units according to arbitrary force or kinematic profiles which act as "drive" signals for the simulated motor unit population. The default behavior of the simulator is based on [Fuglevand et al., 1993b](https://doi.org/10.1152/jn.1993.70.6.2470).

To try the class, you must first install the dependencies. This can be done by configuring an environment with [`miniconda`](https://www.anaconda.com/docs/getting-started/miniconda/main) or [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) with `conda create --name MUsim --file requirements.txt`, or `micromamba env create -f environment.yaml`. You must first install one of these, or use your favorite environment manager.

This class was used to produce the simulations for the main results in the EMUsort paper. To be able to run these simulations directly on your own system, you can download the proper data and generation scripts by switching to either the `rat-simulation` branch or the `monkey-simulation` branch with either of the below commands in each section below:
    

## Rat Simulation

    git checkout rat-simulation

You can find example usages of the `MUsim` class in `generate_simulated_dataset_RAT.py`, particularly in the `sample_MUsim_obj` and `batch_run_MUsim` functions. This `generate_simulated_dataset_RAT.py` file allows you to generate the exact same spike times with the exact same waveform shapes as in the simulated rat dataset used for all comparisons in the EMUsort paper. Due to non-deterministic operations in the Gaussian noise generation method we used with PyTorch, the `continuous*.dat` file output will not be exactly the same on your system, so we provide that on Internet Archive (see below). The continuous data, spike times and cluster IDs, and sample numbers files used in the EMUsort paper are all provided there in a ZIP archive under `output_files/continuous_dat_files`, `output_files/spikes_npy_files` and `output_files/sample_numbers_npy_files`, respectively. 

When you run `generate_simulated_dataset_RAT.py`, new files will be generated into the proper subfolders of `output_files`. Afterwards, you can check for identical matches between the spikes files you generated and the ones provided in the `output_files` folder using `sha256sum` (or similar commands). You may check the provided `checksum_spikes_npy_RAT.sha256` file to see what the hash should be for the full-sized spikes array. We only provide a checksum file for the large, time-stamped `.npy` spikes array, but if desired, you could compute the checksum for the `spike_clusters.npy` and `spike_times.npy` files we provide at `output_files/spikes_npy_files` and compare those to the checksums of the corresponding files you've generated.

#### Full Rat Simulation Outputs Available on Internet Archive
If you wish, you may also download a file archive containing the full simulation outputs that were used in the EMUsort paper from our Internet Archive listing with the unique identifier: `MUsim-rat-simulation` (see [HERE](https://archive.org/details/MUsim-rat-simulation)). The ZIP archive may also be downloaded directly using the following command on Linux systems:

    wget https://archive.org/download/MUsim-rat-simulation/rat-simulation_output_files.zip


## Monkey Simulation

    git checkout monkey-simulation

You can find example usages of the `MUsim` class in `generate_simulated_dataset_MONKEY.py`, particularly in the `sample_MUsim_obj` and `batch_run_MUsim` functions. This `generate_simulated_dataset_MONKEY.py` file allows you to generate the exact same spike times with the exact same waveform shapes as in the simulated monkey dataset used for all comparisons in the EMUsort paper. Due to non-deterministic operations in the Gaussian noise generation method used with PyTorch, the `continuous*.dat` file output will not be exactly the same on your system, so we provide that along with all other outputs on Internet Archive (see below). The spike times and sample numbers files used in the EMUsort paper are also provided in `output_files/spikes_npy_files` and `output_files/sample_numbers_npy_files`, respectively. 

When you run `generate_simulated_dataset_MONKEY.py`, new files will be generated into the proper subfolders of `output_files`. Afterwards, you can check for identical matches between the spikes files you generated and the ones provided in the `output_files` folder using `sha256sum` (or similar commands). You may check the provided `checksum_spikes_npy_MONKEY.sha256` file to see what the hash should be for the full-sized spikes array. We only provide a checksum file for the large, time-stamped `.npy` spikes array, but if desired, you could compute the checksum for the `spike_clusters.npy` and `spike_times.npy` files we provide at `output_files/spikes_npy_files` and compare those to the checksums of the corresponding files you've generated.

#### Full Monkey Simulation Outputs Available on Internet Archive
If you wish, you may also download a file archive containing the full simulation outputs that were used in the EMUsort paper from our Internet Archive listing with the unique identifier: `MUsim-monkey-simulation` (see [HERE](https://archive.org/details/MUsim-monkey-simulation)). The ZIP archive may also be downloaded directly using the following command on Linux systems:

    wget https://archive.org/download/MUsim-monkey-simulation/monkey-simulation_output_files.zip

## EMUsort Paper Supplements

All data used for computing Tables 1-2 and Figures 6A-B and 7A-B in the EMUsort publication are provided in the `EMUsort_paper_supplements` folder. The raw outputs used to create these spreadsheets were generated by the `plot4()` function in `compute_ground_truth_metrics.py` and are stored inside the `EMUsort_paper_supplements/raw_outputs` folder.