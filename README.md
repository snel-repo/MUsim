# `MUsim`

### A Python class for arbitrary motor unit simulations

This class can be used to produce firing rates for any number of simulated motor units according to arbitrary force or kinematic profiles which act as "drive" signals for the simulated motor unit population. The default behavior of the simulator is based on [Fuglevand et al., 1993b](https://doi.org/10.1152/jn.1993.70.6.2470).

To try the class, you must first install the dependencies. This can be done by configuring an environment with [`miniconda`](https://www.anaconda.com/docs/getting-started/miniconda/main) or [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) with `conda create --name MUsim --file requirements.txt`, or `micromamba env create -f environment.yaml`. You must first install one of these, or use your favorite environment manager.

This class was used to produce the simulations for testing in the EMUsort publication. To be able to run your own simulations directly, you can download the proper data and generation script by switching to either the `rat-simulation` branch or the `monkey-simulation` branch with either of the below commands:
    

## Rat Simulation

    git checkout rat-simulation

You can find example usages of the MUsim class in `generate_simulated_dataset_RAT.py`, particularly in the `sample_MUsim_obj` and `batch_run_MUsim` functions. This `generate_simulated_dataset_RAT.py` file allows you to generate the exact same spike times with the exact same waveform shapes as in the simulated rat dataset used for all comparisons in the EMUsort paper. Due to non-deterministic operations in the Gaussian noise generation method we used with PyTorch, the continuous.dat file output will not be exactly the same, so we provide that as an output file in `output_files/continuous_dat_files`. The spike times and sample numbers files as used in the paper are also provided in `output_files/spikes_npy_files` and `output_files/sample_numbers_npy_files`, respectively. 

If you run `generate_simulated_dataset_RAT.py`, new files will be generated into the proper folders in `output_files`, and you can check for an identical match between the spikes files you generate using `sha256sum` or similar commands and check against the provided `checksum_spikes_npy_RAT.sha256` file. We only provide a checksum file for the large `.npy` spikes file, but if desired, you could compute the checksum for the `spike_clusters.npy` and `spike_times.npy` files we provide and compare those to the checksums of the corresponding files you've generated.

If you wish, you may also download a file archive containing the full simulation outputs that were used in the EMUsort paper from our Internet Archive listing with the unique identifier: `MUsim-rat-simulation` (see [HERE](https://archive.org/details/MUsim-rat-simulation)). The ZIP archive may also be downloaded directly using the following command on Linux systems:

    wget https://archive.org/download/MUsim-rat-simulation/rat-simulation_output_files.zip


## Monkey Simulation

    git checkout monkey-simulation

You can find example usages of the MUsim class in `generate_simulated_dataset_MONKEY.py`, particularly in the `sample_MUsim_obj` and `batch_run_MUsim` functions. This `generate_simulated_dataset_MONKEY.py` file allows you to generate the exact same spike times with the exact same waveform shapes as in the simulated rat dataset used for all comparisons in the EMUsort paper. Due to non-deterministic operations in the Gaussian noise generation method we used with PyTorch, the continuous.dat file output will not be exactly the same, so we provide that as an output file in `output_files/continuous_dat_files`. The spike times and sample numbers files as used in the paper are also provided in `output_files/spikes_npy_files` and `output_files/sample_numbers_npy_files`, respectively. 

If you run `generate_simulated_dataset_MONKEY.py`, new files will be generated into the proper folders in `output_files`, and you can check for an identical match between the spikes files you generate using `sha256sum` or similar commands and check against the provided `checksum_spikes_npy_MONKEY.sha256` file. We only provide a checksum file for the large `.npy` spikes file, but if desired, you could compute the checksum for the `spike_clusters.npy` and `spike_times.npy` files we provide and compare those to the checksums of the corresponding files you've generated.

If you wish, you may also download a file archive containing the full simulation outputs that were used in the EMUsort paper from our Internet Archive listing with the unique identifier: `MUsim-monkey-simulation` (see [HERE](https://archive.org/details/MUsim-monkey-simulation)). The ZIP archive may also be downloaded directly using the following command on Linux systems:

    wget https://archive.org/download/MUsim-monkey-simulation/monkey-simulation_output_files.zip