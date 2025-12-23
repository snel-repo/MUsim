class for motor unit simulations, produce firing rates according to arbitrary force profiles; operates on size principle when in "static" mode; "dynamic" mode applies unit MU threshold reversals when dF/dt exceeds a threshold value; shuffled thresholds can also be simulated in "MUsim_2Dkde_pair.py", "MUsim_2Dkde_PCA.py", "MUsim_2Dkde_PCA_multiRun.py".

Configure an environment with miniconda or micromamba with `conda create --name mu --file requirements.txt`, or `micromamba env create -f environment.yaml`. You must first install one of these, or use your favorite environment manager.

To be able to run your own simulations directly, switch to the `rat_simulation` or `monkey_simulation` branches with either of the below commands:
    
    git checkout rat_simulation

or

    git checkout monkey_simulation

You can find usages of the MUsim class in `generate_simulated_dataset_RAT.py`, particularly in the `sample_MUsim_obj` and `batch_run_MUsim` functions. This `generate_simulated_dataset_RAT.py` file allows you to generate the exact same spike times with the exact same waveform shapes as in the simulated rat dataset used for all comparisons in the EMUsort paper. Due to non-deterministic operations in the Gaussian noise generation method we used with PyTorch, the continuous.dat file output will not be exactly the same, so we provide that as an output file in `output_files/continuous_dat_files`. The spike times and sample numbers files as used in the paper are also provided in `output_files/spikes_npy_files` and `output_files/sample_numbers_npy_files`, respectively. 

If you run the code, new files will be generated into the proper folders within the `output_files` folder, and you can check for an identical match between the spikes files using `sha256sum` or similar commands.