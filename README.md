class for motor unit simulations, produce firing rates according to arbitrary force profiles; operates on size principle when in "static" mode; "dynamic" mode applies unit MU threshold reversals when dF/dt exceeds a threshold value; shuffled thresholds can also be simulated in "MUsim_2Dkde_pair.py", "MUsim_2Dkde_PCA.py", "MUsim_2Dkde_PCA_multiRun.py".

Configure an environment with miniconda or micromamba with `conda create --name mu --file requirements.txt`, or `micromamba env create -f environment.yaml`. You must first install one of these, or use your favorite environment manager.

To be able to run your own simulations directly, switch to the `rat_simulation` or `monkey_simulation` branches with either of the below commands:
    
    git checkout rat_simulation

or

    git checkout monkey_simulation

You can also start by skimming through the MUsim class to understand how it's organized, and skip over the functions with leading underscores. You may also find the usages of "MUsim" in the `generate_synthetic_dataset*` files useful, particularly `sample_MUsim_obj` and `batch_run_MUsim`.