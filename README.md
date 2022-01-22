class for motor unit simulations, produce firing rates according to arbitrary force profiles; operates on size principle when in "static" mode; "dynamic" mode applies unit MU threshold reversals when dF/dt exceeds a threshold value; shuffled thresholds can also be simulated in "MUsim_2Dkde_pair.py", "MUsim_2Dkde_PCA.py", "MUsim_2Dkde_PCA_multiRun.py".

Once your environment is configured with `conda create --name mu --file requirements.txt`, run `MUtest.py` for an example of a range of simulation parameters and plot outputs (see below).

![Output of MUtest.py](MUtest.html)
