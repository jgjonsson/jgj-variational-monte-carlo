

The following commands assume current directory is FYS5429 (if instead running from root of repo, remove -C .. or ../ respectively).

Building the executable for Restricted Boltzmann Machine (RBM) with Neural Quantum States (NQS) using Adam optimizer:
```
make -C .. app=FYS5429/rbm/probe_nqs_rbm_repulsive_adam
```

Run RBS for 2 dimensions, 2 particles, 20 hidden nodes, 600 epochs, learning rate 0.01, and 54 million Monte Carlo cycles:
```
time ../bin/FYS5429/rbm/probe_nqs_rbm_repulsive_adam.out 2 2 20 600 0.01 54000000
```
To give an idea what run time to expect, this takes about 20 minutes on a 2.5 GHz Intel Core 13th gen i7, with 14 cores - 20 logical cores.

This creates the file energies.csv
To run resampling to calculate statistical error, run:
```
python3 blocking.py
```

