

The following commands assume current directory is FYS5429 (if instead running from root of repo, remove -C .. or ../ respectively).

Building the executable for Restricted Boltzmann Machine (RBM) with Neural Quantum States (NQS) using Adam optimizer:
```
make -C .. app=FYS5429/rbm/probe_nqs_rbm_repulsive_adam
```

Run RBS for 2 dimensions, 2 particles, 20 hidden nodes, 600 epochs, learning rate 0.01, and 50 million Monte Carlo cycles:
```
time ../bin/FYS5429/rbm/probe_nqs_rbm_repulsive_adam.out 2 2 20 600 0.01 50000000
```

