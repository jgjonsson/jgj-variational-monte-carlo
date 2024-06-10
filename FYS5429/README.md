

The following commands assume current directory is FYS5429 (if instead running from root of repo, remove -C .. or ../ respectively).

## Restricted Boltzmann Machine (RBM)

Building the executable for Restricted Boltzmann Machine (RBM) with Neural Quantum States (NQS) using Adam optimizer:
```
make -C .. app=FYS5429/rbm/probe_nqs_rbm_repulsive_adam
```

Run RBS for 2 dimensions, 2 particles, 20 hidden nodes, 600 epochs, learning rate 0.01, and 54 million Monte Carlo cycles:
```
time ../bin/FYS5429/rbm/probe_nqs_rbm_repulsive_adam.out 2 2 20 600 0.01 54000000
```
To give an idea what run time to expect, this takes about 20 minutes on a 2.5 GHz Intel Core 13th gen i7, with 14 cores - 20 logical cores.
Other things like monte caro step length, and number of threads are hardcoded but can be changed by changing variables in the code.

This creates the file energies.csv
To run resampling to calculate statistical error, run:
```
python3 blocking.py
```

The full table of results for 2 particles 2D can be obtained by running this, but replacing 20, 600 and 0.01 with other values. 

Another example, to run 5 particles in 3D (bosonic interaction and cyllindrical trap) with 30 hidden nodes.
```
time ../bin/FYS5429/rbm/probe_nqs_rbm_repulsive_adam.out 3 5 30 600 0.01 54000000
```

## Mixed Neural Network with exact solution for non-interacting part

Building the executable for Mixed Neural Network:
```
make -C .. app=FYS5429/mixed/probe_nqs_repulsive_mixed
```

Example for running:
```
time ../bin/FYS5429/mixed/probe_nqs_repulsive_mixed.out 2 2 20 300 0.001 41000000 INTERACTION METROPOLIS
```

## Pure generic neural network with one layer

Building the executable for pure generic Neural Network:
```
make -C .. app=FYS5429/neuralnetwork/pretrain
make -C .. app=FYS5429/neuralnetwork/probe_nqs_repulsive_nn_train
```

Example for running:
```
time ../bin/FYS5429/neuralnetwork/pretrain.out 2 2 16 100 0.05 3100000 INTERACTION METROPOLIS_HASTINGS
time ../bin/FYS5429/neuralnetwork/probe_nqs_repulsive_nn_train.out 2 2 16 250 0.01 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_16_0.050000.csv
```
For this 16 nodes example, expect 30min pretraining and 100min training. It's on the lower end and might not converge well.

Here is two other examples that take significantly longer to run, but should match the best results in the report:
```
time ../bin/FYS5429/neuralnetwork/pretrain.out 2 2 30 500 0.05 3100000 INTERACTION METROPOLIS_HASTINGS
time ../bin/FYS5429/neuralnetwork/probe_nqs_repulsive_nn_train.out 2 2 30 500 0.01 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_30_0.050000.csv
```

```
time ../bin/FYS5429/neuralnetwork/pretrain.out 2 2 40 500 0.1 3100000 INTERACTION METROPOLIS_HASTINGS
time ../bin/FYS5429/neuralnetwork/probe_nqs_repulsive_nn_train.out 2 2 40 500 0.01 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_40_0.100000.csv
```

Plotting K-values and energies from this run:
```
python3 plot_training_energies.py --datafile energies_plot_pure_2_2_40.csv --ylabel "<E> a.u." --savefig energies_plot_pure_2_2_40.pdf
python3 plot_training_k.py --datafile K_pretrain_2_2_40_0.100000.csv --ylabel "K" --savefig k_plot_2_2_40.pdf
python3 plot_1-K_log.py --datafile K_pretrain_2_2_40_0.100000.csv --ylabel "1-K" --savefig k-1_plot_2_2_40.pdf
```

Example for 5 particles in 3D (bosonic interaction and cyllindrical trap) with 30 hidden nodes - similar but not exactly the run used in report:
```
time ../bin/FYS5429/neuralnetwork/pretrain.out 3 5 12 0.05 11100000 INTERACTION METROPOLIS_HASTINGS
time ../bin/FYS5429/neuralnetwork/probe_nqs_repulsive_nn_train.out 3 5 30 250 0.01 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_3_5_30.csv
```

## Pure generic neural network with two layers
```
make -C .. app=FYS5429/neuralnetwork_twolayer/pretrain_twolayer
make -C .. app=FYS5429/neuralnetwork_twolayer/probe_nqs_repulsive_nn_train_two_layer
```
Example for running:
```
time ../bin/FYS5429/neuralnetwork_twolayer/pretrain_twolayer.out 2 2 16 100 0.05 3100000 INTERACTION METROPOLIS_HASTINGS
time ../bin/FYS5429/neuralnetwork_twolayer/probe_nqs_repulsive_nn_train_two_layer.out 2 2 16 250 0.01 21100000 INTERACTION METROPOLIS_HASTINGS NNparams_pretrain_2_2_16_0.050000.csv
```
Warning: Extreamly slow - expect 10 hours just for pretraining. Then training does not even converge well.


## Tests - unit tests and benchmark tests
```
make -C .. app=FYS5429/tests/benchmark_differentiation
make -C .. app=FYS5429/tests/unit_tests
make -C .. app=FYS5429/tests/unit_tests_backprop
make -C .. app=FYS5429/tests/unit_tests_performance
make -C .. app=FYS5429/tests/unit_tests_qforce
make -C .. app=FYS5429/tests/unit_tests_reverse
make -C .. app=FYS5429/tests/unit_tests_two_layer
```

Run tests:
```
../bin/FYS5429/tests/benchmark_differentiation.out
../bin/FYS5429/tests/unit_tests.out
../bin/FYS5429/tests/unit_tests_backprop.out
../bin/FYS5429/tests/unit_tests_performance.out
../bin/FYS5429/tests/unit_tests_qforce.out
../bin/FYS5429/tests/unit_tests_reverse.out
../bin/FYS5429/tests/unit_tests_two_layer.out
```
