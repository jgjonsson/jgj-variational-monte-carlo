# Problem1e implementation

This section contains only a Python script for computing standard deviation via Blocking method. 
The script was taken and reused from an article - see python/README for details. 

This README will however list several command lines, used to produce data for error analysis sections of the report.

# Results recreation

This assumes the executables of 1g was built first. If not, build by
```
make app=Project-1/Problem-1g/repulsive_hamiltonian
```

Run the following chain of commands, from root directory of repo:

```
time bin/Project-1/Problem-1g/repulsive_hamiltonian.out 3 5 1000000 0.6
python3 Project-1/Problem-1e/python/blocking.py
time bin/Project-1/Problem-1g/repulsive_hamiltonian.out 3 10 1000000 0.6
python3 Project-1/Problem-1e/python/blocking.py
time bin/Project-1/Problem-1g/repulsive_hamiltonian.out 3 20 1000000 0.6
python3 Project-1/Problem-1e/python/blocking.py
time bin/Project-1/Problem-1g/repulsive_hamiltonian.out 3 50 1000000 0.6
python3 Project-1/Problem-1e/python/blocking.py
time bin/Project-1/Problem-1g/repulsive_hamiltonian.out 3 100 1000000 0.6
python3 Project-1/Problem-1e/python/blocking.py
time bin/Project-1/Problem-1g/repulsive_hamiltonian.out 3 50 1000000 0.6
python3 Project-1/Problem-1e/python/blocking.py
```

Data for report has to be extracted by copying from the output. This is how table 1 in report was created. 
For illustration, here is output example for 50 particles. 

```
$ time bin/Project-1/Problem-1g/repulsive_hamiltonian.out 3 50 1000000 0.6
Iteration 0
Predictions: 0.6 138.973 0.0654613 21.9089
Iteration 1
Predictions: 0.5 129.753 0.0128407 4.28064
Iteration 2
Predictions: 0.480462 128.1 0.0140876 0.408995
Iteration 3
Predictions: 0.478595 127.567 0.0104204 1.09653
Iteration 4
Predictions: 0.47359 127.19 0.00934976 -0.844408
Iteration 5
Predictions: 0.477444 127.535 0.00860825 -0.00444147
Parameters converged after 5 iterations.

  -- System info --
 Number of particles  : 50
 Number of dimensions : 3
 Number of Metropolis steps run : 10^6
 Step length used : 0.1
 Ratio of accepted steps: 0.893125

  -- Wave function parameters --
 Number of parameters : 1
 Parameter 1 : 0.477464

  -- Results --
 Energy : 127.345
 Standard deviation Energy : 0.00198028
 Computed gradient : -6.45094


real    100m13.432s
user    768m8.291s
sys     0m5.152s
$ python3 Project-1/Problem-1e/python/blocking.py
Warning: Data size = 9999, is not a power of 2.
Truncating data to 8192.
Standard error = 0.03588
```
