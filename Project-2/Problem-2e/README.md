# Problem2e implementation

Here we for the first time probe the NQS wavefunction on a system with Coulomb repulsion.

# Building the code
If run from the root directory of the Git-repo, the build commands are:

```
make app=Project-2/Problem-2e/probe_nqs_repulsive
```

# Running simulations

Sample invokation
```
./bin/Project-2/Problem-2e/probe_nqs_repulsive.out 2 2 50 100
```

```
./bin/NeuralNet/ny/probe_nqs_repulsive.out 2 2 50 100
```

results in a WF with 254 parameters for 2D, and gives best energy of 3.11 (compared to 3.0 from analytics)

# Command lines used for report data
# For Importance sampling:
./bin/Project-2/Problem-2e/probe_nqs_repulsive.out 2 2 10 100
python3 Project-1/Problem-1e/python/blocking.py
./bin/Project-2/Problem-2e/probe_nqs_repulsive.out 2 2 50 100
python3 Project-1/Problem-1e/python/blocking.py

# For brute force, currently code must be changed manually, changing row 116 from MetropolisHastings to Metropolis