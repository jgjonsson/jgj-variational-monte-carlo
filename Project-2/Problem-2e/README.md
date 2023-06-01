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
./bin/Project-2/Problem-2e/importance_sampling_vs_brute_force.out 2 2 50 100
```
results in a WF with 254 parameters for 2D, and gives best energy of 3.11 (compared to 3.0 from analytics)