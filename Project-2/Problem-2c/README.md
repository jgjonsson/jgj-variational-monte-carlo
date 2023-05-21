# Problem2c implementation

Problem 2c was about implementing importance samling (Metropolis-Hastings)and compare with brute force (Metropolis).

# Building the code
If run from the root directory of the Git-repo, the build commands are:

```
make app=Project-2/Problem-2c/importance_sampling_vs_brute_force
```

# Running simulations

We added 4 more parameters as compared to Problem 1b, enabling chosing number of particles and dimensions, and type of Hamiltonian and Montecarlo algorithm.

Example with the parameters in order specifying:
2 dimensions
2 particles
6 hidden layer nodes
100 iterations set for optmization run (always, regardless of tolerance)
0.01 learning rate
10^7 MCMC cycles for the final large calculation
Harmonic oscilator (no interaction repulsive force)
Metropolis algorithm (brute-force)
```
time bin/Project-2/Problem-2c/importance_sampling_vs_brute_force.out 2 2 6 100 0.01 10000000 HARMONIC METROPOLIS
```
Should result in 34 parameters, and energy very close to 2.0. Run takes about 2.5 minutes.

Switching to importance sampling:
```
time bin/Project-2/Problem-2c/importance_sampling_vs_brute_force.out 2 2 6 100 0.01 10000000 HARMONIC METROPOLIS_HASTINGS
```
Should also result in 34 parameters, and energy very close to 2.0. Run takes about 2.5 minutes.

Another example, 3 dimensions, 10 particles, Importance sampling

Since we still have the Blocking script from Project 1, it's also possible to run that directly after if you like:
```
python3 Project-1/Problem-1e/python/blocking.py
```
