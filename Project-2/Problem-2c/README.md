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
Should result in 34 parameters, and energy very close to 2.0. ~93% accepted steps. Run takes about 1.5 minutes.


Since we still have the Blocking script from Project 1, it's also possible to run that directly after if you like:
(Do it after every run of the C++ program, as it relies on energies.csv which is overwritten each time.)
```
python3 Project-1/Problem-1e/python/blocking.py
```

Switching to importance sampling:
```
time bin/Project-2/Problem-2c/importance_sampling_vs_brute_force.out 2 2 6 100 0.01 10000000 HARMONIC METROPOLIS_HASTINGS
```
Should result in 34 parameters, and energy very close to 2.0. ~98.9% accepted steps. Run takes about 2.5 minutes.

If we wanna go bigger - another example, 3 dimensions, 10 particles, Importance sampling
```
time bin/Project-2/Problem-2c/importance_sampling_vs_brute_force.out 3 10 12 100 0.01 16000000 HARMONIC METROPOLIS_HASTINGS
```
Runs 11 minutes, energy close to 15 as expected (0.5 * 10*3) 
