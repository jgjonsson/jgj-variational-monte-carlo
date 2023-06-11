# Problem2b implementation

Problem 2b was about implementing the necessary classes for the Restricted Boltzmann Machine type of wave function, plus updating the gradient descent and start experimenting with learning rate. 

# Building the code
If run from the root directory of the Git-repo, the build commands are:

```
make app=Project-2/Problem-2b/unit_tests
make app=Project-2/Problem-2b/single_particle 
make app=Project-2/Problem-2b/large_simulation
```

# Running tests

To run the unit tests, run this program:
```
bin/Project-2/Problem-2b/unit_tests.out
```

No parameter needed. It runs some tests, print various output, and if everything is ok ends with the line: All tests passed!

# Running simulations

Example for single particle (in 3D), 6 hidden nodes, 100 iterations for optmization, 0.01 learning rate, 10^7 MCMC cycles. 
Results in energy pretty cloes to 1.5 as expected. 
```
bin/Project-2/Problem-2b/single_particle.out 6 100 0.01 10000000
```

Example for simulating 2 dimensions, 2 Particles, 2 10^7, MCMC cycles. Rest of parameters hardcoded so check source code.
Should give a result of energy cloes to 2. 
```
bin/Project-2/Problem-2b/large_simulation.out 2 2 10000000
```

# Commands useful for reproducing report results 
## For table 1 in report
```
time bin/Project-2/Problem-2b/large_simulation.out 1 1 1000000
```

## Single particle initial investigation different learning rates and number of hidden nodes
Comparing different learning rates:
```
time bin/Project-2/Problem-2b/single_particle.out 6 100 1 10000000
time bin/Project-2/Problem-2b/single_particle.out 6 100 0.1 10000000
time bin/Project-2/Problem-2b/single_particle.out 6 100 0.05 10000000
time bin/Project-2/Problem-2b/single_particle.out 6 100 0.01 10000000
```
giving the below results per learning rate:
1: Results just diverging - learning rate too high
0.1: E=1.50011 sigma=1.52223e-05, parameterdelta=0.00252424
0.05: E=1.50009 sigma=1.59472e-05, parameterdelta=0.000423908
0.01: E=1.5005 sigma=2.13325e-05, parameterdelta=0.00150819

Sigma being MCMC standard deviation, and parameterdelta being "Total change" - sum of absolut change in parameter values at last step.
So in this case learning rate 0.05 was best since it was closest to 1.5 as well as resulting in the lowest change in parameter value by last optmization step. 
At the same time it has decent standard deviation on energy (pratically tied with 0.1).
For all cases, the simulation run the full 100 optmization loops that we set as a fixed maximum. For a different number of loops, the best learning rate could well have been different. 
Increasing number of MCMC cycles from 10^6 to 10^7 however did not seem significant for this conclusion.

Comparing different number of hidden nodes:
```
time bin/Project-2/Problem-2b/single_particle.out 6 100 0.05 10000000
time bin/Project-2/Problem-2b/single_particle.out 8 100 0.05 10000000
time bin/Project-2/Problem-2b/single_particle.out 10 100 0.05 10000000
time bin/Project-2/Problem-2b/single_particle.out 12 100 0.05 10000000
```

Here 6 hidden nodes gave E=1.50014 which was the value closest to 1.5. 
The number 6 was initially a heuristic guess when studying 1 particle in 3D assuming at least two hidden nodes per degree of freedom seemed reasonable. 
In retrospect we should have started all the way from 1 node because some lower number could potentially actually have given a better value. 
