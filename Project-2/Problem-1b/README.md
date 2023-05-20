# Problem2b implementation

Problem 2b was about implementing the necessary classes for the Restricted Boltzmann Machine type of wave function, plus updating the gradient descent and start experimenting with learning rate. 

# Building the code
If run from the root directory of the Git-repo, the build commands are:

```
make app=Project-2/Problem-1b/unit_tests
make app=Project-2/Problem-1b/single_particle 
make app=Project-2/Problem-1b/large_simulation
```

# Running tests

To run the unit tests, run this program:
```
bin/Project-2/Problem-1b/unit_tests.out
```

No parameter needed. It runs some tests, print various output, and if everything is ok ends with the line: All tests passed!

# Running simulations

Example for single particle (in 3D), 6 hidden nodes, 100 iterations for optmization, 0.01 learning rate, 10^7 MCMC cycles. 
Results in energy pretty cloes to 1.5 as expected. 
```
bin/Project-2/Problem-1b/single_particle.out 6 100 0.01 10000000
```

Example for simulating 2 dimensions, 2 Particles, 2 10^7, MCMC cycles. Rest of parameters hardcoded so check source code.
Should give a result of energy cloes to 2. 
```
bin/Project-2/Problem-1b/large_simulation.out 2 2 10000000
```
