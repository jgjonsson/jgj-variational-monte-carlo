# Problem1g implementation

This section mostly originated both from implementation of optimizer and OpenMP, and is mostly dedicated for production code for repulsive system with many particles.

# Results recreation

I assume one runs the code from the directory with this README.

## Manual unit testing

To probe the repulsive system, one can run the following commands:

```bash
make app=Project-1/Problem-1g/repulsive_hamiltonian -C ../../
```

Run it with 

```bash
../../bin/Project-1/Problem-1g/repulsive_hamiltonian.out <Ndimensions> <Nparticles> <Ncycles> <initial_guess_alpha>
```

We do not know the exact minimum, but we know that it is somewhere around `alpha=0.5`, with `E=NDIM*NPARTCLS/2`, for low number of particles. Further result are discussed in the report.