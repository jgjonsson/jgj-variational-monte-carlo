# Problem1d implementation

Prior to probing actual repulsive system, where we have not much theoretical knowledge, we introduce custom gradient descent for generic parameter optimization.

# Results recreation

I assume one runs the code from the directory with this README.

## Manual unit testing

In the directory you are both provided with optimizer for Harmonic Oscillator system, as well as with optimizer for repulsive system. One can mess with parameters on leasure, but main points:
- Descending rate is self-adaptive, so it is not necessary to tune it. It goes at the price one has to start a bit away from expected minimum (away by 0.1 should be most comfortable), and the parameters are expected to be of order ~E1.
- Optimizer runs at way lower cycle number (controlled by MC_reduction=100). While this is comfortable, it may not be enough to find the actual minimum. One can introduce more gradual reduction of MC cycles, but it is currently sufficient to just increase the number of cycles.
- OpenMP is not enabled here

To probe the optimizer, one can run the following commands:

```bash
make app=Project-1/Problem-1d/probe_optimizer_gaussian -C ../../
```

Run it with 

```bash
../../bin/Project-1/Problem-1d/probe_optimizer_gaussian.out <Ndimensions> <Nparticles> <Ncycles> <initial_guess_alpha>
```

For this very system, the minimum is at `alpha=0.5`, with `E=NDIM*NPARTCLS/2`. The energy curve is quite smooth, so don't expect much problems even with low number of cycles.

You can try repulsive system as well here..