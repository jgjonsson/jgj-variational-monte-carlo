# Problem1c implementation

Another program with simple tests with gaussian trial function. 
Almost identical to 1b, except here we run Metropolis-Hastings (Importance sampling) instead of Metropolis (bruteforce).

# Results recreation

I assume one runs the code from the directory with this README.

## MC data accumulation for 1D without optimization

Build `importance_sampling.cpp` with make

```bash
make app=Project-1/Problem-1b/importance_sampling -C ../../
```

Run the accumulator (You might want to change the number of cycles in the script, if you don't wish to wait a few hours.)

```bash
bash accumulate_MC_data.sh
```

## Comparison between analytical derivative computation and numerical derivative computation

Build `importance_sampling.cpp` with make same as above.

Run them on reasonable settings (Ndimensions = 3; Nparticles = 10 is sufficient) and compare the timings (energies calculated are the same of course).

## Plotting

Run the plotting script

```bash
python3 ../Problem-1b/plot_xy.py --datafile <path to x|y data file> --savefig <path to save figure=temp.pdf>
```

Run visual inspection of the plots to make sure they look reasonable. Parameter alpha corresponds to the true wave function parameter when equals to 0.5.

## Runs used in report, from repo root. Used in section for Importance samling under result.
time bin/Project-1/Problem-1b/energy_calc.out 3 20 1000000 0.5
time bin/Project-1/Problem-1c/importance_sampling.out 3 20 1000000 0.5
