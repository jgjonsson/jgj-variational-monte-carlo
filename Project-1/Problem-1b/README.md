# Problem1b implementation

Here we run simple tests with gaussian trial function to make sure we have the correct implementation of variational Monte Carlo method.

# Results recreation

I assume one runs the code from the directory with this README.

## MC data accumulation for 1D without optimization

Build `energy_calc.cpp` with make

```bash
make app=Project-1/Problem-1b/energy_calc -C ../../
```

Run the accumulator (You might want to change the number of cycles in the script, if you don't wish to wait a few hours.)

```bash
bash accumulate_MC_data.sh
```

## Comparison between analytical derivative computation and numerical derivative computation

Build both `energy_calc.cpp` and `inefficient_energy_calc.cpp` with make same as above.

Run them on reasonable settings (Ndimensions = 3; Nparticles = 10 is sufficient) and compare the timings (energies calculated are the same of course).

`inefficient_energy_calc.cpp` reimplements SimpleGaussian so that default implementations for derivatives and ratio evaluators are used. This is a very inefficient way of doing things, but it is a good way to check that both the analytical derivatives are correct and default implementations are correct.

## Plotting

Run the plotting script

```bash
python3 plot_xy.py --datafile <path to x|y data file> --savefig <path to save figure=temp.pdf>
```

Run visual inspection of the plots to make sure they look reasonable. Parameter alpha corresponds to the true wave function parameter when equals to 0.5.