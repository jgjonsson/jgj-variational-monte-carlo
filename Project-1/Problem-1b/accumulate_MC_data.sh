#!/bin/bash

# This script is used for invoking ../../bin/Project-1/Problem-1b/energy_calc.out to form csv files for plotting

# Path to the executable
path_to_executable="../../bin/Project-1/Problem-1b/energy_calc.out"


# Run with 1E6 MC cycles, [0.1, 0.2, ..., 1.0] alpha, and [1, 3, 10, 50, 500] particles
# stash the output for same number of particles in the same file

mkdir -p data
for particles in 1 3 10 50 500
do
    touch "data/energy_vs_alpha_particles_$particles.csv"
    for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        ./$path_to_executable 1 $particles 1000000 $alpha >> "data/energy_vs_alpha_particles_$particles.csv"
    done
done