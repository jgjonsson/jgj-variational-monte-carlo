#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "omp.h"

#include "../../include/system.h"
#include "../../include/gaussianjastrow.h"
#include "../../include/hamiltonian_cyllindric_repulsive.h"
#include "../../include/initialstate.h"
#include "../../include/metropolis_hastings.h"
#include "../../include/metropolis.h"
#include "../../include/random.h"
#include "../../include/particle.h"
#include "../../include/sampler.h"
#include "../../include/file_io.h"
#include "../../include/rbm.h"

using namespace std;

int main(int argc, char **argv)
{
    // Seed for the random number generator
    int seed = 2023;

    size_t numberOfDimensions = argc > 1 ? stoi(argv[1]) : 1;
    size_t numberOfParticles = argc > 2 ? stoi(argv[2]) : 1;
    size_t numberOfMetropolisSteps = argc > 3 ? stoi(argv[3]) : 1e6;
    size_t numberOfEquilibrationSteps = numberOfMetropolisSteps / 10;
    double omega = 1.0;                                         // Oscillator frequency.
    double beta = 2.82843;                                      // Frequency ratio
    double hard_core_size = 0.0043 / sqrt(omega);               // Hard core size
    std::vector<double> params{argc > 4 ? stod(argv[4]) : 0.5}; // Variational parameter.
    double stepLength = 0.1;                                    // Metropolis step length.
    size_t MC_reduction = 100;                                  // Number of MC steps to reduce by at intermediate steps
    bool verbose = true;                                        // Verbosity of output

    // Let's perform optimization here; Gradient descent to be used

    std::vector<double> learning_rate; // deduced automatically
    double parameter_tolerance = 1e-4;
    size_t max_iterations = 1e2;
    bool converged = false;

    std::unique_ptr<Sampler> combinedSampler;

    int numThreads = 1; //Wait a bit with parallellization
    omp_set_num_threads(numThreads);
    std::unique_ptr<Sampler> samplers[numThreads] = {};

	size_t rbs_M = numberOfParticles*numberOfDimensions;
	size_t rbs_N = rbs_M; //This N is something to experiment with, but start by trying equal to M.
	
    for (size_t count = 0; count < max_iterations; ++count)
    {
        // Random number setup in the way recommended for parallell computing, at https://github.com/anderkve/FYS3150/blob/master/code_examples/random_number_generation/main_rng_in_class_omp.cpp
        //  Use the system clock to get a base seed
        unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();
#pragma omp parallel shared(samplers, count) // Start parallel region.
        {
            int thread_id = omp_get_thread_num();

            // Seed the generator with a seed that is unique for this thread
            unsigned int my_seed = base_seed + thread_id;
            auto rng = std::make_unique<Random>(my_seed);
            auto rng2 = std::make_unique<Random>(my_seed+numThreads);  //Make a second generator, with +numThreads making this seed unique
            //This is done so that SimpleRBM and MetropolisHastings below get their own rnd. std::move will make it only available to one
            //TODO: See if std::move should be used for both. Maybe we'll keep using rng2 for stochastic gradient descent to. To be figured out.

            size_t numberOfMetropolisStepsPerGradientIteration = numberOfMetropolisSteps / MC_reduction * (converged ? MC_reduction : 1);
            numberOfMetropolisStepsPerGradientIteration /= numThreads; // Split by number of threads.

            std::unique_ptr<Sampler> sampler;
            std::unique_ptr<System> system;

            // Initialize particles
            auto particles = setupRandomUniformInitialStateWithRepulsion(stepLength, hard_core_size, numberOfDimensions, numberOfParticles, *rng);

            // Construct a unique pointer to a new System
            system = std::make_unique<System>(
                // Construct unique_ptr to Hamiltonian
                std::make_unique<RepulsiveHamiltonianCyllindric>(omega, beta),
                // Construct unique_ptr to wave function
                std::make_unique<SimpleRBM>(rbs_M, rbs_N, *rng2 ),
                // Construct unique_ptr to solver, and move rng
                std::make_unique<MetropolisHastings>(std::move(rng)),
                // Move the vector of particles to system
                std::move(particles));

            // Run steps to equilibrate particles
            auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
                stepLength,
                numberOfMetropolisStepsPerGradientIteration);

            // Run the Metropolis algorithm
            samplers[thread_id] = system->runMetropolisSteps(
                stepLength,
                numberOfMetropolisStepsPerGradientIteration);
        }

        // Create a new Sampler object containing the average of all the others.
        combinedSampler = std::unique_ptr<Sampler>(new Sampler(samplers, numThreads));

        if (converged)
            break;

        // Extract the gradient
        auto gradient = std::vector<double>(params.size());
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            gradient[param_num] = combinedSampler->getObservables()[2 + param_num];
        }

        // At first iteration, choose reasonable learning rate
        if (count == 0)
        {
            learning_rate = std::vector<double>(params.size());
            for (size_t param_num = 0; param_num < params.size(); ++param_num)
            {
                /*
                if (fabs(gradient[param_num]) < 0.1)
                    learning_rate[param_num] = 1;
                else
                    learning_rate[param_num] = fabs(0.1 / gradient[param_num]);
                */
                //learning_rate[param_num] = 0.01;  //Using hardcoded value like in lecture examples, rather than trying to calculate a more optimal one.
                learning_rate[param_num] = 0.1/numberOfParticles;  //Purely on emprihical basis, we experience divergence problems with higher learning rate on large number of particles.
                cout << "Learning rate: " << learning_rate[param_num] << endl;
            }
        }

        // Update the parameter
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            params[param_num] -= learning_rate[param_num] * gradient[param_num];
        }
        if (verbose)
        {
            cout << "Iteration " << count << endl;
            cout << "Predictions: ";
            combinedSampler->printOutputToTerminal();
            cout << endl;
        }

        // Check if the parameter has converged (estimate as total parameter change < tolerance)

        double total_change = 0;
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            total_change += fabs(learning_rate[param_num] * gradient[param_num]);
        }
        if (total_change < parameter_tolerance)
        {
            if (verbose)
                cout << "Parameters converged after " << count << " iterations." << endl;
            converged = true;
            continue;
        }
    }
    // Output information from the simulation
    combinedSampler->printOutputToTerminal(verbose);

    //Write energies to file, to be used by blocking method script.
    one_columns_to_csv("energies.csv", combinedSampler->getEnergyArrayForBlocking(), ",", 0, 6);

    return 0;
}
