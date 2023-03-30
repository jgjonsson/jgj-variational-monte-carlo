#include <iostream>
#include <vector>
#include <memory>

#include "../../include/system.h"
#include "../../include/simplegaussian.h"
#include "../../include/harmonicoscillator.h"
#include "../../include/initialstate.h"
#include "../../include/metropolis_hastings.h"
#include "../../include/random.h"
#include "../../include/particle.h"
#include "../../include/sampler.h"

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
    std::vector<double> params{argc > 4 ? stod(argv[4]) : 0.5}; // Variational parameter.
    double stepLength = 0.1;                                    // Metropolis step length.
    size_t MC_reduction = 100;                                  // Number of MC steps to reduce by at intermediate steps
    bool verbose = true;                                        // Verbosity of output     

    // Let's perform optimization here; Gradient descent to be used

    std::vector<double> learning_rate; // deduced automatically
    double parameter_tolerance = 1e-4;
    size_t max_iterations = 1e2;
    bool converged = false;

    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<System> system;

    for (size_t count = 0; count < max_iterations; ++count)
    {
        // The random engine can also be built without a seed
        auto rng = std::make_unique<Random>(seed);
        // Initialize particles
        auto particles = setupRandomUniformInitialState(stepLength, numberOfDimensions, numberOfParticles, *rng);
        // Construct a unique pointer to a new System
        system = std::make_unique<System>(
            // Construct unique_ptr to Hamiltonian
            std::make_unique<HarmonicOscillator>(omega),
            // Construct unique_ptr to wave function
            std::make_unique<SimpleGaussian>(params[0]),
            // Construct unique_ptr to solver, and move rng
            std::make_unique<MetropolisHastings>(std::move(rng)),
            // Move the vector of particles to system
            std::move(particles));

        // Run steps to equilibrate particles
        auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
            stepLength,
            numberOfEquilibrationSteps / MC_reduction * (converged ? MC_reduction : 1));

        // Run the Metropolis algorithm
        sampler = system->runMetropolisSteps(
            stepLength,
            numberOfMetropolisSteps / MC_reduction * (converged ? MC_reduction : 1));

        if (converged)
            break;

        // Extract the gradient
        auto gradient = std::vector<double>(params.size());
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            gradient[param_num] = sampler->getObservables()[2 + param_num];
        }

        // At first iteration, choose reasonable learning rate
        if (count == 0)
        {
            learning_rate = std::vector<double>(params.size());
            for (size_t param_num = 0; param_num < params.size(); ++param_num)
            {
                if (fabs(gradient[param_num]) < 0.1)
                    learning_rate[param_num] = 1;
                else
                    learning_rate[param_num] = fabs(0.1 / gradient[param_num]);
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
            sampler->printOutputToTerminal(*system);
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
    sampler->printOutputToTerminal(*system, verbose);

    return 0;
}
