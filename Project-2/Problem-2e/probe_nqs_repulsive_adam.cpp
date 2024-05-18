#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "omp.h"

#include "../../include/system.h"
#include "../../include/gaussianjastrow.h"
#include "../../include/harmonicoscillator.h"
#include "../../include/hamiltonian_cyllindric_repulsive.h"
#include "../../include/hamiltonian_coulomb.h"
#include "../../include/initialstate.h"
#include "../../include/metropolis_hastings.h"
#include "../../include/metropolis.h"
#include "../../include/random.h"
#include "../../include/particle.h"
#include "../../include/sampler.h"
#include "../../include/file_io.h"
#include "../../include/rbm.h"
#include "../../include/adam.h"

using namespace std;

std::unique_ptr<Sampler> runParallellMonteCarloSimulation(unsigned int globalSeed, size_t numberOfMetropolisSteps, size_t MC_reduction, bool converged, size_t max_iterations, size_t count, int numThreads, double stepLength, size_t numberOfDimensions, size_t numberOfParticles, std::vector<double> params, double omega, double inter_strength, size_t rbs_M, size_t rbs_N, double hard_core_size, std::unique_ptr<Sampler> samplers[]) {

    // Random number setup in the way recommended for parallell computing, at https://github.com/anderkve/FYS3150/blob/master/code_examples/random_number_generation/main_rng_in_class_omp.cpp
    //  Use the system clock to get a base seed
    unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();

    #pragma omp parallel shared(samplers, count) // Start parallel region.
    {
        int thread_id = omp_get_thread_num();

        // Seed the generator with a seed that is unique for this thread
        unsigned int my_seed = base_seed + thread_id;
        auto rng = std::make_unique<Random>(my_seed);

        size_t numberOfMetropolisStepsPerGradientIteration = numberOfMetropolisSteps / MC_reduction * (converged | count == max_iterations - 1 ? MC_reduction : 1);
        numberOfMetropolisStepsPerGradientIteration /= numThreads; // Split by number of threads.

        std::unique_ptr<Sampler> sampler;
        std::unique_ptr<System> system;

        // Initialize particles
        auto particles = setupRandomUniformInitialStateWithRepulsion(
            stepLength,
            hard_core_size,
            numberOfDimensions,
            numberOfParticles,
            *rng);

        // Construct a unique pointer to a new System
        system = std::make_unique<System>(
            // Construct unique_ptr to Hamiltonian
            std::make_unique<CoulombHamiltonian>(omega, inter_strength),
            // Construct unique_ptr to wave function
            std::make_unique<SimpleRBM>(rbs_M, rbs_N, params, omega),
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
    return std::unique_ptr<Sampler>(new Sampler(samplers, numThreads));
}

std::vector<double> optimizeAndUpdateParameters(std::vector<double>& params, std::unique_ptr<Sampler>& combinedSampler, AdamOptimizer& adamOptimizer, bool verbose, double& parameter_tolerance, size_t count) {
    auto gradient = std::vector<double>(params.size());
    for (size_t param_num = 0; param_num < params.size(); ++param_num)
    {
        gradient[param_num] = combinedSampler->getObservables()[2 + param_num];
    }
    auto NewParams = adamOptimizer.adamOptimization(params, gradient);
    double sum = 0.0;
    for (size_t i = 0; i < params.size(); ++i) {
        double diff = NewParams[i] - params[i];
        sum += diff * diff;
    }
    double meanSquareDifference = sum / params.size();

    if (verbose)
    {
        cout << "Iteration " << count << endl;
        cout << endl;
    }

    cout << "Tolerance " << parameter_tolerance << " Total change: " << meanSquareDifference << endl;

    return NewParams;
}

int main(int argc, char **argv)
{
    // This program is for experimenting with learning rate and number of hidden layers.
    // We hard code number of particles to one.
    size_t numberOfDimensions = argc > 1 ? stoi(argv[1]) : 3;
    size_t numberOfParticles = argc > 2 ? stoi(argv[2]) : 2;

    // Number of visible layers for the RBM
    size_t rbs_M = numberOfParticles * numberOfDimensions;

    // Number of hidden layers for the RBM
    size_t rbs_N = argc > 3 ? stoi(argv[3]) : 3;

    // Number of times the gradient descent will run.
    // We don't know yet what tolerance is good to use, so start by just experimenting with this number and see what tolerance we may achieve.
    size_t fixed_number_optimization_runs = argc > 4 ? stoi(argv[4]) : 50;

    // Variational parameters for the RBM
    std::vector<double> params{};

    // Start with all parameters as random values
    int parameter_seed = 2023;         // For now, pick a hardcoded seed, so we get the same random number generator every run, since our goal is to compare settings.
    double parameterGuessSpread = 0.1; // Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.

    params = SimpleRBM::generateRandomParameterSet(rbs_M, rbs_N, parameter_seed, parameterGuessSpread);

    // We're experimenting with what learning rate works best.
    double fixed_learning_rate = argc > 5 ? stod(argv[5]) : 0.05;

    // Number of MCMC cycles for the large calculation after optimization
    size_t numberOfMetropolisSteps = argc > 6 ? stoi(argv[6]) : 1e6;

    size_t numberOfEquilibrationSteps = numberOfMetropolisSteps / 10;
    double omega = 1.0; // Oscillator frequency.
    double hard_core_size = 0.0043 / sqrt(omega); // Hard core size of particles, only used for initialisation of particle positions here, just to not make them start on top of each other.

    size_t MC_reduction = 100; // Number of MC steps to reduce by at intermediate steps

    double inter_strength = 1.0; // Strength of interaction.
                                 /*
                                 //Other parameters we used in project 1 that we might want to add back in.
                                 */
    double stepLength = 0.1;     // Metropolis step length.
    bool verbose = true;         // Verbosity of output

    // Let's perform optimization here; Gradient descent to be used

    std::vector<double> learning_rate; // deduced automatically
    double parameter_tolerance = 1e-2;
    size_t max_iterations = fixed_number_optimization_runs + 1; // 1e2;  //TODO: hack for converge condition on set number of iterations
    bool converged = false;

    std::unique_ptr<Sampler> combinedSampler;

    int numThreads = 14;
    omp_set_num_threads(numThreads);
    std::unique_ptr<Sampler> samplers[numThreads] = {};

    //Initialize Adam optimizer
    AdamOptimizer adamOptimizer(params.size(), fixed_learning_rate);

    for (size_t count = 0; count < max_iterations; ++count)
    {
        combinedSampler = runParallellMonteCarloSimulation(parameter_seed, numberOfMetropolisSteps, MC_reduction, converged, max_iterations, count, numThreads, stepLength, numberOfDimensions, numberOfParticles, params, omega, inter_strength, rbs_M, rbs_N, hard_core_size, samplers);
        params = optimizeAndUpdateParameters(params, combinedSampler, adamOptimizer, verbose, parameter_tolerance, count);
        /*
        // Extract the gradient
        auto gradient = std::vector<double>(params.size());
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            gradient[param_num] = combinedSampler->getObservables()[2 + param_num];
        }
        auto NewParams = adamOptimizer.adamOptimization(params, gradient);
        double sum = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            double diff = NewParams[i] - params[i];
            sum += diff * diff;
        }
        double meanSquareDifference = sum / params.size();
        params = NewParams;

        if (verbose)
        {
            cout << "Iteration " << count << endl;
            // cout << "Predictions: ";
            // combinedSampler->printOutputToTerminal();
            cout << endl;
        }

        cout << "Tolerance " << parameter_tolerance << " Total change: " << meanSquareDifference << endl;
        */
    }
    // Output information from the simulation
    combinedSampler->printOutputToTerminal(verbose);

    //Write energies to file, to be used by blocking method script.
    one_columns_to_csv("energies.csv", combinedSampler->getEnergyArrayForBlocking(), ",", 0, 6);

    return 0;
}
