#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "omp.h"

#include <cmath>
#include <random>
#include <cstdlib>
#include <iomanip>

#include "../../include/pretrain_system.h"
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
#include "../../include/pretrain_sampler.h"
#include "../../include/file_io.h"
#include "../../include/simplegaussian.h"
#include "../../include/nn_wave.h"
#include "../../include/nn_wave_pure.h"
#include "../../include/adam.h"

// Define some ANSI escape codes for colors
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

using namespace std;

int main3();

std::unique_ptr<class MonteCarlo> createSolverFromArgument(string algoritm, std::unique_ptr<class Random> rng)
{
    if (algoritm=="METROPOLIS") return std::make_unique<Metropolis>(std::move(rng));
    if (algoritm=="METROPOLIS_HASTINGS") return std::make_unique<MetropolisHastings>(std::move(rng));
    cout << "Invalid type of algoritm for Monte Carlo, valid types are METROPOLIS and METROPOLIS_HASTINGS " << endl;
    exit(-1);
}

std::unique_ptr<Sampler> runParallellMonteCarloSimulation(unsigned int globalSeed, size_t numberOfEquilibrationStepsPerEpoch, size_t numberOfSteadyStateStepsPerEpoch, size_t count, int numThreads, double stepLength, size_t numberOfDimensions, size_t numberOfParticles, std::vector<double> params, double omega, double inter_strength, size_t rbs_M, size_t rbs_N, double hard_core_size, std::unique_ptr<Sampler> samplers[], bool finalRunNoGradientNeeded) {

    // Random number setup in the way recommended for parallell computing, at https://github.com/anderkve/FYS3150/blob/master/code_examples/random_number_generation/main_rng_in_class_omp.cpp
    //  Use the system clock to get a base seed
    unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();

    #pragma omp parallel shared(samplers, count) // Start parallel region.
    {
        int thread_id = omp_get_thread_num();

        // Seed the generator with a seed that is unique for this thread
        unsigned int my_seed = base_seed + thread_id;
        auto rng = std::make_unique<Random>(my_seed);

        //size_t numberOfSteadyStateStepsPerEpoch = numberOfMetropolisSteps / MC_reduction * (converged | count == max_iterations - 1 ? MC_reduction : 1);
        //numberOfSteadyStateStepsPerEpoch /= numThreads; // Split by number of threads.

        std::unique_ptr<Sampler> sampler;
        std::unique_ptr<System> system;

        // Initialize particles
        auto particles = setupRandomUniformInitialStateWithRepulsion(
            stepLength,
            hard_core_size,
            numberOfDimensions,
            numberOfParticles,
            *rng);
auto alpha=0.5;  //TODO: REmove this and remove variable from NN
auto beta=2.82843; //TODO: REmove this and remove variable from NN
auto omega=1.0; //TODO: REmove this and remove variable from NN
        // Construct a unique pointer to a new System
        system = std::make_unique<System>(
            // Construct unique_ptr to Hamiltonian
            std::make_unique<CoulombHamiltonian>(omega, inter_strength),
            // Construct unique_ptr to wave function
            std::make_unique<PureNeuralNetworkWavefunction>(rbs_M, rbs_N, params, omega, alpha, beta, 0.0),
            //std::make_unique<PureNeuralNetworkWavefunction>(rbs_M, rbs_N, params, omega, alpha, beta, adiabaticFactor),
            // Construct unique_ptr to solver, and move rng
            std::make_unique<MetropolisHastings>(std::move(rng)),
            // Move the vector of particles to system
            std::move(particles));
//cout << "Equilibrium steps: " << numberOfEquilibrationStepsPerEpoch << " Steady state steps: " << numberOfSteadyStateStepsPerEpoch << endl;
        // Run steps to equilibrate particles
        auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
            stepLength,
            numberOfEquilibrationStepsPerEpoch);

        // Run the Metropolis algorithm
        samplers[thread_id] = system->runMetropolisSteps(
            stepLength,
            numberOfSteadyStateStepsPerEpoch,
            finalRunNoGradientNeeded);
    }

    // Create a new Sampler object containing the average of all the others.
    return std::unique_ptr<Sampler>(new Sampler(samplers, numThreads));
}


std::vector<double> optimizeAndUpdateParameters(std::vector<double>& params, std::unique_ptr<Sampler>& combinedSampler, AdamOptimizer& adamOptimizer, bool verbose, size_t count) {
    auto gradient = std::vector<double>(params.size());
    for (size_t param_num = 0; param_num < params.size(); ++param_num)
    {
        gradient[param_num] = combinedSampler->getObservables()[2 + param_num];
    }
    auto newParams = adamOptimizer.adamOptimization(params, gradient);

    if (verbose)
    {
        double sum = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            double diff = newParams[i] - params[i];
            sum += diff * diff;
        }
        double meanSquareDifference = sum / params.size();
        cout << "Iteration " << count << " ";
        cout << " Total change: " << meanSquareDifference << endl;
    }

    return newParams;
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
    int parameter_seed = 2023;//111;//2023;         // For now, pick a hardcoded seed, so we get the same random number generator every run, since our goal is to compare settings.
    double parameterGuessSpread = 0.001; // Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.

    params = NeuralNetworkWavefunction::generateRandomParameterSet(rbs_M, rbs_N, parameter_seed, parameterGuessSpread);

double alpha = 0.5;//m_parameters[0]; // alpha is the first and only parameter for now.
double beta = 2.82843; // beta is the second parameter for now.

    // We're experimenting with what learning rate works best.
    double fixed_learning_rate = argc > 5 ? stod(argv[5]) : 0.01;

    // Number of MCMC cycles for the large calculation after optimization
    size_t numberOfMetropolisSteps = argc > 6 ? stoi(argv[6]) : 1e6;

	//Type of Hamiltonian to use, ie interaction or not, and shape of potential. Ex: HARMONIC, HARMONIC_CYLINDRIC_INTERACTION
    auto hamiltonianChoice = argc > 7 ? argv[7] : "INTERACTION";

    //Algoritm for sampling. Ex: METROPOLIS, METROPOLIS_HASTINGS (brute force and importance samling respectively)
    auto algoritmChoice = argc > 8 ? argv[8] : "METROPOLIS";

    auto parametersInputFile = argc > 9 ? argv[9] : "";

    size_t numberOfEquilibrationSteps = numberOfMetropolisSteps / 5;
    double omega = 1.0; // Oscillator frequency.             // Frequency ratio
    double hard_core_size = 0.0043 / sqrt(omega);

    size_t MC_reduction = 100; // Number of MC steps to reduce by at intermediate steps

    double inter_strength = 1.0; // Strength of interaction.
                                 /*
                                 //Other parameters we used in project 1 that we might want to add back in.
                                 */
    double stepLength = 0.6;     // Metropolis step length.
    bool verbose = true;         // Verbosity of output

    // Let's perform optimization here; Gradient descent to be used

    std::vector<double> learning_rate; // deduced automatically
    double parameter_tolerance = 1e-2;
    size_t max_iterations = fixed_number_optimization_runs + 1; // 1e2;  //TODO: hack for converge condition on set number of iterations
    bool converged = false;

    std::unique_ptr<Sampler> combinedSampler;
    std::unique_ptr<PretrainSampler> combinedPretrainSampler;

    int numThreads = 20;//18;//20;//20;//12;//14;
    omp_set_num_threads(numThreads);
    std::unique_ptr<Sampler> samplers[numThreads] = {};
    std::unique_ptr<PretrainSampler> pretrainSamplers[numThreads] = {};

    //For collecting energies during training and print to energiesTraining.csv for plotting.
    std::vector<double> KPreTraining{};
    std::vector<double> epochsPreTraining{};
    std::vector<double> energiesTraining{};
    std::vector<double> epochsTraining{};
    //std::vector<double> alphasTraining{};

    bool hasResetAdamAtEndOfAdiabaticChange = false;

    int max_iterations_pre_training = 500;

    if(!(parametersInputFile && parametersInputFile[0] != '\0'))
    {
        //Mandatory parameter, print error and exit
        cout << "No parameters input file given, exiting." << endl;
        return 1;
    }

    cout << "Reading parameters from file " << parametersInputFile << endl;
    params = csv_to_one_column(parametersInputFile);

    //Initialize Adam optimizer
    AdamOptimizer adamOptimizer(params.size(), fixed_learning_rate);

    size_t numberOfSteadyStateStepsPerEpoch = numberOfMetropolisSteps / MC_reduction / numThreads;
    size_t numberOfEquilibrationStepsPerEpoch = numberOfEquilibrationSteps/ MC_reduction / numThreads;
    cout << "Optimization run. Equilibrium steps: " << numberOfEquilibrationStepsPerEpoch << " Steady state steps: " << numberOfSteadyStateStepsPerEpoch << endl;
    for (size_t count = 0; count < fixed_number_optimization_runs; ++count)
    {
        double inter_strength = (count*2.0)/fixed_number_optimization_runs;
        inter_strength = std::min(1.0, inter_strength);
        cout << "Iteration " << BLUE << count+1 << RESET << " Adiabatically set interaction strength: " << BLUE << inter_strength << RESET << endl;
        auto combinedSampler = runParallellMonteCarloSimulation(parameter_seed, numberOfEquilibrationStepsPerEpoch, numberOfSteadyStateStepsPerEpoch, count, numThreads, stepLength, numberOfDimensions, numberOfParticles, params, omega, inter_strength, rbs_M, rbs_N, hard_core_size, samplers, false);
        params = optimizeAndUpdateParameters(params, combinedSampler, adamOptimizer, verbose, count);
        combinedSampler->printOutputToTerminalMini(verbose);

        //For plots with Python scripts
        auto energyEstimate = combinedSampler->getObservables()[0];
        energiesTraining.push_back(energyEstimate);
        epochsTraining.push_back(count);

        if(inter_strength==1.0 && !hasResetAdamAtEndOfAdiabaticChange)
        {
            //Imortand to make Adam not keep going up (by keeping momemtum) when ramp up of interaction stops.
            cout << "Resetting Adam optimizer" << endl;
            adamOptimizer.reset();
            hasResetAdamAtEndOfAdiabaticChange = true;
        }
    }

    size_t numberOfSteadyStateStepsBigComputation = numberOfMetropolisSteps / numThreads;
    size_t numberOfEquilibrationStepsBigComputation = numberOfEquilibrationSteps/ numThreads;
    cout << "Final large computation run. Equilibrium steps: " << numberOfEquilibrationStepsBigComputation << " Steady state steps: " << numberOfSteadyStateStepsBigComputation << endl;
    auto finalCombinedSampler = runParallellMonteCarloSimulation(parameter_seed, numberOfEquilibrationStepsBigComputation, numberOfSteadyStateStepsBigComputation, fixed_number_optimization_runs, numThreads, stepLength, numberOfDimensions, numberOfParticles, params, omega, inter_strength, rbs_M, rbs_N, hard_core_size, samplers, false);

    // Output information from the simulation
    finalCombinedSampler->printOutputToTerminal(false);
    //combinedSampler->printOutputToTerminal(verbose);

    string fileSuffix = to_string(numberOfDimensions) + "_" + to_string(numberOfParticles) + "_"+ to_string(rbs_N);
    string energies_plot_filename = "energies_plot_pure_" + fileSuffix  + ".csv";

    //Write energies to file, to be used by blocking method script.
    one_columns_to_csv("energies_nn2.csv", finalCombinedSampler->getEnergyArrayForBlocking(), ",", 0, 6);

    //one_columns_to_csv("energiesTraining.csv", energiesTraining, ",", 0, 6);
    two_columns_to_csv(energies_plot_filename, epochsTraining, energiesTraining, ",", 0, 6);

    return 0;
}
