#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "omp.h"

#include <cmath>
#include <random>
#include <cstdlib>
#include <iomanip>

//#include <autodiff/forward/dual.hpp>
//#include <autodiff/forward/dual/eigen.hpp>

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
/*using namespace autodiff;
using namespace Eigen;
*/

int main3();

std::unique_ptr<class MonteCarlo> createSolverFromArgument(string algoritm, std::unique_ptr<class Random> rng)
{
    if (algoritm=="METROPOLIS") return std::make_unique<Metropolis>(std::move(rng));
    if (algoritm=="METROPOLIS_HASTINGS") return std::make_unique<MetropolisHastings>(std::move(rng));
    cout << "Invalid type of algoritm for Monte Carlo, valid types are METROPOLIS and METROPOLIS_HASTINGS " << endl;
    exit(-1);
}

std::unique_ptr<PretrainSampler> runPreTrainParallelMonteCarloSimulation(
    size_t count,
    size_t numberOfMetropolisSteps,
    size_t fixed_number_optimization_runs,
    size_t numThreads,
    double stepLength,
    size_t numberOfDimensions,
    size_t numberOfParticles,
    size_t rbs_M,
    size_t rbs_N,
    size_t numberOfEquilibrationSteps,
    double omega,
    double alpha,
    double beta,
    std::vector<double>& params,
    const std::string& algoritmChoice)
{
    unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();
    size_t numberOfMetropolisStepsPerGradientIteration = numberOfMetropolisSteps / fixed_number_optimization_runs;
    size_t numberOfEquilibrationStepsPerIteration = numberOfEquilibrationSteps/ fixed_number_optimization_runs;
    numberOfMetropolisStepsPerGradientIteration /= numThreads;
    numberOfEquilibrationStepsPerIteration  /= numThreads;

    std::unique_ptr<PretrainSampler> pretrainSamplers[numThreads] = {};

cout << "This round " << count << " of PRE-TRAINING gets " << numberOfMetropolisStepsPerGradientIteration << " MC steps, split on " << numThreads << " threads.";
        numberOfMetropolisStepsPerGradientIteration /= numThreads; // Split by number of threads.
        cout << " so " << numberOfMetropolisStepsPerGradientIteration << " per thread. " << endl ;

#pragma omp parallel shared(pretrainSamplers, count)
    {
        int thread_id = omp_get_thread_num();
        unsigned int my_seed = base_seed + thread_id;
        auto rng = std::make_unique<Random>(my_seed);

        auto particles = setupRandomUniformInitialState(stepLength,numberOfDimensions,numberOfParticles,*rng);

        auto system = std::make_unique<PretrainSystem>(
            std::make_unique<HarmonicOscillator>(omega),
            std::make_unique<PureNeuralNetworkWavefunction>(rbs_M, rbs_N, params, omega, alpha, beta, 0.0),
            std::make_unique<SimpleGaussian>(alpha),
            createSolverFromArgument(algoritmChoice, std::move(rng)),
            std::move(particles));

        auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
            stepLength,
            1000);
            //numberOfEquilibrationStepsPerIteration);//numberOfMetropolisStepsPerGradientIteration / numberOfEquilibrationSteps);

        pretrainSamplers[thread_id] = system->runMetropolisSteps(
            stepLength,
            10000);
            //numberOfMetropolisStepsPerGradientIteration);
    }

    return std::unique_ptr<PretrainSampler>(new PretrainSampler(pretrainSamplers, numThreads));
}

std::unique_ptr<Sampler> runNonInteractingParallelMonteCarloSimulation(
    size_t count,
    size_t numberOfMetropolisSteps,
    size_t fixed_number_optimization_runs,
    size_t numThreads,
    double stepLength,
    size_t numberOfDimensions,
    size_t numberOfParticles,
    size_t rbs_M,
    size_t rbs_N,
    size_t numberOfEquilibrationSteps,
    double omega,
    double alpha,
    double beta,
    std::vector<double>& params,
    const std::string& algoritmChoice)
{
    unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();
    size_t numberOfMetropolisStepsPerGradientIteration = numberOfMetropolisSteps / fixed_number_optimization_runs;
    size_t numberOfEquilibrationStepsPerIteration = numberOfEquilibrationSteps/ fixed_number_optimization_runs;
    numberOfMetropolisStepsPerGradientIteration /= numThreads;
    numberOfEquilibrationStepsPerIteration  /= numThreads;

    std::unique_ptr<Sampler> samplers[numThreads] = {};

cout << "This round " << count << " of TRAINING gets " << numberOfMetropolisStepsPerGradientIteration << " MC steps, split on " << numThreads << " threads.";
        numberOfMetropolisStepsPerGradientIteration /= numThreads; // Split by number of threads.
        cout << " so " << numberOfMetropolisStepsPerGradientIteration << " per thread. " << endl ;

#pragma omp parallel shared(samplers, count)
    {
        int thread_id = omp_get_thread_num();
        unsigned int my_seed = base_seed + thread_id;
        auto rng = std::make_unique<Random>(my_seed);

        auto particles = setupRandomUniformInitialState(stepLength,numberOfDimensions,numberOfParticles,*rng);

        auto system = std::make_unique<System>(
            std::make_unique<HarmonicOscillator>(omega),
            std::make_unique<PureNeuralNetworkWavefunction>(rbs_M, rbs_N, params, omega, alpha, beta, 0.0),
            //createSolverFromArgument(algoritmChoice, std::move(rng)),
            std::make_unique<Metropolis>(std::move(rng)),
            std::move(particles));

        auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
            stepLength,
            1000);
            //numberOfEquilibrationStepsPerIteration);//numberOfMetropolisStepsPerGradientIteration / numberOfEquilibrationSteps);

        samplers[thread_id] = system->runMetropolisSteps(
            stepLength,
            10000);
            //numberOfMetropolisStepsPerGradientIteration);
    }

    return std::unique_ptr<Sampler>(new Sampler(samplers, numThreads));
}

std::vector<double> optimizeParameters(std::vector<double>& params, std::unique_ptr<PretrainSampler>& combinedPretrainSampler, AdamOptimizer& adamOptimizer) {
    auto gradient = std::vector<double>(params.size());
    for (size_t param_num = 0; param_num < params.size(); ++param_num)
    {
        gradient[param_num] = combinedPretrainSampler->getObservables()[2 + param_num];
    }
    // Update the parameter using Adam optimization
    auto NewParams = adamOptimizer.adamOptimization(params, gradient);
    params = NewParams;

    combinedPretrainSampler->printOutputToTerminalMini(true);

    std::cout << "Num params: " << params.size() << " Parameters:" << std::endl;
    std::streamsize original_precision = std::cout.precision(); // Save original precision
    std::cout << std::setprecision(4) << std::fixed;
    for (int i = 0; i < params.size(); ++i) {
        std::cout << params[i] << " ";
    }
    std::cout.precision(original_precision); // Restore original precision

    std::cout << std::endl;
    auto energyEstimate = combinedPretrainSampler->getObservables()[0];
    std::cout << "Energy estimate: " << energyEstimate << std::endl;

    return NewParams;
}

std::vector<double> optimizeParameters(std::vector<double>& params, std::unique_ptr<Sampler>& combinedSampler, AdamOptimizer& adamOptimizer) {
    auto gradient = std::vector<double>(params.size());
    for (size_t param_num = 0; param_num < params.size(); ++param_num)
    {
        gradient[param_num] = combinedSampler->getObservables()[2 + param_num];
    }
    // Update the parameter using Adam optimization
    auto NewParams = adamOptimizer.adamOptimization(params, gradient);
    params = NewParams;

    combinedSampler->printOutputToTerminalMini(true);

    std::cout << "Num params: " << params.size() << " Parameters:" << std::endl;
    std::streamsize original_precision = std::cout.precision(); // Save original precision
    std::cout << std::setprecision(4) << std::fixed;
    for (int i = 0; i < params.size(); ++i) {
        std::cout << params[i] << " ";
    }
    std::cout.precision(original_precision); // Restore original precision

    std::cout << std::endl;
    auto energyEstimate = combinedSampler->getObservables()[0];
    std::cout << "Energy estimate: " << energyEstimate << std::endl;

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
    int parameter_seed = 2023;//111;//2023;         // For now, pick a hardcoded seed, so we get the same random number generator every run, since our goal is to compare settings.
    double parameterGuessSpread = 0.001; // Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.

    params = NeuralNetworkWavefunction::generateRandomParameterSet(rbs_M, rbs_N, parameter_seed, parameterGuessSpread);

double alpha = 0.5;//m_parameters[0]; // alpha is the first and only parameter for now.
double beta = 2.82843; // beta is the second parameter for now.
//double adiabaticFactor = 2 * (double)(count+1)/ (double)fixed_number_optimization_runs;
    //params.push_back(alpha);//No alpha in neural network now!!

    // We're experimenting with what learning rate works best.
    double fixed_learning_rate = argc > 5 ? stod(argv[5]) : 0.01;

    // Number of MCMC cycles for the large calculation after optimization
    size_t numberOfMetropolisSteps = argc > 6 ? stoi(argv[6]) : 1e6;

	//Type of Hamiltonian to use, ie interaction or not, and shape of potential. Ex: HARMONIC, HARMONIC_CYLINDRIC_INTERACTION
    auto hamiltonianChoice = argc > 7 ? argv[7] : "HARMONIC";

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

    int numThreads = 18;//20;//20;//12;//14;
    omp_set_num_threads(numThreads);
    std::unique_ptr<Sampler> samplers[numThreads] = {};
    std::unique_ptr<PretrainSampler> pretrainSamplers[numThreads] = {};

    //For collecting energies during training and print to energiesTraining.csv for plotting.
    std::vector<double> KPreTraining{};
    std::vector<double> epochsPreTraining{};
    std::vector<double> energiesTraining{};
    std::vector<double> epochsTraining{};
    //std::vector<double> alphasTraining{};

    //Initialize Adam optimizer
    AdamOptimizer adamOptimizerPretrain(params.size(), 0.01);
    AdamOptimizer adamOptimizer(params.size(), fixed_learning_rate);
    bool hasResetAdamAtEndOfAdiabaticChange = false;

    int max_iterations_pre_training = 500;
////// Ok, lets try get some pre-training going

    if(parametersInputFile && parametersInputFile[0] != '\0')
    //if(parametersInputFile)
    {
        cout << "Reading parameters from file " << parametersInputFile << endl;
        params = csv_to_one_column(parametersInputFile);
    }
    else
    {
        for (size_t count = 0; count < max_iterations_pre_training; ++count)
        {
            auto combinedPretrainSampler = runPreTrainParallelMonteCarloSimulation(count, numberOfMetropolisSteps, fixed_number_optimization_runs, numThreads, stepLength, numberOfDimensions, numberOfParticles, rbs_M, rbs_N, numberOfEquilibrationSteps, omega, alpha, beta, params, algoritmChoice);
            params = optimizeParameters(params, combinedPretrainSampler, adamOptimizerPretrain);
            KPreTraining.push_back(combinedPretrainSampler->getObservables()[0]);
            epochsPreTraining.push_back(count);
        }

        string fileSuffix = to_string(numberOfDimensions) + "_" + to_string(numberOfParticles) + "_"+ to_string(rbs_N);
        string paramFilename = "NNparams_pretrain_" + fileSuffix  + ".csv";
        string kFilename = "K_pretrain_" + fileSuffix + ".csv";
        one_columns_to_csv(paramFilename, params, ",", 0, 6);
        //one_columns_to_csv("NNparams1.csv", params, ",", 0, 6);
        two_columns_to_csv(kFilename, epochsPreTraining, KPreTraining, ",", 0, 6);
    }

/*
    //Do the same except for calling runNonInteractingParallelMonteCarloSimulation
    for(size_t count = 0; count < max_iterations_pre_training; ++count)
    {
        auto combinedPretrainSampler = runNonInteractingParallelMonteCarloSimulation(count, numberOfMetropolisSteps, fixed_number_optimization_runs, numThreads, stepLength, numberOfDimensions, numberOfParticles, rbs_M, rbs_N, numberOfEquilibrationSteps, omega, alpha, beta, params, algoritmChoice);
        params = optimizeParameters(params, combinedPretrainSampler, adamOptimizer);
    }

    one_columns_to_csv("NNparams2.csv", params, ",", 0, 6);
*/
    adamOptimizer.reset();
//////////////
    for (size_t count = 0; count < max_iterations; ++count)
    {
        converged = count == fixed_number_optimization_runs; // TODO: hack for converge condition on set number of iterations

        // Random number setup in the way recommended for parallell computing, at https://github.com/anderkve/FYS3150/blob/master/code_examples/random_number_generation/main_rng_in_class_omp.cpp
        //  Use the system clock to get a base seed
        unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();

double adiabaticFactorStart = 0.001;
double adiabaticFactor = count*count*adiabaticFactorStart;
adiabaticFactor = std::min(1.0, adiabaticFactor);
//adiabaticFactor = 1.0;//If we want to test without adiabatic change
cout << "Iteration " << BLUE << count+1 << RESET << " Adiabatic factor: " << BLUE << adiabaticFactor << RESET << endl;

        //size_t numberOfMetropolisStepsPerGradientIteration = numberOfMetropolisSteps / MC_reduction * (converged | count == max_iterations - 1 ? MC_reduction : 1);
        size_t numberOfMetropolisStepsPerGradientIteration = numberOfMetropolisSteps / fixed_number_optimization_runs;
        cout << "This round of optimization gets " << numberOfMetropolisStepsPerGradientIteration << " MC steps, split on " << numThreads << " threads.";
        numberOfMetropolisStepsPerGradientIteration /= numThreads; // Split by number of threads.
        cout << " so " << numberOfMetropolisStepsPerGradientIteration << " per thread. " ;
#pragma omp parallel shared(samplers, count) // Start parallel region.
        {
            int thread_id = omp_get_thread_num();

            // Seed the generator with a seed that is unique for this thread
            unsigned int my_seed = base_seed + thread_id;
            auto rng = std::make_unique<Random>(my_seed);


            //std::unique_ptr<Sampler> sampler;
            std::unique_ptr<System> system;

            // Initialize particles
            auto particles = setupRandomUniformInitialStateWithRepulsion(stepLength, hard_core_size, numberOfDimensions, numberOfParticles, *rng);
            /*
            auto particles = setupRandomUniformInitialState(
                stepLength,
                numberOfDimensions,
                numberOfParticles,
                *rng);
*/
            // Construct a unique pointer to a new System
            system = std::make_unique<System>(
                // Construct unique_ptr to Hamiltonian
                std::make_unique<CoulombHamiltonian>(omega, adiabaticFactor*inter_strength),
                // Construct unique_ptr to wave function
                std::make_unique<PureNeuralNetworkWavefunction>(rbs_M, rbs_N, params, omega, alpha, beta, adiabaticFactor),
                // Construct unique_ptr to solver, and move rng
                //std::make_unique<MetropolisHastings>(std::move(rng)),
                //std::make_unique<Metropolis>(std::move(rng)),
                createSolverFromArgument(algoritmChoice, std::move(rng)),
                // Move the vector of particles to system
                std::move(particles));
//cout << "numberOfMetropolisStepsPerGradientIteration is " << numberOfMetropolisStepsPerGradientIteration << endl;
            // Run steps to equilibrate particles
            auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
                stepLength,
                numberOfMetropolisStepsPerGradientIteration);

            // Run the Metropolis algorithm
            samplers[thread_id] = system->runMetropolisSteps(
                stepLength,
                numberOfMetropolisStepsPerGradientIteration);
        }
cout << "Finished parallel region" << endl;
        // Create a new Sampler object containing the average of all the others.
        //combinedSampler = std::unique_ptr<Sampler>(samplers[0]);//
        combinedSampler = std::unique_ptr<Sampler>(new Sampler(samplers, numThreads));
/*if (!samplers.empty()) {
    combinedSampler = samplers[0];
} else {
    // Handle the case where samplers is empty
}*/
        // TODO: Code below contains a mess with commented out code, from tolerance test previously used.
        // As it stands right now it will always run the set number of optimization, and
        if (converged)
            break;
//if(count%2==1){ //Temporary try to only update parameters every 2 step, to investigate how much MC spreads with parameters unchanged
        // Extract the gradient
        auto gradient = std::vector<double>(params.size());
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            gradient[param_num] = combinedSampler->getObservables()[2 + param_num];
        }
        //Try to adress the fact that the same learning rate is not good for alpha as for the neural network parameters.
        //double suppressAlphaChange = 0.1; // Set your value
        //gradient.back() *= suppressAlphaChange;

        //double alphaBefore = params[params.size()-1];
        // Update the parameter using Adam optimization
        auto NewParams = adamOptimizer.adamOptimization(params, gradient);
        double sum = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            double diff = fabs(NewParams[i] - params[i]);
            sum += diff * diff;
        }
        double meanSquareDifference = sum / params.size();
        params = NewParams;

        //cout << "Old alpha " << alphaBefore << " New alpha " << params[params.size()-1] << " Overridde alpha" << alphaBefore - 0.01 * gradient[params.size()-1] << endl;
        //Trying to override Adams for alpha with fixed learning rate to see if it helps.
        //params[params.size()-1] = alphaBefore - 0.01 * gradient[params.size()-1];
        //params[params.size()-1] = 0.5; //Whatever, fix it to 0.5, see what happens.
//}
        combinedSampler->printOutputToTerminalMini(verbose);

        cout << "Num params: " << params.size() << " Parameters:" << endl;
        std::streamsize original_precision = std::cout.precision(); // Save original precision
        std::cout << std::setprecision(4) << std::fixed;
        //cout << "Alpha: " << params[params.size()-1] << ", ";
        for (int i = 0; i < 8 && i < params.size(); ++i) {
            std::cout << params[i] << " ";
        }
        std::cout.precision(original_precision); // Restore original precision

        std::cout << std::endl;
        //cout << "Tolerance " << parameter_tolerance << " Adam MSE Total change: " << meanSquareDifference << endl;
        auto energyEstimate = combinedSampler->getObservables()[0];
        cout << "Energy estimate: " << energyEstimate << endl;
        energiesTraining.push_back(energyEstimate);
        epochsTraining.push_back(count);
        //alphasTraining.push_back(params[params.size()-1]);

        if(adiabaticFactor==1.0 && !hasResetAdamAtEndOfAdiabaticChange)
        {
            cout << "Resetting Adam optimizer" << endl;
            adamOptimizer.reset();
            hasResetAdamAtEndOfAdiabaticChange = true;
        }
    }
    // Output information from the simulation
    combinedSampler->printOutputToTerminal(verbose);

    //Write energies to file, to be used by blocking method script.
    //one_columns_to_csv("energies.csv", combinedSampler->getEnergyArrayForBlocking(), ",", 0, 6);

    //one_columns_to_csv("energiesTraining.csv", energiesTraining, ",", 0, 6);
    two_columns_to_csv("energiesTraining_pure.csv", epochsTraining, energiesTraining, ",", 0, 6);
    //two_columns_to_csv("alphasTraining2.csv", epochsTraining, alphasTraining, ",", 0, 6);

//    main3();

    return 0;
}
