#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "omp.h"

#include <cmath>
#include <random>
#include <cstdlib>
#include <iomanip>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

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
#include "../../include/neural.h"
#include "../../include/nn_wave.h"
#include "../../include/adam.h"

using namespace std;
using namespace autodiff;
using namespace Eigen;


int main3();

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
    double parameterGuessSpread = 0.1; // Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.

    params = NeuralNetworkWavefunction::generateRandomParameterSet(rbs_M, rbs_N, parameter_seed, parameterGuessSpread);

    // We're experimenting with what learning rate works best.
    double fixed_learning_rate = argc > 5 ? stod(argv[5]) : 0.01;

    // Number of MCMC cycles for the large calculation after optimization
    size_t numberOfMetropolisSteps = argc > 6 ? stoi(argv[6]) : 1e6;

    size_t numberOfEquilibrationSteps = numberOfMetropolisSteps / 10;
    double omega = 1.0; // Oscillator frequency.

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

    int numThreads = 20;//12;//14;
    omp_set_num_threads(numThreads);
    std::unique_ptr<Sampler> samplers[numThreads] = {};

    //Initialize Adam optimizer
    AdamOptimizer adamOptimizer(params.size(), fixed_learning_rate);
    bool hasResetAdamAtEndOfAdiabaticChange = false;

    for (size_t count = 0; count < max_iterations; ++count)
    {
        converged = count == fixed_number_optimization_runs; // TODO: hack for converge condition on set number of iterations

        // Random number setup in the way recommended for parallell computing, at https://github.com/anderkve/FYS3150/blob/master/code_examples/random_number_generation/main_rng_in_class_omp.cpp
        //  Use the system clock to get a base seed
        unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();

double alpha = 0.5;//m_parameters[0]; // alpha is the first and only parameter for now.
double beta = 2.82843; // beta is the second parameter for now.
double adiabaticFactor = 2 * (double)(count+1)/ (double)fixed_number_optimization_runs;
adiabaticFactor = std::min(1.0, adiabaticFactor);

cout << "Iteration " << count+1 << " Adiabatic factor: " << adiabaticFactor << endl;

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
            auto particles = setupRandomUniformInitialState(
                stepLength,
                numberOfDimensions,
                numberOfParticles,
                *rng);

            // Construct a unique pointer to a new System
            system = std::make_unique<System>(
                // Construct unique_ptr to Hamiltonian
                std::make_unique<CoulombHamiltonian>(omega, adiabaticFactor*inter_strength),
                // Construct unique_ptr to wave function
                std::make_unique<NeuralNetworkWavefunction>(rbs_M, rbs_N, params, omega, alpha, beta, adiabaticFactor),
                // Construct unique_ptr to solver, and move rng
                //std::make_unique<MetropolisHastings>(std::move(rng)),
                std::make_unique<MetropolisHastings>(std::move(rng)),
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

        // Extract the gradient
        auto gradient = std::vector<double>(params.size());
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            gradient[param_num] = combinedSampler->getObservables()[2 + param_num];
        }

        // Update the parameter using Adam optimization
        auto NewParams = adamOptimizer.adamOptimization(params, gradient);
        double sum = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            double diff = fabs(NewParams[i] - params[i]);
            sum += diff * diff;
        }
        double meanSquareDifference = sum / params.size();
        params = NewParams;

        combinedSampler->printOutputToTerminalMini(verbose);

        std::streamsize original_precision = std::cout.precision(); // Save original precision
        for (const auto &param : params) {
            std::cout << std::setprecision(3) << std::fixed << param << " ";
        }
        std::cout.precision(original_precision); // Restore original precision

        std::cout << std::endl;
        cout << "Tolerance " << parameter_tolerance << " Adam MSE Total change: " << meanSquareDifference << endl;
        cout << "Energy estimate: " << combinedSampler->getObservables()[0] << endl;

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
    one_columns_to_csv("energies.csv", combinedSampler->getEnergyArrayForBlocking(), ",", 0, 6);

//    main3();

    return 0;
}

/*
int main3() {
    srand((unsigned) time(NULL)); // Initialize random seed

    int inputNodes = 4;
    int hiddenNodes = 4;

    std::vector<double> randNumbers;
    //int totalNumbers = 4 * 4 + 4 + 1 + 4 + 4 * 4 + 1; // Total number of weights and biases for the neural network
    int totalNumbers = inputNodes * hiddenNodes + hiddenNodes + 1 + inputNodes + hiddenNodes * inputNodes + 1; // Total number of weights and biases for the neural network
    //double targetOutput = 0.714;
    double targetOutput = 0.612;
    double learningRate = 0.0005;

    for(int i = 0; i < totalNumbers; i++) {
        double randNumber = static_cast<double>(rand()) / RAND_MAX / 1000;
        randNumbers.push_back(randNumber);
    }

    NeuralNetwork neuralNetwork(randNumbers, inputNodes, hiddenNodes);

    std::vector<double> inputs = {0.1, 0.2, 0.3, 0.4};


    for(int i = 0; i < 50000; i++) { // Training for 1000 iterations
        double output = neuralNetwork.feedForward(inputs);
        neuralNetwork.backpropagate(inputs, targetOutput, learningRate);
        if(i % 1000 == 0) { // Print the output every 100 iterations
            std::cout << "Iteration: " << i << " Output: " << output << std::endl;
        }
    }



    neuralNetwork.printParameters2();

    double output = neuralNetwork.feedForward(inputs);
    std::cout << "Output: " << output << std::endl;

    VectorXdual inputsDual = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<dual>();


    dual outputDual = neuralNetwork.feedForwardDual2(inputsDual);
    std::cout << "Output Dual: " << outputDual << std::endl;

    std::cout << "Before autodiff" << std::endl;
    std::function<VectorXdual(VectorXdual, VectorXdual)>  gradientFunction = neuralNetwork.getGradientFunction();
    std::cout << "After autodiff" << std::endl;
    VectorXdual theGradient = gradientFunction(neuralNetwork.parametersDual, inputsDual);

    std::cout << "After eval diff" << std::endl;
    //auto theGradient = neuralNetwork.getGradient(inputsDual);
    cout << "Gradient: " << theGradient.transpose() << endl;
//    neuralNetwork.printParameters();
//        std::cout << "Sweet: " << output << std::endl;


    return 0;
}

*/