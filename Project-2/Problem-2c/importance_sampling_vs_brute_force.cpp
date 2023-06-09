#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "omp.h"

#include "../../include/system.h"
#include "../../include/gaussianjastrow.h"
#include "../../include/harmonicoscillator.h"
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


std::unique_ptr<class MonteCarlo> createSolverFromArgument(string algoritm, std::unique_ptr<class Random> rng)
{
    if (algoritm=="METROPOLIS") return std::make_unique<Metropolis>(std::move(rng));
    if (algoritm=="METROPOLIS_HASTINGS") return std::make_unique<MetropolisHastings>(std::move(rng));
    cout << "Invalid type of algoritm for Monte Carlo, valid types are METROPOLIS and METROPOLIS_HASTINGS " << endl;
    exit(-1);
}

std::unique_ptr<class Hamiltonian> createHamiltonianFromArgument(string type, double omega, double beta)
{
    if (type=="HARMONIC") return std::make_unique<HarmonicOscillator>(omega);
    if (type=="HARMONIC_CYLINDRIC_INTERACTION") return std::make_unique<RepulsiveHamiltonianCyllindric>(omega, beta);
    cout << "Invalid type of Hamiltonian, valid types are HARMONIC and HARMONIC_CYLINDRIC_INTERACTION " << endl;
    exit(-1);
}

std::vector<std::unique_ptr<Particle>> createParticlesFromArgument(string type, double stepLength, double hardCoreSize, size_t numberOfDimensions, size_t numberOfParticles, Random &randomEngine)
{
    //Don't know if it's the prettiest choice to reuse parameter for Hamiltonian, but it makes sense considering how we tended to use it in Project 1.
    if (type=="HARMONIC") return setupRandomUniformInitialState(stepLength, numberOfDimensions, numberOfParticles, randomEngine);
    if (type=="HARMONIC_CYLINDRIC_INTERACTION") return setupRandomUniformInitialStateWithRepulsion(stepLength, hardCoreSize, numberOfDimensions, numberOfParticles, randomEngine);
    cout << "Invalid type of Hamiltonian, valid types are HARMONIC and HARMONIC_CYLINDRIC_INTERACTION " << endl;
    exit(-1);
}

int main(int argc, char **argv)
{
	// This program is for experimenting with learning rate and number of hidden layers.
	// We hard code number of particles to one. 
	size_t numberOfDimensions = argc > 1 ? stoi(argv[1]) : 3;
    size_t numberOfParticles = argc > 2 ? stoi(argv[2]) : 2;
    
	// Number of visible layers for the RBM
	size_t rbs_M = numberOfParticles*numberOfDimensions;
	
	// Number of hidden layers for the RBM
    size_t rbs_N = argc > 3 ? stoi(argv[3]) : 3;

	// Number of times the gradient descent will run. 
	// We don't know yet what tolerance is good to use, so start by just experimenting with this number and see what tolerance we may achieve. 
    size_t fixed_number_optimization_runs = argc > 4 ? stoi(argv[4]) : 50;
	
	// Variational parameters for the RBM
    std::vector<double> params{};
	
	//Start with all parameters as random values
	int parameter_seed = 2023;   //For now, pick a hardcoded seed, so we get the same random number generator every run, since our goal is to compare settings.
	double parameterGuessSpread = 0.1;  //Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.^M

	params = SimpleRBM::generateRandomParameterSet(rbs_M, rbs_N, parameter_seed, parameterGuessSpread);
   
    // We're experimenting with what learning rate works best.
    double fixed_learning_rate = argc > 5 ? stod(argv[5]) : 0.05;
	
	//Number of MCMC cycles for the large calculation after optimization
    size_t numberOfMetropolisSteps = argc > 6 ? stoi(argv[6]) : 1e6;
	
	//Type of Hamiltonian to use, ie interaction or not, and shape of potential. Ex: HARMONIC, HARMONIC_CYLINDRIC_INTERACTION
    auto hamiltonianChoice = argc > 7 ? argv[7] : "HARMONIC";

	//Algoritm for sampling. Ex: METROPOLIS, METROPOLIS_HASTINGS (brute force and importance samling respectively)
    auto algoritmChoice = argc > 8 ? argv[8] : "METROPOLIS";
	
    size_t numberOfEquilibrationSteps = numberOfMetropolisSteps / 10;
    double omega = 1.0;                                         // Oscillator frequency.
	
	size_t MC_reduction = 100;                                  // Number of MC steps to reduce by at intermediate steps

    double beta = 2.82843;                                      // Frequency ratio
    double hard_core_size = 0.0043 / sqrt(omega);               // Hard core size
	/*
	//Other parameters we used in project 1 that we might want to add back in.
	*/
    double stepLength = 0.1;                                    // Metropolis step length.
    bool verbose = true;                                        // Verbosity of output

    // Let's perform optimization here; Gradient descent to be used

    std::vector<double> learning_rate; // deduced automatically
    double parameter_tolerance = 1e-2;
    size_t max_iterations = fixed_number_optimization_runs + 1;//1e2;  //TODO: hack for converge condition on set number of iterations
    bool converged = false;

    std::unique_ptr<Sampler> combinedSampler;

    int numThreads = 8;
    omp_set_num_threads(numThreads);
    std::unique_ptr<Sampler> samplers[numThreads] = {};

    for (size_t count = 0; count < max_iterations; ++count)
    {
		converged = count==fixed_number_optimization_runs; //TODO: hack for converge condition on set number of iterations
		
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
            auto particles = createParticlesFromArgument(hamiltonianChoice, stepLength, hard_core_size, numberOfDimensions, numberOfParticles, *rng);

            // Construct a unique pointer to a new System
            system = std::make_unique<System>(
                // Construct unique_ptr to Hamiltonian
				createHamiltonianFromArgument(hamiltonianChoice, omega, beta),
                // Construct unique_ptr to wave function
                std::make_unique<SimpleRBM>(rbs_M, rbs_N, params, omega),
                // Construct unique_ptr to solver, and move rng
				createSolverFromArgument(algoritmChoice, std::move(rng)),
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

		//TODO: Code below contains a mess with commented out code, from tolerance test previously used.
		//As it stands right now it will always run the set number of optimization, and 
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
                //learning_rate[param_num] = 0.1/numberOfParticles;  //Purely on emprihical basis, we experience divergence problems with higher learning rate on large number of particles.
                learning_rate[param_num] = fixed_learning_rate;
                //cout << "Learning rate: " << learning_rate[param_num] << endl;
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
            //cout << "Predictions: ";
            //combinedSampler->printOutputToTerminal();
            cout << endl;
        }

        // Check if the parameter has converged (estimate as total parameter change < tolerance)

        double total_change = 0;
        bool everyCloseEnough = true;
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            if(fabs(learning_rate[param_num] * gradient[param_num]) > parameter_tolerance){
                everyCloseEnough = false;
            }
            total_change += fabs(learning_rate[param_num] * gradient[param_num]);
        }
        cout << "Tolerance " << parameter_tolerance << " Total change: " << total_change << endl;
    }
    // Output information from the simulation
    combinedSampler->printOutputToTerminal(verbose);

    //Write energies to file, to be used by blocking method script.
    one_columns_to_csv("energies.csv", combinedSampler->getEnergyArrayForBlocking(), ",", 0, 6);

    return 0;
}
