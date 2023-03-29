#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#include "omp.h"

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
    size_t numberOfDimensions = argc > 1 ? stoi(argv[1]) : 1;
    size_t numberOfParticles = argc > 2 ? stoi(argv[2]) : 1;
    size_t numberOfMetropolisSteps = argc > 3 ? stoi(argv[3]) : 1e6;
    size_t numberOfEquilibrationSteps = numberOfMetropolisSteps/10;
    double omega = 1.0;      // Oscillator frequency.
    double alpha = argc > 4 ? stod(argv[4]) : 0.5;      // Variational parameter.
    double stepLength = 0.1; // Metropolis step length.

    // Random number setup in the way recommended for parallell computing, at https://github.com/anderkve/FYS3150/blob/master/code_examples/random_number_generation/main_rng_in_class_omp.cpp
    //  Use the system clock to get a base seed
    unsigned int base_seed = chrono::system_clock::now().time_since_epoch().count();

    int numThreads = 4;

    omp_set_num_threads(numThreads);
    #pragma omp parallel // Start parallel region.
	{
		// Which thread is this?
		int thread_id = omp_get_thread_num();
        #pragma omp critical
        {
            cout << "I am thread number " << thread_id << endl;
        }

        // Seed the generator with a seed that is unique for this thread
        unsigned int my_seed = base_seed + thread_id;
        auto rng = std::make_unique<Random>(my_seed);
        //generator.seed(my_seed);

        // Initialize particles
        auto particles = setupRandomUniformInitialState(stepLength, numberOfDimensions, numberOfParticles, *rng);
        // Construct a unique pointer to a new System
        auto system = std::make_unique<System>(
            // Construct unique_ptr to Hamiltonian
            std::make_unique<HarmonicOscillator>(omega),
            // Construct unique_ptr to wave function
            std::make_unique<SimpleGaussian>(alpha),
            // Construct unique_ptr to solver, and move rng
            std::make_unique<MetropolisHastings>(std::move(rng)),
            // Move the vector of particles to system
            std::move(particles));

        // Run steps to equilibrate particles
        auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
            stepLength,
            numberOfEquilibrationSteps);

        // Run the Metropolis algorithm
        auto sampler = system->runMetropolisSteps(
            stepLength,
            numberOfMetropolisSteps);

        //For now simply print the whole summary for each sampler. Using critical to avoid printouts mixing up.
        //This could also be modified to accumulate every sampler into one, and make only one total result out of it.
        #pragma omp critical
        {
            // Output information from the simulation
            sampler->printOutputToTerminal(*system);
        }

    } // End entire parallel region

    return 0;
}
