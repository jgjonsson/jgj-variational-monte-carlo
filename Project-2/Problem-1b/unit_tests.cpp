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
    // Seed for the random number generator. 
	//By having same seed every time we can get known values to directly check for correctness in tests. 
    int seed = 2023;

    size_t numberOfDimensions = 2;
    size_t numberOfParticles = 2;
	
    double omega = 1.0;                                         // Oscillator frequency.
    double beta = 2.82843;                                      // Frequency ratio
    double hard_core_size = 0.0043 / sqrt(omega);               // Hard core size
    std::vector<double> params{argc > 4 ? stod(argv[4]) : 0.5}; // Variational parameter.
    double stepLength = 0.1;                                    // Metropolis step length.
    size_t MC_reduction = 100;                                  // Number of MC steps to reduce by at intermediate steps
    bool verbose = true;                                        // Verbosity of output


	size_t rbs_M = numberOfParticles*numberOfDimensions;
	size_t rbs_N = rbs_M; //This N is something to experiment with, but start by trying equal to M.

	cout << " ------------------------------ 0" << endl;
	
	auto rng = std::make_unique<Random>(seed);
	// Initialize particles
	auto particles = setupRandomUniformInitialStateWithRepulsion(stepLength, hard_core_size, numberOfDimensions, numberOfParticles, *rng);

	cout << " ------------------------------ 1" << endl;
	
	auto hamiltonian = std::make_unique<RepulsiveHamiltonianCyllindric>(omega, beta);
	auto waveFunction = std::make_unique<SimpleRBM>(rbs_M, rbs_N, *rng);
	
	cout << " ------------------------------ 2" << endl;
	double lap = waveFunction->computeLocalLaplasian(particles);
	cout << " ------------------------------ 3" << endl;
	
	for (size_t i = 0; i < particles.size(); i++)
    {
        auto position = particles[i]->getPosition();
		cout << "Position " ;
		auto numDimensions = position.size();
        for (size_t j=0; j<numDimensions; j++)
        {
            cout << position[j] << " ";
        }
		cout << endl;
	}
	//cout << "First particle coordinates " << (particles[0]->getPosition()) << endl;
	double value = waveFunction->evaluate(particles);
	cout << " ------------------------------ 4" << endl;
	
	cout << " ------------------------------ " << endl;
	cout << "Wave function value " << value << endl;
	cout << "Wave function Laplaian " << lap << endl;
	cout << " ------------------------------ " << endl;
	
    return 0;
}
