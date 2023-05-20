#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cassert>

#include "../../include/system.h"
#include "../../include/hamiltonian_cyllindric_repulsive.h"
#include "../../include/initialstate.h"
#include "../../include/random.h"
#include "../../include/particle.h"
#include "../../include/sampler.h"
#include "../../include/rbm.h"

using namespace std;

const double closeEnoughTolerance = 0.001;
bool closeEnough(double x, double y)
{
    return fabs(x-y) < closeEnoughTolerance;
}

int main(int argc, char **argv)
{
    // Seed for the random number generator. 
	//By having same seed every time we can get known values to directly check for correctness in tests. 
    int seed = 2023;
    double parameterGuessSpread = 0.1;  //Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.

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

	size_t rbs_N = 1;//rbs_M;//1; //Only one hidden node is on the extreme small side in practical scenarios. rbs_N = rbs_M would have been more realistic.
	//However in a unit test setting it gives a nice small set of values for unit testing.
	//Also since M != N, we get non square matrix, and might uncover bugs related to matrix dimensionalities in matrix multiplications and such.

	cout << " ------------------------------ " << endl;
	
	auto rng = std::make_unique<Random>(seed);
	// Initialize particles
	auto particles = setupRandomUniformInitialStateWithRepulsion(stepLength, hard_core_size, numberOfDimensions, numberOfParticles, *rng);

	cout << " ------------------------------ " << endl;
	
	auto hamiltonian = std::make_unique<RepulsiveHamiltonianCyllindric>(omega, beta);


    //Start with all parameters as random values
    auto randomParameters = SimpleRBM::generateRandomParameterSet(rbs_M, rbs_N, seed, parameterGuessSpread);
	auto waveFunction = std::make_unique<SimpleRBM>(rbs_M, rbs_N, randomParameters);

	double lap = waveFunction->computeLocalLaplasian(particles);
	cout << " ------------------------------ " << endl;

	
	for (size_t i = 0; i < particles.size(); i++)
    {
        auto position = particles[i]->getPosition();
		cout << "Position particle " << i << ": " ;
		auto numDimensions = position.size();
        for (size_t j=0; j<numDimensions; j++)
        {
            cout << position[j] << " ";
        }
		cout << endl;
	}
	//cout << "First particle coordinates " << (particles[0]->getPosition()) << endl;
	double value = waveFunction->evaluate(particles);
	cout << " ------------------------------ " << endl;

	cout << "Wave function value " << value << endl;
	cout << "Wave function Laplacian " << lap << endl;

	cout << " ------------------------------ " << endl;

	auto qForce = waveFunction->computeQuantumForce(particles, 0);
    cout << "Quantum-force (size M*D=" << qForce.size() << "): ";
    for(int i=0; i<qForce.size(); i++){
      cout << qForce[i] << " " ;
    }
    cout << endl;
	cout << " ------------------------------ " << endl;

	auto logPsiDerivatives = waveFunction->computeLogPsiDerivativeOverParameters(particles);
    cout << "All derivatives w.r.t parameters, of ln psi (size M+N+M*N=" << logPsiDerivatives.size() << "): ";
    for(int i=0; i<logPsiDerivatives.size(); i++){
      cout << logPsiDerivatives[i] << " " ;
    }
    cout << endl;
	cout << " ------------------------------ " << endl;

    //Now run assert on some selected values. If failed, the program will print out which row in this source code file it happened.
    assert(closeEnough(value, 1.86413));
    assert(closeEnough(lap, -3.84772));
    assert(closeEnough(qForce[0], 0.100514));
    assert(closeEnough(logPsiDerivatives[0], -0.106325));  //One derivative related to a
    assert(closeEnough(logPsiDerivatives[rbs_M], 0.487879)); //One derivative related to b
    assert(closeEnough(logPsiDerivatives[logPsiDerivatives.size()-1], 0.135884)); //One derivative related to W

	cout << "All tests passed!" << endl;  //If we reach this line, no assert failed.

    return 0;
}
