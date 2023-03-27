#include <iostream>
#include <vector>
#include <memory>

#include "../../include/system.h"
#include "../../include/harmonicoscillator.h"
#include "../../include/initialstate.h"
#include "../../include/metropolis.h"
#include "../../include/random.h"
#include "../../include/particle.h"
#include "../../include/sampler.h"

using namespace std;

//Recreates SimpleGaussianInefficient class, but with minima reimplementations
class SimpleGaussianInefficient : public WaveFunction {
public:
    /// @brief Constructor for the SimpleGaussianInefficient class.
    /// @param alpha Variational parameter present in the exponent.
    SimpleGaussianInefficient(double alpha)
    {
        //assert(alpha >= 0);
        m_numberOfParameters = 1;
        m_parameters.push_back(alpha);
    }
    /// @brief Evaluate the trial wave function.
    /// @param particles Vector of particles.
    /// @return The value of the trial wave function.
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
    {
        double psi = 1.0;
        double alpha = m_parameters[0]; // alpha is the first and only parameter for now.

        for (size_t i = 0; i < particles.size(); i++)
        {
            // Let's support as many dimensions as we want.
            double r2 = 0;
            for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
                r2 += particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
            // spherical ansatz
            double g = exp(-alpha * r2);

            // Trial wave function is product of g for all particles.
            // f ignored for now, due to considering non interacting particles.
            psi = psi * g;
        }
        return psi;
    }
    // else implementations are taken default...
};


int main(int argc, char **argv)
{
    if (argc == 1) 
        cout << "Usage: " << argv[0] << " <number of dimensions = 1> <number of particles = 1> <number of Metropolis steps = 1e6> <alpha=0.5>" << endl;

    // Seed for the random number generator
    int seed = 2023;

    size_t numberOfDimensions = argc > 1 ? stoi(argv[1]) : 1;
    size_t numberOfParticles = argc > 2 ? stoi(argv[2]) : 1;
    size_t numberOfMetropolisSteps = argc > 3 ? stoi(argv[3]) : 1e6;
    size_t numberOfEquilibrationSteps = numberOfMetropolisSteps/10;
    double omega = 1.0;      // Oscillator frequency.
    double alpha = argc > 4 ? stod(argv[4]) : 0.5;      // Variational parameter.
    double stepLength = 0.1; // Metropolis step length.

    // The random engine can also be built without a seed
    auto rng = std::make_unique<Random>(seed);
    // Initialize particles
    auto particles = setupRandomUniformInitialState(stepLength, numberOfDimensions, numberOfParticles, *rng);
    // Construct a unique pointer to a new System
    auto system = std::make_unique<System>(
        // Construct unique_ptr to Hamiltonian
        std::make_unique<HarmonicOscillator>(omega),
        // Construct unique_ptr to wave function
        std::make_unique<SimpleGaussianInefficient>(alpha),
        // Construct unique_ptr to solver, and move rng
        std::make_unique<Metropolis>(std::move(rng)),
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

    // Output information from the simulation
    sampler->printOutputToTerminal(*system);

    return 0;
}
