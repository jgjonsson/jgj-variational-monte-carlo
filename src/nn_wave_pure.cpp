#include <memory>
#include <cmath>
#include <cassert>
#include <numeric>


#include "../include/nn_wave_pure.h"
#include "../include/particle.h"
#include "../include/random.h"
#include "../include/neural_reverse.h"

using namespace std;


//TODO: 0.5 is nearly optimal, maybe exactly optimal for case 2 part 2D? However we should parametrize this as well later.
//double alpha = 0.5;//m_parameters[0]; // alpha is the first and only parameter for now.
//double m_beta = 2.82843; // beta is the second parameter for now.

PureNeuralNetworkWavefunction::PureNeuralNetworkWavefunction(size_t rbs_M, size_t rbs_N, std::vector<double> parameters, double omega, double alpha, double beta, double adiabaticFactor)
: m_neuralNetwork(parameters, rbs_M, rbs_N)
{
    assert(rbs_M > 0);
    assert(rbs_N > 0);
/*
    m_alpha = alpha;
    m_beta = beta;
    m_adiabaticFactor = adiabaticFactor;
    m_omega = omega;
*/
    //TODO: Consider parameterizing this. However project spec says only look at sigma=1.0 so this is perhaps ok.
    m_sigmaSquared = 1.0;

    m_numberOfParameters = parameters.size();

    this->m_M = rbs_M;
    this->m_N = rbs_N;

    m_numberOfParameters = parameters.size();
}

std::vector<double> PureNeuralNetworkWavefunction::flattenParticleCoordinatesToVector(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M)
{
    std::vector<double> x(m_M);
    for (size_t i = 0; i < particles.size(); i++)
    {
        auto position = particles[i]->getPosition();
        auto numDimensions = position.size();
        for (size_t j=0; j<numDimensions; j++)
        {
            x[i*numDimensions + j] = position[j];
        }
    }
    return x;
}

double PureNeuralNetworkWavefunction::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    auto x = flattenParticleCoordinatesToVector(particles, m_M);
    return m_neuralNetwork.feedForward(x);
}

/** Compute the double derivative of the trial wave function over trial wave function.
 *  This is based on an analythical derivation using product rule showing that is equivalent
 *  to the expression you see below.
 *  Which is involving gradient and laplacian of the logarithm of that wave function.
 *  @param particles Vector of particles.
 *  @return The local value of Laplasian.
 */
 //Ok, we try the exact same laplacian as in the gaussian wave function.
double PureNeuralNetworkWavefunction::computeLocalLaplasian(std::vector<std::unique_ptr<class Particle>> &particles)
{
    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);

    double interactionLaplacian = m_neuralNetwork.calculateNumericalLaplacianWrtInput(xInputs);
    auto theGradientVector = m_neuralNetwork.getTheGradientVectorWrtInputs(xInputs);

    double interactionGradSquared = std::inner_product(theGradientVector.begin(), theGradientVector.end(), theGradientVector.begin(), 0.0);
//    cout << "Laplacian adding " << sum_laplasian << " with " << interactionLaplacian << " and " << interactionGradSquared << endl;
    return interactionLaplacian + interactionGradSquared;
}

double PureNeuralNetworkWavefunction::evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator, std::vector<std::unique_ptr<class Particle>> &particles_denominator)
{
    assert(particles_numerator.size() == particles_denominator.size());

    double jastrowNumerator = m_neuralNetwork.feedForward(flattenParticleCoordinatesToVector(particles_numerator, m_M));
    double jastrowDenominator = m_neuralNetwork.feedForward(flattenParticleCoordinatesToVector(particles_denominator, m_M));
    /*This is verified the same
    ouble value1 = evaluate(particles_numerator);
    double value2 = evaluate(particles_denominator);
    cout << value1/value2 << " vs " << ratio * jastrowNumerator/jastrowDenominator << endl;*/
    return jastrowNumerator/jastrowDenominator;
}

/** Calculate the quantum force, defined by 2 * 1/Psi * grad(Psi)
 */
std::vector<double> PureNeuralNetworkWavefunction::computeQuantumForceOld(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    //vec x = flattenParticleCoordinatesToVector(particles, m_M);

    // I assume again that we do not arrive to forbidden states (r < r_hard_core), so I do not check for that.
    std::vector<double> quantumForce = std::vector<double>();
    std::vector<double> position = particles[particle_index]->getPosition();

    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);
    //VectorXdual xDual = flattenParticleCoordinatesToVectorAutoDiffFormat(particles, m_M);
    auto theGradientVector = m_neuralNetwork.getTheGradientVectorWrtInputs(xInputs);
    //auto theGradientVector = transformVectorXdualToVector(theGradient);

    //auto position = particles[0]->getPosition();
    auto numDimensions = position.size();
    //std::vector<double> result(quantumForce.size());
    size_t start = particle_index * numDimensions;
    size_t end = start + numDimensions;

    for(size_t i = start; i < end; i++) {
        auto interactionPartOfQuantumForce = 2 * theGradientVector[i];
        quantumForce.push_back(interactionPartOfQuantumForce);
    }
    return quantumForce;
}


std::vector<double> PureNeuralNetworkWavefunction::computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    //vec x = flattenParticleCoordinatesToVector(particles, m_M);

    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);

    std::vector<double> quantumForce = std::vector<double>();
    std::vector<double> position = particles[particle_index]->getPosition();
    for (int j = 0; j < position.size(); j++)
    {
        int indexInX = particle_index * position.size() + j;
        double derivative = m_neuralNetwork.calculateNumericalDeriviateWrtInput(xInputs, indexInX);
        double qForceInteraction = 2 * derivative;
        quantumForce.push_back(qForceInteraction);
    }
    return quantumForce;
}

std::vector<double> PureNeuralNetworkWavefunction::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);
    return m_neuralNetwork.getTheGradientVectorWrtParameters(xInputs);
}