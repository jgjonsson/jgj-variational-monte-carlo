#include <memory>
#include <cmath>
#include <cassert>
#include <numeric>


#include "../include/nn_wave_pure.h"
#include "../include/particle.h"
#include "../include/random.h"
#include "../include/neural_onelayer.h"

using namespace std;


PureNeuralNetworkWavefunction::PureNeuralNetworkWavefunction(size_t rbs_M, size_t rbs_N, std::vector<double> parameters, double omega, double alpha, double beta, double adiabaticFactor)
    : m_neuralNetwork(parameters, rbs_M, rbs_N)
{

    assert(rbs_M > 0);
    assert(rbs_N > 0);

    m_alpha = alpha;
    m_beta = beta;
    m_omega = omega;

    //TODO: Consider parameterizing this. However project spec says only look at sigma=1.0 so this is perhaps ok.
    m_sigmaSquared = 1.0;

    m_numberOfParameters = parameters.size();

    this->m_M = rbs_M;
    this->m_N = rbs_N;

    // Initialize m_neuralNetwork after popping the last element from parameters
    //m_neuralNetwork = std::make_unique<NeuralNetworkReverse>(parameters, rbs_M, rbs_N);
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
    return exp(m_neuralNetwork.feedForward(x));
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
    return m_neuralNetwork.laplacianOfLogarithmWrtInputs(xInputs);
}

double PureNeuralNetworkWavefunction::evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator, std::vector<std::unique_ptr<class Particle>> &particles_denominator)
{
    assert(particles_numerator.size() == particles_denominator.size());

    double jastrowNumerator = m_neuralNetwork.feedForward(flattenParticleCoordinatesToVector(particles_numerator, m_M));
    double jastrowDenominator = m_neuralNetwork.feedForward(flattenParticleCoordinatesToVector(particles_denominator, m_M));
    return exp(jastrowNumerator-jastrowDenominator);
}

std::vector<double> PureNeuralNetworkWavefunction::computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
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
