#include <memory>
#include <cmath>
#include <cassert>
#include <numeric>


#include "../include/nn_wave.h"
#include "../include/particle.h"
#include "../include/random.h"
#include "../include/neural_reverse.h"

using namespace std;


//TODO: 0.5 is nearly optimal, maybe exactly optimal for case 2 part 2D? However we should parametrize this as well later.
//double alpha = 0.5;//m_parameters[0]; // alpha is the first and only parameter for now.
//double m_beta = 2.82843; // beta is the second parameter for now.

NeuralNetworkWavefunction::NeuralNetworkWavefunction(size_t rbs_M, size_t rbs_N, std::vector<double> parameters, double omega, double alpha, double beta, double adiabaticFactor)
: m_neuralNetwork(parameters, rbs_M, rbs_N)
{
    assert(rbs_M > 0);
    assert(rbs_N > 0);

    m_alpha = alpha;
    m_beta = beta;
    m_adiabaticFactor = adiabaticFactor;
    m_omega = omega;

    //TODO: Consider parameterizing this. However project spec says only look at sigma=1.0 so this is perhaps ok.
    m_sigmaSquared = 1.0;

    m_numberOfParameters = parameters.size();

    this->m_M = rbs_M;
    this->m_N = rbs_N;

    m_numberOfParameters = parameters.size();
}

std::vector<double> NeuralNetworkWavefunction::flattenParticleCoordinatesToVector(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M)
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

double NeuralNetworkWavefunction::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{

    double psi = 1.0;

    for (size_t i = 0; i < particles.size(); i++)
    {
        // Let's support as many dimensions as we want.
        double r2 = 0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
            r2 += (j == 2 ? m_beta : 1.0) * particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        // spherical ansatz
        double g = exp(-m_alpha * r2);
//cout << "g for particle " << i << " is " << g << endl;
        // Trial wave function is product of g for all particles.
        psi = psi * g;
    }

    auto x = flattenParticleCoordinatesToVector(particles, m_M);
    double psiInteractionJastrow = m_neuralNetwork.feedForward(x);

    return psi * psiInteractionJastrow;//1*psi2;
}

/** Compute the double derivative of the trial wave function over trial wave function.
 *  This is based on an analythical derivation using product rule showing that is equivalent
 *  to the expression you see below.
 *  Which is involving gradient and laplacian of the logarithm of that wave function.
 *  @param particles Vector of particles.
 *  @return The local value of Laplasian.
 */
 //Ok, we try the exact same laplacian as in the gaussian wave function.
double NeuralNetworkWavefunction::computeLocalLaplasian(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // The expression I got for a single laplasian is, in invariant form, follows:
    // (4 * m_alpha^2 * r_i^2 - 2 * m_alpha * NDIM)
    // so it takes to sum over all particles.
    //double m_alpha = m_parameters[0];
    double sum_laplasian = 0.0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        double r2 = 0.0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); ++j){
//                                      r2 += particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
            r2 += (j == 2 ? m_beta : 1.0) * particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
            }

        sum_laplasian += 4 * m_alpha * m_alpha * r2 - 2 * m_alpha * particles[i]->getPosition().size();
    }

    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);

    double interactionLaplacian = m_neuralNetwork.calculateNumericalLaplacianWrtInput(xInputs);
    auto theGradientVector = m_neuralNetwork.getTheGradientVectorWrtInputs(xInputs);

    double interactionGradSquared = std::inner_product(theGradientVector.begin(), theGradientVector.end(), theGradientVector.begin(), 0.0);
//    cout << "Laplacian adding " << sum_laplasian << " with " << interactionLaplacian << " and " << interactionGradSquared << endl;
    //sum_laplasian += interactionLaplacian + interactionGradSquared;

    return sum_laplasian;
}

double NeuralNetworkWavefunction::evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator, std::vector<std::unique_ptr<class Particle>> &particles_denominator)
{
    assert(particles_numerator.size() == particles_denominator.size());

    double ratio = 1.0;

    // Regular gaussian part.
    for (size_t i = 0; i < particles_numerator.size(); i++)
    {
        double r2_numerator = 0.0;
        double r2_denominator = 0.0;
        for (size_t j = 0; j < particles_numerator[i]->getPosition().size(); j++)
        {
            r2_numerator += (j == 2 ? m_beta : 1.0) * particles_numerator[i]->getPosition()[j] * particles_numerator[i]->getPosition()[j];
            r2_denominator += (j == 2 ? m_beta : 1.0) * particles_denominator[i]->getPosition()[j] * particles_denominator[i]->getPosition()[j];
        }
        ratio *= exp(-m_alpha * (r2_numerator - r2_denominator));
    }

    double jastrowNumerator = m_neuralNetwork.feedForward(flattenParticleCoordinatesToVector(particles_numerator, m_M));
    double jastrowDenominator = m_neuralNetwork.feedForward(flattenParticleCoordinatesToVector(particles_denominator, m_M));
    /*This is verified the same
    ouble value1 = evaluate(particles_numerator);
    double value2 = evaluate(particles_denominator);
    cout << value1/value2 << " vs " << ratio * jastrowNumerator/jastrowDenominator << endl;*/
    return ratio * jastrowNumerator/jastrowDenominator;
}

/** Calculate the quantum force, defined by 2 * 1/Psi * grad(Psi)
 */
std::vector<double> NeuralNetworkWavefunction::computeQuantumForceOld(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    //vec x = flattenParticleCoordinatesToVector(particles, m_M);

    // I assume again that we do not arrive to forbidden states (r < r_hard_core), so I do not check for that.
    //double alpha = 0.5;
    std::vector<double> quantumForce = std::vector<double>();
    std::vector<double> position = particles[particle_index]->getPosition();
    for (int j = 0; j < position.size(); j++)
    {
        //Cylinder would need this:
        quantumForce.push_back(-4 * m_alpha * position[j] * (j == 2 ? m_beta : 1.0));
        //quantumForce.push_back(-4 * alpha * position[j]);
    }
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
        quantumForce[i - start] += interactionPartOfQuantumForce;
    }
    return quantumForce;
}


std::vector<double> NeuralNetworkWavefunction::computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    //vec x = flattenParticleCoordinatesToVector(particles, m_M);

    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);

    // I assume again that we do not arrive to forbidden states (r < r_hard_core), so I do not check for that.
    //double alpha = 0.5;
    std::vector<double> quantumForce = std::vector<double>();
    std::vector<double> position = particles[particle_index]->getPosition();
    for (int j = 0; j < position.size(); j++)
    {
        double qForceHarmonic = -4 * m_alpha * position[j] * (j == 2 ? m_beta : 1.0);
        int indexInX = particle_index * position.size() + j;
        double derivative = m_neuralNetwork.calculateNumericalDeriviateWrtInput(xInputs, indexInX);
        double qForceInteraction = 2 * derivative;
        quantumForce.push_back(qForceHarmonic + qForceInteraction);
    }
    return quantumForce;
}

std::vector<double> NeuralNetworkWavefunction::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);
    return m_neuralNetwork.getTheGradientVectorWrtParameters(xInputs);
/*
    VectorXdual xDual = flattenParticleCoordinatesToVectorAutoDiffFormat(particles, m_M);
    auto theGradient = m_neuralNetwork.getTheGradient(xDual);
    std::vector<double> logPsiDerivativeOverParameters(theGradient.size());
    std::transform(theGradient.begin(), theGradient.end(), logPsiDerivativeOverParameters.begin(), [](const dual& d) { return d.val; });
    return logPsiDerivativeOverParameters;*/
}