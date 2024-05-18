#include <memory>
#include <cmath>
#include <cassert>
#include <numeric>


#include "../include/nn_wave_pure.h"
#include "../include/particle.h"
#include "../include/random.h"
#include "../include/neural_reverse.h"

using namespace std;




PureNeuralNetworkWavefunction::PureNeuralNetworkWavefunction(size_t rbs_M, size_t rbs_N, std::vector<double> parameters, double omega, double alpha, double beta, double adiabaticFactor)
    : m_neuralNetwork(parameters, rbs_M, rbs_N)
{

    assert(rbs_M > 0);
    assert(rbs_N > 0);

    //m_alpha = parameters.back(); // Store the last element in m_alpha
    //parameters.pop_back(); // Remove the last element from the vector

    m_alpha = alpha;
    m_beta = beta;
//    m_adiabaticFactor = adiabaticFactor;
    m_omega = omega;

    //TODO: Consider parameterizing this. However project spec says only look at sigma=1.0 so this is perhaps ok.
    m_sigmaSquared = 1.0;

    m_numberOfParameters = parameters.size();

    this->m_M = rbs_M;
    this->m_N = rbs_N;

    // Initialize m_neuralNetwork after popping the last element from parameters
    //m_neuralNetwork = std::make_unique<NeuralNetworkReverse>(parameters, rbs_M, rbs_N);
}
/*
double PureNeuralNetworkWavefunction::ratioToTrainingGaussian_A(std::vector<std::unique_ptr<class Particle>> &particles)
{
    //Calculate Psi_train -
    double psiTrain = 1.0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        //Print particles position, in all dimensions
        cout << "Particle " << i << " position: ";
        for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
        {
            cout << particles[i]->getPosition()[j] << " ";
        }

        // Let's support as many dimensions as we want.
        double r2 = 0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
            r2 += (j == 2 ? m_beta : 1.0) * particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        // spherical ansatz
        double g = exp(-m_alpha * r2);
        // Trial wave function is product of g for all particles.
        psiTrain = psiTrain * g;
        cout << " g = " << g << " r2 = " << r2 << " psiTrain = " << psiTrain << endl;
    }

    //Calculate Psi of the neural network
    auto x = flattenParticleCoordinatesToVector(particles, m_M);
    double psiNN = exp(m_neuralNetwork.feedForward(x));
cout << "Psi_train: " << psiTrain << " Psi_NN: " << psiNN << " Return value: " << psiTrain / psiNN << endl;
//exit(-1);
    //Calculate A = Psi_train / Psi_NN used in eqs (6),(7) in Saito's article.
    return psiTrain / psiNN;
}
*/
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
/*
    double interactionLaplacian = m_neuralNetwork.calculateNumericalLaplacianWrtInput(xInputs);
    auto theGradientVector = m_neuralNetwork.getTheGradientVectorWrtInputs(xInputs);

    double interactionGradSquared = std::inner_product(theGradientVector.begin(), theGradientVector.end(), theGradientVector.begin(), 0.0);
*/
    double laplacianOfLogarithmWrtInputs = m_neuralNetwork.laplacianOfLogarithmWrtInputs(xInputs);
    return laplacianOfLogarithmWrtInputs;
    /*
    auto xInputs = flattenParticleCoordinatesToVector(particles, m_M);

    double interactionLaplacian = m_neuralNetwork.calculateNumericalLaplacianWrtInput(xInputs);
    auto theGradientVector = m_neuralNetwork.getTheGradientVectorWrtInputs(xInputs);

    double interactionGradSquared = std::inner_product(theGradientVector.begin(), theGradientVector.end(), theGradientVector.begin(), 0.0);
//    cout << "Laplacian adding " << sum_laplasian << " with " << interactionLaplacian << " and " << interactionGradSquared << endl;
    return interactionLaplacian + interactionGradSquared;*/
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