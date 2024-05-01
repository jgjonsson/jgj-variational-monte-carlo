#include <memory>
#include <cmath>
#include <cassert>

#include "../include/nn_wave.h"
#include "../include/particle.h"
#include "../include/random.h"
#include "../include/neural.h"

using namespace std;
using namespace arma;

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

    //NeuralNetwork neuralNetwork(parameters, inputNodes, hiddenNodes);
    //NeuralNetwork neuralNetwork(parameters, rbs_M, rbs_N);
    // Initialize m_neuralNetwork here after the assertions
    //m_neuralNetwork = NeuralNetwork(parameters, rbs_M, rbs_N);

    m_numberOfParameters = parameters.size();


cout << "Satte upp et neural network med " << rbs_M << " och " << rbs_N << " noder, " << m_numberOfParameters << " params." << endl;

    //Number of parameters, M and N
    this->m_M = rbs_M;
    this->m_N = rbs_N;
/*
    //Parameters for the wave function, initialize as vectors and matrices
    m_W.set_size(m_M, m_N);
    m_a.set_size(m_M);
    m_b.set_size(m_N);

    insertParameters(parameters);*/
/*
    cout << "Initial a = " << m_a << endl;
    cout << "Initial b = " << m_b << endl;
    cout << "Initial W = " << m_W << endl;
*/
    m_numberOfParameters = parameters.size();
}

/** Helper-function to turn the P particles times D dimensions coordinates into a M=P*D vector
*/
//vec flattenParticleCoordinatesToVector(std::vector<class Particle*> particles, size_t m_M)
/*vec NeuralNetworkWavefunction::flattenParticleCoordinatesToVector(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M)
{
    vec x(m_M);
    for (size_t i = 0; i < particles.size(); i++)
    {
        auto position = particles[i]->getPosition();
        auto numDimensions = position.size();
        for (size_t j=0; j<numDimensions; j++)
        {
            x(i*numDimensions + j) = position[j];
        }
    }
    return x;
}
*/
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

VectorXdual NeuralNetworkWavefunction::flattenParticleCoordinatesToVectorAutoDiffFormat(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M)
//VectorXdual flattenParticleCoordinatesToVectorAutoDiffFormat(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M)
{
    VectorXdual x(m_M);
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
//    cout <<"Psi interaction jastrow is " << psiInteractionJastrow << endl;
/*
    vec xMinusA = x - m_a;
    double psi1 = exp(-1/(2*m_sigmaSquared)*dot(xMinusA, xMinusA));

    vec xTimesW = m_W.t()*x; //Transpose is necessary to get the matching dimensions.
    vec psiFactors = 1 + exp(m_b + 1/m_sigmaSquared*(xTimesW));
    double psi2 = prod(psiFactors);
*/
    //cout << "Evaluated wave function to " << psi1 <<"*" << psi2 << "=" << (psi1*psi2) << endl;

//cout << "Returning " << psi * m_adiabaticFactor*psiInteractionJastrow << endl;
    return psi * m_adiabaticFactor*psiInteractionJastrow;//1*psi2;
}
/*
double NeuralNetworkWavefunction::gradientSquaredOfLnWaveFunction(vec x)
{
    vec sigmoid(m_N);
    vec gradientLnPsi(m_M);

    sigmoid = 1/(1 + exp(-(m_b + 1/m_sigmaSquared*(m_W.t()*x))));

    gradientLnPsi = 1/m_sigmaSquared*(m_a - x + m_W*sigmoid);

    return dot(gradientLnPsi, gradientLnPsi);
}

double NeuralNetworkWavefunction::laplacianOfLnWaveFunction(vec x)
{
    cout << "Trying to compute the LAPLACIAN" << endl;
    vec sigmoidParameter = (m_b + 1/m_sigmaSquared*(m_W.t()*x));
    vec sigmoid = 1/(1 + exp(-sigmoidParameter));
    vec sigmoidNegative = 1/(1 + exp(sigmoidParameter));
    vec sigmoidTimesSigmoidNegative = sigmoid%sigmoidNegative;  //Elementwise multiplication to obtain all S(bj+...)S(-bj-...) terms.
    vec termsLaplacianLnPsi = -1/m_sigmaSquared + 1/(m_sigmaSquared*m_sigmaSquared)*(square(m_W)*sigmoidTimesSigmoidNegative);
    return sum(termsLaplacianLnPsi);
}
*/
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
        for (size_t j = 0; j < particles[i]->getPosition().size(); ++j)
            r2 += particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        sum_laplasian += 4 * m_alpha * m_alpha * r2 - 2 * m_alpha * particles[i]->getPosition().size();
    }
    return sum_laplasian;
}
 /*
double NeuralNetworkWavefunction::computeLocalLaplasian(std::vector<std::unique_ptr<class Particle>> &particles)
{

    //TODO: return symbolic derivative of ln(psi) after we have joined the neural network with gaussian trial wave function.
    //Alternatively try to compute the laplacian of the wave function by autodiff.
    return 0.0;
    / *vec x = flattenParticleCoordinatesToVector(particles, m_M);
    return gradientSquaredOfLnWaveFunction(x) + laplacianOfLnWaveFunction(x);
    * /
}*/

double NeuralNetworkWavefunction::evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator, std::vector<std::unique_ptr<class Particle>> &particles_denominator)
{
    assert(particles_numerator.size() == particles_denominator.size());

    double value1 = evaluate(particles_numerator);
    double value2 = evaluate(particles_denominator);
//    cout << "Values are " << value1 << " and " << value2 << endl;
    //TODO: Shall we really square this?
    double jastrowRatio = (value1/value2)*(value1/value2);

//cout << "ratio is " << jastrowRatio << endl;
//exit(0);
    return jastrowRatio;
}

/** Calculate the quantum force, defined by 2 * 1/Psi * grad(Psi)
 */
std::vector<double> NeuralNetworkWavefunction::computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    cout << "Trying to compute the QUANTUM FORCE" << endl;
    vec x = flattenParticleCoordinatesToVector(particles, m_M);
    /*
    vec sigmoid(m_N);
    vec gradientLnPsi(m_M);

    sigmoid = 1/(1 + exp(-(m_b + 1/m_sigmaSquared*(m_W.t()*x))));

    gradientLnPsi = 1/m_sigmaSquared*(m_a - x + m_W*sigmoid);
    vec quantumForceVector = 2 * gradientLnPsi;
    std::vector<double> quantumForce = arma::conv_to < std::vector<double> >::from(quantumForceVector);

    //Pick out the quantum force for only the particle of requested index. From subset of the total quantum force vector,
    size_t dimensions = particles[0]->getNumberOfDimensions();
    size_t firstIndex = particle_index*dimensions;
    size_t lastIndex = firstIndex + dimensions;// - 1;
    std::vector<double> quantumForceOneSingleParticle(quantumForce.begin()+firstIndex, quantumForce.begin()+lastIndex);
*/
    //return quantumForceOneSingleParticle;
        std::vector<double> dummyValue = {0.0, 0.0, 0.0};
        return dummyValue;
}
