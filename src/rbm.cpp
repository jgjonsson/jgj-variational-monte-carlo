#include <memory>
#include <cmath>
#include <cassert>

#include "../include/rbm.h"
#include "../include/particle.h"
#include "../include/random.h"

using namespace std;
using namespace arma;

SimpleRBM::SimpleRBM(size_t rbs_M, size_t rbs_N, Random &randomEngine)
{
    assert(rbs_M > 0);
    assert(rbs_N > 0);

    m_sigmaSquared = 0.0;
    m_omega = 0.0;

    //Number of parameters, M and N
    this->m_M = rbs_M;
    this->m_N = rbs_N;

    //Parameters for the wave function, initialize as vectors and matrices
    m_W.set_size(m_M, m_N);
    m_a.set_size(m_M);
    m_b.set_size(m_N);

    //Start with all parameters as random values
    for (size_t i = 0; i < m_M; i++){
        m_a(i) = randomEngine.nextDouble();
    }

    for (size_t i = 0; i < m_N; i++){
        m_b(i) = randomEngine.nextDouble();
    }

    for (size_t i = 0; i < m_M; i++){
        for (size_t j = 0; j < m_N; j++){
            m_W(i,j) = randomEngine.nextDouble();
        }
    }
    cout << "Initial a = " << m_a << endl;
    cout << "Initial b = " << m_b << endl;
    cout << "Initial W = " << m_W << endl;

    /*
    Don't actually use the normal parameter array from WaveFunction, so maybe we want to rethink this inheritence structure.
    m_numberOfParameters = 1;
    m_parameters.push_back(alpha);
    */
}

/** Helper-function to turn the P particles times D dimensions coordinates into a M=P*D vector
*/
//vec flattenParticleCoordinatesToVector(std::vector<class Particle*> particles, size_t m_M)
vec flattenParticleCoordinatesToVector(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M)
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

double SimpleRBM::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    vec x = flattenParticleCoordinatesToVector(particles, m_M);

    vec xMinusA = x - m_a;
    double psi1 = exp(-1/(2*m_sigmaSquared)*dot(xMinusA, xMinusA));

    vec xTimesW = m_W.t()*x; //Transpose is necessary to get the matching dimensions.
    vec psiFactors = 1 + exp(m_b + 1/m_sigmaSquared*(xTimesW));
    double psi2 = prod(psiFactors);

    //cout << "Evaluated wave function to " << psi1 <<"*" << psi2 << "=" << (psi1*psi2) << endl;

    return psi1*psi2;
}

double SimpleRBM::computeLocalLaplasian(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // The expression I got for a single laplasian is, in invariant form, follows:
    // (4 * alpha^2 * r_i^2 - 2 * alpha * NDIM)
    // so it takes to sum over all particles.
    double alpha = m_parameters[0];
    double sum_laplasian = 0.0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        double r2 = 0.0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); ++j)
            r2 += particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        sum_laplasian += 4 * alpha * alpha * r2 - 2 * alpha * particles[i]->getPosition().size();
    }
    return sum_laplasian;
}

double SimpleRBM::evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator, std::vector<std::unique_ptr<class Particle>> &particles_denominator)
{
    assert(particles_numerator.size() == particles_denominator.size());
    double ratio = 1.0;
    double alpha = m_parameters[0];

    for (size_t i = 0; i < particles_numerator.size(); i++)
    {
        double r2_numerator = 0.0;
        double r2_denominator = 0.0;
        for (size_t j = 0; j < particles_numerator[i]->getPosition().size(); j++)
        {
            r2_numerator += particles_numerator[i]->getPosition()[j] * particles_numerator[i]->getPosition()[j];
            r2_denominator += particles_denominator[i]->getPosition()[j] * particles_denominator[i]->getPosition()[j];
        }
        ratio *= exp(-alpha * (r2_numerator - r2_denominator));
    }
    return ratio;
}

std::vector<double> SimpleRBM::computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    double alpha = m_parameters[0];
    std::vector<double> quantumForce = std::vector<double>();
    std::vector<double> position = particles[particle_index]->getPosition();
    for (int j = 0; j < position.size(); j++)
    {
        quantumForce.push_back(-4 * alpha * position[j]);
    }

    return quantumForce;
}

std::vector<double> SimpleRBM::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double alpha = m_parameters[0];
    std::vector<double> logPsiDerivativeOverParameters = std::vector<double>();
    double sum = 0.0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        double r2 = 0.0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
            r2 += particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        sum += r2;
    }
    logPsiDerivativeOverParameters.push_back(-sum);
    return logPsiDerivativeOverParameters;
}