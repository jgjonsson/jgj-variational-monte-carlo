#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include "../include/system.h"
#include "../include/sampler.h"
#include "../include/particle.h"
#include "../include/hamiltonian.h"
#include "../include/wavefunction.h"

using std::cout;
using std::endl;

Sampler::Sampler(
    size_t numberOfParticles,
    size_t numberOfDimensions,
    size_t numberOfParameters,
    double stepLength,
    size_t numberOfMetropolisSteps)
{
    m_stepNumber = 0;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_numberOfParticles = numberOfParticles;
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfParameters = numberOfParameters;
    m_stepLength = stepLength;
    m_numberOfAcceptedSteps = 0;
    // El, El^2, El*d ln psi/d alpha, d ln psi/d alpha
    m_cumulatives = std::vector<double>(2 + 2*numberOfParameters, 0.0);
    // <E>, <(E-<E>)^2>, <d E/d alpha>
    m_observables = std::vector<double>(2 + numberOfParameters, 0.0);
}

void Sampler::sample(bool acceptedStep, System *system)
{
    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */
    auto localEnergy = system->computeLocalEnergy();
    //..
    auto particles = std::move(system->getParticles());
    auto gradients = system->getWaveFunction()->computeLogPsiDerivativeOverParameters(particles);
    // I am a terrible person
    system->getParticles() = std::move(particles);
    
    m_cumulatives[0] += localEnergy;
    m_cumulatives[1] += localEnergy * localEnergy;
    for (size_t i = 0; i < m_numberOfParameters; i++)
    {
        m_cumulatives[2 + i] += gradients[i];
        m_cumulatives[2 + m_numberOfParameters + i] += gradients[i] * localEnergy;
    }
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

void Sampler::printOutputToTerminal(System &system, bool verbose)
{
    auto pa = system.getWaveFunction()->getParameters();
    auto p = pa.size();
    if (!verbose)
    {
        for (const auto &x : pa)
            cout << x << " ";
        for (const auto &x : m_observables)
            cout << x << " ";
        return;
    }

    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberOfParticles << endl;
    cout << " Number of dimensions : " << m_numberOfDimensions << endl;
    cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << endl;
    cout << " Step length used : " << m_stepLength << endl;
    cout << " Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << p << endl;
    for (size_t i = 0; i < p; i++)
    {
        cout << " Parameter " << i + 1 << " : " << pa.at(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_observables[0] << endl;
    cout << " Standard deviation Energy : " << m_observables[1] << endl;
    cout << " Computed gradient : " << m_observables[2] << endl;
    cout << endl;
}

void Sampler::computeObservables()
{
    /* Compute the observables out of the sampled quantities.
     */
    m_observables[0] = m_cumulatives[0] / m_numberOfMetropolisSteps;
    m_observables[1] = sqrt((m_cumulatives[1] / m_numberOfMetropolisSteps - m_observables[0] * m_observables[0])/m_numberOfMetropolisSteps);
    for (size_t i = 0; i < m_numberOfParameters; i++)
    {
        m_observables[2 + i] = 2 * (m_cumulatives[2 + m_numberOfParameters + i] / m_numberOfMetropolisSteps - m_observables[0] * m_cumulatives[2 + i] / m_numberOfMetropolisSteps);
    }
}

void Sampler::reset()
{
    m_stepNumber = 0;
    m_cumulatives.clear();
    m_observables.clear();
    m_numberOfAcceptedSteps = 0;
}