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
    double stepLength,
    size_t numberOfMetropolisSteps)
{
    m_stepNumber = 0;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_numberOfParticles = numberOfParticles;
    m_numberOfDimensions = numberOfDimensions;
    m_energy = 0;
    m_cumulativeEnergy = 0;
    m_cumulativeEnergySquare = 0;
    m_stepLength = stepLength;
    m_numberOfAcceptedSteps = 0;
}

void Sampler::sample(bool acceptedStep, System *system)
{
    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */
    auto localEnergy = system->computeLocalEnergy();
    m_cumulativeEnergy += localEnergy;
    m_cumulativeEnergySquare += localEnergy*localEnergy;
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

void Sampler::printOutputToTerminal(System &system, bool verbose)
{
    auto pa = system.getWaveFunctionParameters();
    auto p = pa.size();
    if (!verbose)
    {
        for (const auto &x : pa)
            cout << x << " ";
        cout << m_energy << endl;
        return;
    }

    double energy_standard_dev = sqrt(m_energySquare-m_energy*m_energy) / sqrt(m_numberOfMetropolisSteps);

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
    cout << " Energy : " << m_energy << endl;
    cout << " Standard deviation Energy : " << energy_standard_dev << endl;
    cout << endl;
}

void Sampler::computeAverages()
{
    /* Compute the averages of the sampled quantities.
     */
    m_energy = m_cumulativeEnergy / m_numberOfMetropolisSteps;
    m_energySquare = m_cumulativeEnergySquare / m_numberOfMetropolisSteps;
}
