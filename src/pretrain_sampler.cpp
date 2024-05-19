#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include "../include/pretrain_system.h"
#include "../include/sampler.h"
#include "../include/pretrain_sampler.h"
#include "../include/particle.h"
#include "../include/hamiltonian.h"
#include "../include/wavefunction.h"

using std::cout;
using std::endl;

// Define some ANSI escape codes for colors
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

//This just needs to be a reasonable number to pick a fraction of samples for error analysis.
//Making it a fixed parameter here. 100 was suggested in lecture.
const size_t howOftenStoreSampleForBlocking = 100;

PretrainSampler::PretrainSampler(
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
    m_cumulatives = std::vector<double>(2 + 2 * numberOfParameters +3, 0.0);
    // <E>, <(E-<E>)^2>, <d E/d alpha>
    m_observables = std::vector<double>(2 + numberOfParameters+3, 0.0);
}

/**
 * Constructur that combines a number of other PretrainSampler object, and creates a new one that is the average of all the others.
 */
PretrainSampler::PretrainSampler(std::unique_ptr<PretrainSampler> *samplers, int numberSamplers)
{
    m_stepNumber = samplers[0]->m_stepNumber;
    m_numberOfMetropolisSteps = samplers[0]->m_numberOfMetropolisSteps * numberSamplers;
    m_numberOfParticles = samplers[0]->m_numberOfParticles;
    m_numberOfDimensions = samplers[0]->m_numberOfDimensions;
    m_numberOfParameters = samplers[0]->m_numberOfParameters;
    m_stepLength = samplers[0]->m_stepLength;
    m_numberOfAcceptedSteps = 0;
    // El, El^2, El*d ln psi/d alpha, d ln psi/d alpha
    m_cumulatives = std::vector<double>(2 + 2 * m_numberOfParameters + 3, 0.0);
    // <E>, <(E-<E>)^2>, <d E/d alpha>
    m_observables = std::vector<double>(2 + m_numberOfParameters + 3, 0.0);

    energy_array_for_blocking = std::vector<double>();

    for (int i = 0; i < numberSamplers; i++)
    {
        for (int j = 0; j < 2 + 2 * m_numberOfParameters+3; j++)
        {
            m_cumulatives[j] += samplers[i]->m_cumulatives[j] / numberSamplers;
        }
        for (int j = 0; j < 2 + m_numberOfParameters+3; j++)
        {
            m_observables[j] += samplers[i]->m_observables[j] / numberSamplers;
        }
        m_numberOfAcceptedSteps += samplers[i]->m_numberOfAcceptedSteps;

        auto p = samplers[i]->m_wavefunction_parameters.size();
        m_wavefunction_parameters = samplers[0]->m_wavefunction_parameters;
        for (size_t j = 1; j < p; j++)
        {
            m_wavefunction_parameters[j] += samplers[i]->m_wavefunction_parameters[j] / numberSamplers;
        }

        //Appending the vectors of energies from all the ohter samplers.
        energy_array_for_blocking.insert(std::end(energy_array_for_blocking), std::begin(samplers[i]->energy_array_for_blocking), std::end(samplers[i]->energy_array_for_blocking));
    }
    // cout << "Previous two energies " << samplers[0]->m_observables[0] << " and " << samplers[1]->m_observables[0] << " avareged to  " << m_observables[0]  << endl;
}

void PretrainSampler::sample(bool acceptedStep, PretrainSystem *system)
{
    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */
    auto localEnergy = system->computeLocalEnergy();
    //..

    //The A in eqs. (6) and (7) in the Saito's article.
    double A = system->getRationToTrainTargetWaveFunction();
    auto particles = std::move(system->getParticles());
    auto gradients = system->getWaveFunction()->computeLogPsiDerivativeOverParameters(particles);

    system->setParticles(std::move(particles));

    m_cumulatives[0] += fabs(A);
    m_cumulatives[1] += A*A;
    for (size_t i = 0; i < m_numberOfParameters; i++)
    {
        m_cumulatives[2 + i] += gradients[i];  //The O_W in Saito's article
        m_cumulatives[2 + m_numberOfParameters + i] += gradients[i] * A; //The A*O_W in Saito's article
    }
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
//    cout << "Step number: " << m_stepNumber << " Local energy: " << localEnergy << " A: " << A << endl;
    //append localEnergy, localEnergy^2 and A last in m_cumulatives
    m_cumulatives[2 + m_numberOfParameters*2 + 0] += localEnergy;
    m_cumulatives[2 + m_numberOfParameters*2 + 1] += localEnergy*localEnergy;
    m_cumulatives[2 + m_numberOfParameters*2 + 2] += fabs(A);
//cout << "Nice" << endl;
    //Store every X:th energy value to array. This will later be stored to file for resampling by Python script.
    if(m_stepNumber%howOftenStoreSampleForBlocking == 0){
        energy_array_for_blocking.push_back(localEnergy);
    }
}

// Legacy printout method. It's not actually needed, but kept until we changed to printOutputToTerminal(bool verbose) in all the places it is used.
void PretrainSampler::printOutputToTerminal(PretrainSystem &system, bool verbose)
{
    // storeSystemParameters(&system); //This is also redundant since code in system.cpp ensures it's been called.
    printOutputToTerminal(verbose);
}
void PretrainSampler::printOutputToTerminal(bool verbose)
{
    auto p = m_wavefunction_parameters.size();
    if (!verbose)
    {
        for (const auto &x : m_wavefunction_parameters)
            cout << x << " ";
        for (const auto &x : m_observables)
            cout << x << " ";
        return;
    }

    cout << endl;
    cout << "  -- Pre-training System info -- " << endl;
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
        cout << " Parameter " << i + 1 << " : " << m_wavefunction_parameters.at(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " A : " << m_observables[0] << endl;
    cout << " Standard deviation A : " << m_observables[1] << endl;
    cout << " Computed gradient : " << m_observables[2] << endl;
    cout << endl;
}


void PretrainSampler::printOutputToTerminalMini(bool verbose)
{
    auto p = m_wavefunction_parameters.size();
    cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << endl;
    cout << " Ratio of accepted steps: "
        << RED <<  ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps)
        << RESET << endl;
    cout << " K : " << YELLOW << m_observables[0] << RESET << endl;
    //Now also print E, E^2 and A which are the three last elements of m_observables
    cout << " E : " << YELLOW << m_observables[m_observables.size()-3] << RESET << " ";
    cout << " E^2 : " << YELLOW << m_observables[m_observables.size()-2] << RESET << " ";
    cout << " A : " << YELLOW << m_observables[m_observables.size()-1] << RESET << endl;
    cout << endl;
}

void PretrainSampler::computeObservables()
{
    /* Compute the observables out of the sampled quantities.
     */

    //Here we calculate eq. (7) in Saito's article.
    //Result is derivative of K w.r.t. parameters, used for optmization.
    //Given by formula 2K (<AO_W>/<A> - <O_W>) where K=<A>^2/<A^2>
    m_observables[0] = m_cumulatives[0] / m_numberOfMetropolisSteps;
    double A = m_observables[0];
    double AMeanSquared = m_cumulatives[1] / m_numberOfMetropolisSteps;
    m_observables[1] = sqrt((m_cumulatives[1] / m_numberOfMetropolisSteps - m_observables[0] * m_observables[0]) / m_numberOfMetropolisSteps);
//cout << "m_observables[0]=" << m_observables[0] << "m_cumulatives: " << m_cumulatives[0] << " " << m_cumulatives[1] << " " << m_numberOfMetropolisSteps << endl;
    double K = fabs(A*A / AMeanSquared);
//cout << "K=" << K << " m_numberOfMetropolisSteps " << m_numberOfMetropolisSteps << endl;
    for (size_t i = 0; i < m_numberOfParameters; i++)
    {
        double AO_W = m_cumulatives[2 + m_numberOfParameters + i] / m_numberOfMetropolisSteps;
        double O_W = m_cumulatives[2 + i] / m_numberOfMetropolisSteps;
        double parentesis = AO_W / A - O_W;
  //      cout << "AO_W=" << AO_W << " O_W=" << O_W << " parentesis=" << parentesis << endl;
        //m_observables[2 + i] = - 2 * K * parentesis;
        m_observables[2 + i] = - 2 * K * parentesis;
        //m_observables[2 + i] = - m_observables[2 + i]; //We want to maximize the quantity K.
    }
    m_observables[0] = K;

    //Take the third last element from m_cumulatives, which is the local energy. Calculate mean and append to m_observables.
//    cout << "m_cumulatives.size()=" << m_cumulatives.size() << endl;
//    cout << "m_cumulatives[m_cumulatives.size()-3]=" << m_cumulatives[m_cumulatives.size()-3] << endl;
    double localEnergyMean = m_cumulatives[m_cumulatives.size()-3] / m_numberOfMetropolisSteps;
//    cout << "localEnergyMean=" << localEnergyMean << endl;
    m_observables[2 + m_numberOfParameters + 0] = localEnergyMean;
    //Same with energy squared
    double localEnergySquaredMean = m_cumulatives[m_cumulatives.size()-2] / m_numberOfMetropolisSteps;
//    cout << "localEnergySquaredMean=" << localEnergySquaredMean << endl;
    //m_observables.push_back(localEnergySquaredMean);

    m_observables[2 + m_numberOfParameters + 1] = localEnergySquaredMean;
    //Same with A
    double AMean = m_cumulatives[m_cumulatives.size()-1] / m_numberOfMetropolisSteps;
//    cout << "AMean=" << AMean << endl;

    m_observables[2 + m_numberOfParameters + 2] = AMean;
    //m_observables.push_back(AMean);
}

void PretrainSampler::storeSystemParameters(class PretrainSystem *system)
{
    m_wavefunction_parameters = system->getWaveFunction()->getParameters();
}
