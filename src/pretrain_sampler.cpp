#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include "../include/system.h"
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
    m_cumulatives = std::vector<double>(2 + 2 * numberOfParameters, 0.0);
    // <E>, <(E-<E>)^2>, <d E/d alpha>
    m_observables = std::vector<double>(2 + numberOfParameters, 0.0);
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
    m_cumulatives = std::vector<double>(2 + 2 * m_numberOfParameters, 0.0);
    // <E>, <(E-<E>)^2>, <d E/d alpha>
    m_observables = std::vector<double>(2 + m_numberOfParameters, 0.0);

    energy_array_for_blocking = std::vector<double>();

    for (int i = 0; i < numberSamplers; i++)
    {
        for (int j = 0; j < 2 + 2 * m_numberOfParameters; j++)
        {
            m_cumulatives[j] += samplers[i]->m_cumulatives[j] / numberSamplers;
        }
        for (int j = 0; j < 2 + m_numberOfParameters; j++)
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

void PretrainSampler::sample(bool acceptedStep, System *system)
{
    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */
    auto localEnergy = system->computeLocalEnergy();
    //..
    auto particles = std::move(system->getParticles());
    auto gradients = system->getWaveFunction()->computeLogPsiDerivativeOverParameters(particles);

    //The A in eqs. (6) and (7) in the Saito's article.
    double A = system->getWaveFunction()->ratioToTrainingGaussian_A(particles);

    // I am a terrible person
    system->getParticles() = std::move(particles);

    m_cumulatives[0] += A;
    m_cumulatives[1] += A*A;
    for (size_t i = 0; i < m_numberOfParameters; i++)
    {
        m_cumulatives[2 + i] += gradients[i];  //The O_W in Saito's article
        m_cumulatives[2 + m_numberOfParameters + i] += gradients[i] * A; //The A*O_W in Saito's article
    }
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;

    //Store every X:th energy value to array. This will later be stored to file for resampling by Python script.
    if(m_stepNumber%howOftenStoreSampleForBlocking == 0){
        energy_array_for_blocking.push_back(localEnergy);
    }
}

// Legacy printout method. It's not actually needed, but kept until we changed to printOutputToTerminal(bool verbose) in all the places it is used.
void PretrainSampler::printOutputToTerminal(System &system, bool verbose)
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
        cout << " Parameter " << i + 1 << " : " << m_wavefunction_parameters.at(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_observables[0] << endl;
    cout << " Standard deviation Energy : " << m_observables[1] << endl;
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
    cout << " Energy : " << YELLOW << m_observables[0] << RESET << endl;
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
    m_observables[1] = sqrt((m_cumulatives[1] / m_numberOfMetropolisSteps - m_observables[0] * m_observables[0]) / m_numberOfMetropolisSteps);

    double K = fabs(m_observables[0] * m_observables[0] / m_observables[1]);

    for (size_t i = 0; i < m_numberOfParameters; i++)
    {
        double AO_W = m_cumulatives[2 + m_numberOfParameters + i] / m_numberOfMetropolisSteps;
        double O_W = m_cumulatives[2 + i] / m_numberOfMetropolisSteps;
        double A = m_observables[0];
        double parentesis = AO_W / A - O_W;
        m_observables[2 + i] = 2 * K * parentesis;
    }
}

void PretrainSampler::storeSystemParameters(class System *system)
{
    m_wavefunction_parameters = system->getWaveFunction()->getParameters();
}
