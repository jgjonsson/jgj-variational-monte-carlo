#pragma once
#include <memory>
//#include "../include/sampler.h"

class PretrainSampler// : public Sampler
{
public:
    PretrainSampler(
        size_t numberOfParticles,
        size_t numberOfDimensions,
        size_t numberOfParameters,
        double stepLength,
        size_t numberOfMetropolisSteps);
    PretrainSampler(std::unique_ptr<PretrainSampler>* samplers, int numberSamplers);

    void sample(bool acceptedStep, class PretrainSystem *system);
    void printOutputToTerminal(class PretrainSystem &system, bool verbose = false);
    void printOutputToTerminal(bool verbose = false);
    void printOutputToTerminalMini(bool verbose);
    // Invoked automatically after sampling
    void computeObservables();
    void storeSystemParameters(class PretrainSystem *system);
    const std::vector<double> &getObservables() const { return m_observables; }
    const std::vector<double> &getEnergyArrayForBlocking() const { return energy_array_for_blocking; }



private:
    size_t m_stepNumber = 0;
    size_t m_numberOfMetropolisSteps = 0;
    size_t m_numberOfParameters = 0;
    size_t m_numberOfParticles = 0;
    size_t m_numberOfDimensions = 0;
    size_t m_numberOfAcceptedSteps = 0;
    double m_stepLength = 0;
    std::vector<double> m_cumulatives;
    std::vector<double> m_observables;
    std::vector<double> m_wavefunction_parameters;
    std::vector<double> energy_array_for_blocking;
};
