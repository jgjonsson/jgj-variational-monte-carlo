#pragma once
#include <memory>

class Sampler
{
public:
    Sampler(
        size_t numberOfParticles,
        size_t numberOfDimensions,
        size_t numberOfParameters,
        double stepLength,
        size_t numberOfMetropolisSteps);
    Sampler(std::unique_ptr<Sampler>* samplers, int numberSamplers);

    void sample(bool acceptedStep, class System *system);
    void printOutputToTerminal(class System &system, bool verbose = false);
    void printOutputToTerminal(bool verbose = false);
    // Invoked automatically after sampling
    void computeObservables();
    void storeSystemParameters(class System *system);
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
