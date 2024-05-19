#include <iostream>
#include <memory>
#include <cassert>

#include "../include/system.h"
#include "../include/sampler.h"
#include "../include/particle.h"
#include "../include/wavefunction.h"
#include "../include/hamiltonian.h"
#include "../include/montecarlo.h"

System::System(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles)
{
    m_numberOfParticles = particles.size();
    ;
    m_numberOfDimensions = particles[0]->getNumberOfDimensions();
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
}

size_t System::runEquilibrationSteps(
    double stepLength,
    size_t numberOfEquilibrationSteps)
{
    size_t acceptedSteps = 0;

    for (size_t i = 0; i < numberOfEquilibrationSteps; i++)
    {
        acceptedSteps += m_solver->step(stepLength, *m_waveFunction, m_particles);
    }

    return acceptedSteps;
}

std::unique_ptr<class Sampler> System::runMetropolisSteps(
    double stepLength,
    size_t numberOfMetropolisSteps)
{
    auto sampler = std::make_unique<Sampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepLength,
        numberOfMetropolisSteps);

    for (size_t i = 0; i < numberOfMetropolisSteps; i++)
    {
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepLength, *m_waveFunction, m_particles);

        /* Here you should sample the energy (and maybe other things) using the
         * sampler instance of the Sampler class.
         */
        // ...like what?

        sampler->sample(acceptedStep, this);
    }

    sampler->computeObservables();
    sampler->storeSystemParameters(this);

    return sampler;
}

double System::computeLocalEnergy()
{
    // Helper function
    return m_hamiltonian->computeLocalEnergy(*m_waveFunction, m_particles);
}

void System::setParticles(std::vector<std::unique_ptr<class Particle>> particles) {
    m_particles = std::move(particles);
}