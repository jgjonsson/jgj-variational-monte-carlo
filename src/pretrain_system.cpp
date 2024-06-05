#include <iostream>
#include <memory>
#include <cassert>

#include "../include/pretrain_system.h"
#include "../include/pretrain_sampler.h"
#include "../include/particle.h"
#include "../include/wavefunction.h"
#include "../include/hamiltonian.h"
#include "../include/montecarlo.h"

using namespace std;

PretrainSystem::PretrainSystem(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction,
    std::unique_ptr<class WaveFunction> targetWaveFunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles)
{
    m_numberOfParticles = particles.size();
    ;
    m_numberOfDimensions = particles[0]->getNumberOfDimensions();
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
    m_targetWaveFunction = std::move(targetWaveFunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
}

size_t PretrainSystem::runEquilibrationSteps(
    double stepLength,
    size_t numberOfEquilibrationSteps)
{

//cout << "Number of particles A: " << m_particles.size() << endl;
    size_t acceptedSteps = 0;

    for (size_t i = 0; i < numberOfEquilibrationSteps; i++)
    {
        acceptedSteps += m_solver->step(stepLength, *m_targetWaveFunction, m_particles);
    }

    return acceptedSteps;
}

std::unique_ptr<class PretrainSampler> PretrainSystem::runMetropolisSteps(
    double stepLength,
    size_t numberOfMetropolisSteps,
    bool skipSamplingGradients,
    int stepsPerSample)
{
    auto sampler = std::make_unique<PretrainSampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepLength,
        numberOfMetropolisSteps);

    for (size_t i = 0; i < numberOfMetropolisSteps; i++)
    {/*
        if(stepsPerSample>1){
            for(int j=0; j<stepsPerSample-1; j++){
                bool acceptedStep = m_solver->step(stepLength, *m_waveFunction, m_particles);
            }
        }*/
        runEquilibrationSteps(stepLength, stepsPerSample-1);
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepLength, *m_targetWaveFunction, m_particles);

        /* Here you should sample the energy (and maybe other things) using the
         * sampler instance of the PretrainSampler class.
         */
        // ...like what?
        //if(i%stepsPerSample==0){
            sampler->sample(acceptedStep, this);
        //}
    }

    sampler->computeObservables();
    sampler->storeSystemParameters(this);

    return sampler;
}

double PretrainSystem::computeLocalEnergy()
{
    // Helper function
    return m_hamiltonian->computeLocalEnergy(*m_waveFunction, m_particles);
}

double PretrainSystem::getRationToTrainTargetWaveFunction()
{
//cout << "Number of particles B: " << m_particles.size() << endl;
    /*cout << "particle positions << " << m_particles.size()   << endl;
    for (size_t i = 0; i < m_particles.size(); i++)
    {
        for (size_t j = 0; j < m_particles[i]->getNumberOfDimensions(); j++)
        {
            cout << m_particles[i]->getPosition()[j] << " ";
        }
    }
    cout << endl;*/
    double wave = m_waveFunction->evaluate(m_particles);
    double waveTrainTarget = m_targetWaveFunction->evaluate(m_particles);
    //cout << "wave: " << wave << " waveTrainTarget: " << waveTrainTarget << endl;
    return waveTrainTarget / wave;
}

void PretrainSystem::setParticles(std::vector<std::unique_ptr<class Particle>> particles) {
    m_particles = std::move(particles);
}
