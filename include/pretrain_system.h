#pragma once

#include <memory>
#include <vector>

#include "hamiltonian.h"
#include "montecarlo.h"
#include "particle.h"
#include "pretrain_sampler.h"
#include "wavefunction.h"

class PretrainSystem
{
public:
    PretrainSystem(
        std::unique_ptr<class Hamiltonian> hamiltonian,
        std::unique_ptr<class WaveFunction> waveFunction,
        std::unique_ptr<class WaveFunction> targetWaveFunction,
        std::unique_ptr<class MonteCarlo> solver,
        std::vector<std::unique_ptr<class Particle>> particles);

    size_t runEquilibrationSteps(
        double stepLength,
        size_t numberOfEquilibrationSteps);

    std::unique_ptr<class PretrainSampler> runMetropolisSteps(
        double stepLength,
        size_t numberOfMetropolisSteps);

    double computeLocalEnergy();
    std::unique_ptr<class WaveFunction> &getWaveFunction() { return m_waveFunction; }
    std::vector<std::unique_ptr<class Particle>> &getParticles() { return m_particles; }
    void setParticles(std::vector<std::unique_ptr<class Particle>> particles);
    double getRationToTrainTargetWaveFunction();

private:
    size_t m_numberOfParticles = 0;
    size_t m_numberOfDimensions = 0;

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    std::unique_ptr<class WaveFunction> m_waveFunction;
    std::unique_ptr<class WaveFunction> m_targetWaveFunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};
