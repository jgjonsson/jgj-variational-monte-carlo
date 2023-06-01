#include <memory>
#include <cassert>
#include <iostream>
#include <cmath>

#include "../include/hamiltonian_coulomb.h"
#include "../include/particle.h"
#include "../include/wavefunction.h"


CoulombHamiltonian::CoulombHamiltonian(double omega, double inter_strength)
    : m_omega{omega}, m_inter{inter_strength}
{
    assert(omega > 0);
}

double CoulombHamiltonian::computeLocalEnergy(
    class WaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    auto kineticEnergy = -0.5 * waveFunction.computeLocalLaplasian(particles);
    auto potentialEnergy = 0.0;

    // External potential.
    for (auto &particle : particles)
    {
        auto position = particle->getPosition();
        for (size_t pos_index = 0; pos_index < particle->getNumberOfDimensions(); ++pos_index)
            potentialEnergy += position[pos_index] * position[pos_index];
    }
    potentialEnergy *= 0.5 * m_omega * m_omega;

    // Repulsive Coulomb interaction.
    for (size_t i = 0; i < particles.size(); ++i)
    {
        auto position_i = particles[i]->getPosition();
        for (size_t j = i + 1; j < particles.size(); ++j)
        {
            auto position_j = particles[j]->getPosition();
            auto distance = 0.0;
            for (size_t pos_index = 0; pos_index < particles[i]->getNumberOfDimensions(); ++pos_index)
                distance += (position_i[pos_index] - position_j[pos_index]) * (position_i[pos_index] - position_j[pos_index]);
            potentialEnergy += m_inter / sqrt(distance);
        }
    }


    return kineticEnergy + potentialEnergy;
}
