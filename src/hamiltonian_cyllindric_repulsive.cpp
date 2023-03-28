#include <memory>
#include <cassert>
#include <iostream>

#include "../include/hamiltonian_cyllindric_repulsive.h"
#include "../include/particle.h"
#include "../include/wavefunction.h"

using std::cout;
using std::endl;

RepulsiveHamiltonianCyllindric::RepulsiveHamiltonianCyllindric(double omega, double gamma)
    : m_omega{omega}, m_gamma{gamma}
{
    assert(omega > 0);
}

double RepulsiveHamiltonianCyllindric::computeLocalEnergy(
    class WaveFunction &waveFunction,
    std::vector<std::unique_ptr<class Particle>> &particles)
{
    // The impossibility to get into overlapping hard core state is delegated to the particle sampling algorithm.
    // Therefore, the hard core repulsion is not included in the local energy.

    auto kineticEnergy = -0.5 * waveFunction.computeLocalLaplasian(particles);
    auto potentialEnergy = 0.0;

    for (auto &particle : particles)
    {
        auto position = particle->getPosition();
        for (size_t pos_index = 0; pos_index < particle->getNumberOfDimensions(); ++pos_index)
            potentialEnergy += (pos_index == 2 ? m_gamma * m_gamma : 1.0) * position[pos_index] * position[pos_index];
    }
    potentialEnergy *= 0.5 * m_omega * m_omega;

    return kineticEnergy + potentialEnergy;
}
