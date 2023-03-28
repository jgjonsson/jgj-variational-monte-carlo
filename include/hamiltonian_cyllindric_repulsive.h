#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"
#include "particle.h"

class RepulsiveHamiltonianCyllindric : public Hamiltonian
{
public:
    /// @brief Constructor for the RepulsiveHamiltonianCyllindric class.
    RepulsiveHamiltonianCyllindric(double omega, double gamma);
    /// @brief Compute the local energy of the system.
    /// @param waveFunction The trial wave function to use.
    /// @param particles Vector of particles.
    /// @return The local energy in natural units with m = 1.
    double computeLocalEnergy(
        class WaveFunction &waveFunction,
        std::vector<std::unique_ptr<class Particle>> &particles);

private:
    /// @brief Omega frequency parameter in cyllindric oscillator.
    double m_omega;
    /// @brief Ratio between the two frequencies in cyllindric oscillator.
    double m_gamma;
};
