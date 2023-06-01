#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"
#include "particle.h"

class CoulombHamiltonian : public Hamiltonian
{
public:
    /// @brief Constructor for the CoulombHamiltonian class.
    CoulombHamiltonian(double omega, double inter_strength);
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
    /// @brief Repulsive Coulomb interaction parameter.
    double m_inter;
};
