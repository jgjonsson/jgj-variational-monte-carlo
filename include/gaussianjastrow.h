#pragma once

#include <memory>

#include "wavefunction.h"
#include "particle.h"

/// @brief The GaussianJastrow class is a gaussian product trial wave function.
/// @details The trial wave function is a product of gaussian functions
/// for each particle, and is independent of the other particles, multiplied by a Jastrow factor.
class GaussianJastrow : public WaveFunction
{
private:
    /// @brief Non-variatial parameter responsible for spherical asymmetry.
    double m_beta;
    /// @brief Non-variatial parameter responsible for hard-core size.
    double m_hard_core_size;

public:
    /// @brief Constructor for the GaussianJastrow class.
    /// @param alpha Variational parameter present in the exponent.
    /// @param beta Non-variatial parameter responsible for spherical asymmetry.
    /// @param hard_core_size Non-variatial parameter responsible for hard-core size.
    GaussianJastrow(double alpha, double beta, double hard_core_size);
    /// @brief Evaluate the trial wave function.
    /// @param particles Vector of particles.
    /// @return The value of the trial wave function.
    double evaluate(std::vector<std::unique_ptr<class Particle>> &particles);
    /// @brief Compute the double derivative of the trial wave function over trial wave function.
    /// @param particles Vector of particles.
    /// @return The local value of Laplasian.
    double computeLocalLaplasian(std::vector<std::unique_ptr<class Particle>> &particles);
    /// @brief Efficiently compute ratio of evaluation of trial wave function.
    /// @param particles_numerator Vector of particles in the numerator.
    /// @param particles_denominator Vector of particles in the denominator.
    double evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator,
                         std::vector<std::unique_ptr<class Particle>> &particles_denominator);

    std::vector<double> computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index);
    std::vector<double> computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles);
};