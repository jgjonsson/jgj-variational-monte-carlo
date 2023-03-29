#include "../include/wavefunction.h"
#include "../include/particle.h"

#include <cmath>

double WaveFunction::computeLocalLaplasian(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // Compute the local energy by numerical differentiation
    double h = 1e-4; // never do this, but I assume dimensionless units, so fine
    double laplasian = 0;

    auto mid_val = evaluate(particles);

    for (auto &particle : particles)
    {
        for (size_t i = 0; i < particle->getPosition().size(); i++)
        {
            particle->adjustPosition(h, i);
            auto plus_val = evaluate(particles);
            particle->adjustPosition(-2 * h, i);
            auto minus_val = evaluate(particles);
            particle->adjustPosition(h, i);
            laplasian += (plus_val - 2 * mid_val + minus_val) / (h * h * mid_val);
        }
    }
    return laplasian;
}

double WaveFunction::evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator,
                                   std::vector<std::unique_ptr<class Particle>> &particles_denominator)
{
    return evaluate(particles_numerator) / evaluate(particles_denominator);
}

// compute the quantum force
std::vector<double> WaveFunction::computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    double h = 1e-4; // never do this, but I assume dimensionless units, so fine
    std::vector<double> quantum_force(particles[particle_index]->getPosition().size(), 0);

    auto mid_val = evaluate(particles);

    for (size_t i = 0; i < particles[particle_index]->getPosition().size(); i++)
    {
        particles[particle_index]->adjustPosition(h, i);
        auto plus_val = evaluate(particles);
        particles[particle_index]->adjustPosition(-h, i);
        quantum_force[i] = 2 * (plus_val/mid_val - 1) / h;
    }

    return quantum_force;
}

std::vector<double> WaveFunction::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    std::vector<double> log_psi_derivative_over_parameters(m_numberOfParameters);
    double h = 1e-4; // as usual, never do this, but I assume dimensionless units, so fine

    auto mid_val = log(evaluate(particles));

    for (size_t i = 0; i < m_numberOfParameters; i++)
    {
        m_parameters[i] += h;
        auto plus_val = log(evaluate(particles));
        m_parameters[i] -= h;
        log_psi_derivative_over_parameters[i] = (plus_val - mid_val) / h;
    }
    return log_psi_derivative_over_parameters;
}