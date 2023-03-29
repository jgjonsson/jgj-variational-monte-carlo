
#include <memory>
#include <cmath>
#include <cassert>

#include "../include/gaussianjastrow.h"
#include "../include/particle.h"

GaussianJastrow::GaussianJastrow(double alpha, double beta, double hard_core_size)
    : m_beta{beta}, m_hard_core_size{hard_core_size}, WaveFunction()
{
    assert(alpha >= 0 && beta >= 0 && hard_core_size > 0);
    m_numberOfParameters = 1;
    m_parameters.push_back(alpha);
}

double GaussianJastrow::evaluate(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double psi = 1.0;
    double alpha = m_parameters[0]; // alpha is the first and only parameter for now.

    for (size_t i = 0; i < particles.size(); i++)
    {
        // Let's support as many dimensions as we want.
        double r2 = 0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
            r2 += (j == 2 ? m_beta : 1.0) * particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        // spherical ansatz
        double g = exp(-alpha * r2);

        // Trial wave function is product of g for all particles.
        psi = psi * g;
    }

    // Multiply by Jastrow factor.
    for (size_t i = 0; i < particles.size(); i++)
    {
        for (size_t j = i + 1; j < particles.size(); j++)
        {
            double rij = 0;
            for (size_t k = 0; k < particles[i]->getPosition().size(); k++)
                rij += (particles[i]->getPosition()[k] - particles[j]->getPosition()[k]) * (particles[i]->getPosition()[k] - particles[j]->getPosition()[k]);
            rij = sqrt(rij);
            double jastrow = (1.0 - m_hard_core_size / rij);
            if (jastrow > 0)
                psi = psi * jastrow;
            else
                return 0.0;
        }
    }

    return psi;
}

double GaussianJastrow::computeLocalLaplasian(std::vector<std::unique_ptr<class Particle>> &particles)
{
    // The expression I got for a single laplasian is, in invariant form, follows:
    // (1/4 * F_i^2 - 2 * alpha * (2 + beta) [IF ALL 3 DIMENSIONS!] - sum_j\ne i a^2/(rij^2 * (rij-a)^2))
    // so it takes to sum over all particles.
    double alpha = m_parameters[0];
    double sum_laplasian = 0.0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        double laplasian = 0.0;
        // Simplest term is -2 * alpha * (2 + beta), which is more generally given by
        if (particles[i]->getPosition().size() == 3)
            laplasian -= 2 * alpha * (2 + m_beta);
        else
            laplasian -= 2 * alpha * particles[i]->getPosition().size();
        for (size_t j = i + 1; j < particles.size(); j++)
        {
            // I kinda simplify the expression here, since rij is symmetric matrix, 
            // so instead of computing subdiagonal sum, which is the same as the top diagonal, I just take the top diagonal twice.
            double rij = 0.0;
            for (size_t k = 0; k < particles[i]->getPosition().size(); k++)
                rij += (particles[i]->getPosition()[k] - particles[j]->getPosition()[k]) * (particles[i]->getPosition()[k] - particles[j]->getPosition()[k]);
            rij = sqrt(rij);
            laplasian -= 2 * m_hard_core_size * m_hard_core_size / (rij * rij * (rij - m_hard_core_size) * (rij - m_hard_core_size));
        }
        // Last term is the sum of the squared force.
        auto quantum_force = computeQuantumForce(particles, i);
        for (size_t j = 0; j < quantum_force.size(); j++)
            laplasian += 0.25 * quantum_force[j] * quantum_force[j];
        sum_laplasian += laplasian;
    }
    return sum_laplasian;
}

double GaussianJastrow::evaluateRatio(std::vector<std::unique_ptr<class Particle>> &particles_numerator, std::vector<std::unique_ptr<class Particle>> &particles_denominator)
{
    assert(particles_numerator.size() == particles_denominator.size());
    double ratio = 1.0;
    double alpha = m_parameters[0];

    // Regular gaussian part.
    for (size_t i = 0; i < particles_numerator.size(); i++)
    {
        double r2_numerator = 0.0;
        double r2_denominator = 0.0;
        for (size_t j = 0; j < particles_numerator[i]->getPosition().size(); j++)
        {
            r2_numerator += (j == 2 ? m_beta : 1.0) * particles_numerator[i]->getPosition()[j] * particles_numerator[i]->getPosition()[j];
            r2_denominator += (j == 2 ? m_beta : 1.0) * particles_denominator[i]->getPosition()[j] * particles_denominator[i]->getPosition()[j];
        }
        ratio *= exp(-alpha * (r2_numerator - r2_denominator));
    }
    // Multiply by Jastrow factor.
    for (size_t i = 0; i < particles_numerator.size(); i++)
    {
        for (size_t j = i + 1; j < particles_numerator.size(); j++)
        {
            double rij_numerator = 0;
            double rij_denominator = 0;
            for (size_t k = 0; k < particles_numerator[i]->getPosition().size(); k++)
            {
                rij_numerator += (particles_numerator[i]->getPosition()[k] - particles_numerator[j]->getPosition()[k]) * (particles_numerator[i]->getPosition()[k] - particles_numerator[j]->getPosition()[k]);
                rij_denominator += (particles_denominator[i]->getPosition()[k] - particles_denominator[j]->getPosition()[k]) * (particles_denominator[i]->getPosition()[k] - particles_denominator[j]->getPosition()[k]);
            }
            rij_numerator = sqrt(rij_numerator);
            rij_denominator = sqrt(rij_denominator);
            double jastrow_numerator = (1.0 - m_hard_core_size / rij_numerator);
            double jastrow_denominator = (1.0 - m_hard_core_size / rij_denominator);
            if (jastrow_numerator > 0)
                ratio *= jastrow_numerator / jastrow_denominator;
            else
                return 0.0;
        }
    }

    return ratio;
}

std::vector<double> GaussianJastrow::computeQuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index)
{
    // I assume again that we do not arrive to forbidden states (r < r_hard_core), so I do not check for that.
    double alpha = m_parameters[0];
    std::vector<double> quantumForce = std::vector<double>();
    std::vector<double> position = particles[particle_index]->getPosition();
    for (size_t j = 0; j < position.size(); j++)
    {
        quantumForce.push_back(-4 * alpha * position[j] * (j == 2 ? m_beta : 1.0));
    }
    for (size_t j = 0; j < particles.size(); j++)
    {
        if (j != particle_index)
        {
            double rij = 0;
            for (size_t k = 0; k < particles[j]->getPosition().size(); k++)
                rij += (particles[j]->getPosition()[k] - particles[particle_index]->getPosition()[k]) * (particles[j]->getPosition()[k] - particles[particle_index]->getPosition()[k]);
            rij = sqrt(rij);
            for (size_t k = 0; k < position.size(); k++)
                quantumForce[k] += 2.0 * m_hard_core_size * (position[k] - particles[j]->getPosition()[k]) / (rij * rij * (rij - m_hard_core_size));
        }
    }
    return quantumForce;
}

std::vector<double> GaussianJastrow::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double alpha = m_parameters[0];
    std::vector<double> logPsiDerivativeOverParameters = std::vector<double>();
    double sum = 0.0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        double r2 = 0.0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
            r2 += (j == 2 ? m_beta : 1.0) * particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        sum += r2;
    }
    logPsiDerivativeOverParameters.push_back(-sum);
    return logPsiDerivativeOverParameters;
}