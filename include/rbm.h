#pragma once

#include <memory>
#include <armadillo>

#include "wavefunction.h"
#include "particle.h"
#include "random.h"

using namespace arma;

/// @brief The SimpleRBM class is a Restricted Boltzmann machine.
/// @details Gaussian-Binary without intercations.
class SimpleRBM : public WaveFunction
{
public:
    /// @brief Constructor for the SimpleRBM class.
    /// @param TODO: document
    SimpleRBM(size_t rbs_M, size_t rbs_N, std::vector<double> parameters, double omega);
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

    void insertParameters(std::vector<double> parameters);
    static std::vector<double> generateRandomParameterSet(size_t rbs_M, size_t rbs_N, int randomSeed, double spread);

protected:
    double gradientSquaredOfLnWaveFunction(vec x);
    double laplacianOfLnWaveFunction(vec x);

    //Helper-function to turn the P particles times D dimensions coordinates into a M=P*D vector
    vec flattenParticleCoordinatesToVector(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M);

    //Helper functions for computing gradient for gradient descent.
    vec gradient_a_ln_psi(vec x);
    vec gradient_b_ln_psi(vec x);
    mat gradient_W_ln_psi(vec x);

    //Storing the physical contants for this model. Values are set in constructur.
    double m_sigmaSquared;
    double m_omega;

    //Parameters for the wave function
    vec m_a;  //M parameters. The bias for visible layers.
    vec m_b;  //N parameters. The bias for hidden layers.
    mat m_W;  //M*N parameters. All weights connecting visible and hidden layers.

    //For storing number of parameters, M and N
    size_t m_M = 0;
    size_t m_N = 0;
};
