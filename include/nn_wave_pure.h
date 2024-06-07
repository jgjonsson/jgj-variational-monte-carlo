#pragma once

#include <memory>
#include <armadillo>

#include "wavefunction.h"
#include "particle.h"
#include "random.h"
#include "neural_onelayer.h"

using namespace arma;

/// @brief The PureNeuralNetworkWavefunction class is a Restricted Boltzmann machine.
/// @details Gaussian-Binary without intercations.
class PureNeuralNetworkWavefunction : public WaveFunction
{
public:
    /// @brief Constructor for the PureNeuralNetworkWavefunction class.
    /// @param TODO: document
    //PureNeuralNetworkWavefunction(size_t rbs_M, size_t rbs_N, std::vector<double> parameters, double omega);
    PureNeuralNetworkWavefunction(size_t rbs_M, size_t rbs_N, std::vector<double> parameters, double omega, double alpha, double beta, double adiabaticFactor);
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
    std::vector<double> computeQuantumForceOld(std::vector<std::unique_ptr<class Particle>> &particles, size_t particle_index);
    std::vector<double> computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles);

    void insertParameters(std::vector<double> parameters);
    static std::vector<double> generateRandomParameterSet(size_t rbs_M, size_t rbs_N, int randomSeed, double spread);

    //Used for pre-training to make the neural network target the gaussian distribution.
    double ratioToTrainingGaussian_A(std::vector<std::unique_ptr<class Particle>> &particles);

    //std::vector<double> generateRandomParameterSet(size_t rbs_M, size_t rbs_N, int randomSeed, double spread);

protected:
    double gradientSquaredOfLnWaveFunction(vec x);
    double laplacianOfLnWaveFunction(vec x);

    //Helper-function to turn the P particles times D dimensions coordinates into a M=P*D vector
    std::vector<double> flattenParticleCoordinatesToVector(std::vector<std::unique_ptr<class Particle>> &particles, size_t m_M);

    //Storing the physical contants for this model. Values are set in constructur.
    double m_sigmaSquared;
    double m_omega;
    double m_alpha;
    double m_beta;
    //double m_adiabaticFactor;
    NeuralNetworkOneLayer m_neuralNetwork;

    //For storing number of parameters, M and N
    size_t m_M = 0;
    size_t m_N = 0;
};
