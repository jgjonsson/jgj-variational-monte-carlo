//#include <gtest/gtest.h>
#include <cassert>
#include <cmath> // for std::fabs

#include "../../include/nn_wave_pure.h"
#include "../../include/particle.h"

int main() {
    // Initialize parameters
    size_t numberOfParticles = 2;
    size_t numberOfDimensions = 3;
    size_t inputNodes = numberOfParticles * numberOfDimensions;
    size_t hiddenNodes = 2;
    size_t numberParameters = inputNodes * hiddenNodes + hiddenNodes * 2;

    std::vector<double> parameters(numberParameters);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    for (size_t i = 0; i < numberParameters; ++i) {
        parameters[i] = dis(gen);
    }
    for (size_t i = 0; i < numberParameters; ++i) {
        std::cout << "Parameter " << i << ": " << parameters[i] << std::endl;
    }

    double omega = 1.0;
    double alpha = 0.5;
    double beta = 1.0;
    double adiabaticFactor = 1.0;

    // Create an instance of PureNeuralNetworkWavefunction
    PureNeuralNetworkWavefunction wavefunction(inputNodes, hiddenNodes, parameters, omega, alpha, beta, adiabaticFactor);

    // Create a vector of particles
    std::vector<std::unique_ptr<class Particle>> particles;
    for (size_t i = 0; i < numberOfParticles; ++i) {
        std::vector<double> position(numberOfDimensions, 1.0);
        particles.push_back(std::make_unique<Particle>(position));
    }
/*
    for (size_t i = 0; i < numberOfParticles; ++i) {
        std::vector<double> quantumForce = wavefunction.computeQuantumForceOld(particles, i);
        for (size_t j = 0; j < numberOfDimensions; ++j) {
            std::cout << "Old q-force   " << i << " " << j << ": " << quantumForce[j] << std::endl;
        }
    }*/

    for (size_t i = 0; i < numberOfParticles; ++i) {
        std::vector<double> quantumForce = wavefunction.computeQuantumForce(particles, i);
        for (size_t j = 0; j < numberOfDimensions; ++j) {
            std::cout << "Quantum force " << i << " " << j << ": " << quantumForce[j] << std::endl;
        }
    }
    // Compute quantum force
    std::vector<double> quantumForce = wavefunction.computeQuantumForce(particles, 0);

    // Check the size of the quantum force vector
    //ASSERT_EQ(quantumForce.size(), numberOfDimensions);
    assert(quantumForce.size() == numberOfDimensions);

    // Check the values of the quantum force vector
    // The expected values depend on the specific implementation of your computeQuantumForce method
    // Here, we're just checking that the quantum force for each dimension is not zero
    for (double force : quantumForce) {
    cout << "Force: " << std::fabs(force) << endl;
        assert(std::fabs(force) > 1e-9); // assert that the absolute value of force is not zero
//        ASSERT_NE(force, 0.0);
    }
}