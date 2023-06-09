#pragma once

#include <memory>
#include <vector>

#include "particle.h"
#include "random.h"

/// @brief Setup a random uniform initial state for the particles
/// @param stepLength Effective box size for particle placement
/// @param numberOfDimensions number of space dimensions
/// @param numberOfParticles number of particles
/// @param randomEngine random engine
/// @return uniformly random vector of particles
std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    double stepLength,
    size_t numberOfDimensions,
    size_t numberOfParticles,
    Random &randomEngine);

/// @brief Setup a random uniform initial state for the particles with a hard core repulsion safety check
/// @param stepLength Effective box size for particle placement
/// @param numberOfDimensions number of space dimensions
/// @param numberOfParticles number of particles
/// @param randomEngine random engine
/// @return uniformly random vector of particles
std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialStateWithRepulsion(
    double stepLength,
    double hardCoreSize,
    size_t numberOfDimensions,
    size_t numberOfParticles,
    Random &randomEngine);