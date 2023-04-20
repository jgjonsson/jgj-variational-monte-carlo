#include <memory>
//#include <iostream>
#include <cassert>
#include <cmath>

#include "../include/initialstate.h"
#include "../include/particle.h"
#include "../include/random.h"

std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    double stepLength,
    size_t numberOfDimensions,
    size_t numberOfParticles,
    Random &rng)
{
    assert(numberOfDimensions > 0 && numberOfParticles > 0);

    auto particles = std::vector<std::unique_ptr<Particle>>();

    for (size_t i = 0; i < numberOfParticles; i++)
    {
        std::vector<double> position = std::vector<double>();
        for (size_t j = 0; j < numberOfDimensions; j++)
        {
            // uniformly distributed random number between -stepLength/2 and stepLength/2
            double pos = -stepLength / 2 + rng.nextDouble() * stepLength;
            position.push_back(pos);
        }
        particles.push_back(std::make_unique<Particle>(position));
    }

    return particles;
}

std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialStateWithRepulsion(double stepLength, double hardCoreSize, size_t numberOfDimensions, size_t numberOfParticles, Random &randomEngine)
{
    assert(numberOfDimensions > 0 && numberOfParticles > 0 && pow(stepLength/hardCoreSize, numberOfDimensions) > numberOfParticles);

    // vector of particles
    auto particles = std::vector<std::unique_ptr<Particle>>();

    auto distance2 = [](const std::vector<double> &pos1, const std::vector<double> &pos2)
    {
        double dist2 = 0;
        for (size_t i = 0; i < pos1.size(); i++)
            dist2 += (pos1[i] - pos2[i]) * (pos1[i] - pos2[i]);
        return dist2;
    };

    //Theoretically multiplying with a factor 4 below should be enough because hardCoreSize is the radius,
    //meaning distance is 2*hardCoreSize before taking square results in no overlap.
    //However it's shown empirically that even up to distances near 4*hardCoreSize quantum force gets so big
    //the suggested jumps are too improbably and we get stuck with too low change for acceptance.
    //Therefore we multiply with 32 below (16 would be the border case).
    //Techically we disallow nearly overlapping initial state, rather than only disallowing truly overlapping states.
    //In the end this should not matter too much what we multiply with as the initial state matters less with large
    //numbers of cycles.
    auto safeDistanceSquare = 32 * hardCoreSize * hardCoreSize;

    auto initialWidthOfParticleCollection = 1.0;

    for (size_t i = 0; i < numberOfParticles; i++)
    {
        std::vector<double> position = std::vector<double>();
        bool overlap = false;
        for (size_t j = 0; j < numberOfDimensions; j++)
        {
            // uniformly distributed random number between -initialWidthOfParticleCollection/2 and initialWidthOfParticleCollection/2
            double pos = -initialWidthOfParticleCollection / 2 + randomEngine.nextDouble() * initialWidthOfParticleCollection;

            position.push_back(pos);
        }
        for (size_t k = 0; k < i; k++)
        {
            if (distance2(position, particles[k]->getPosition()) < safeDistanceSquare)
            {
                i--;
                overlap = true;
                //std::cout << "Overlap" << std::endl;
                break;
            }
        }
        if (overlap)
            continue;
        particles.push_back(std::make_unique<Particle>(position));
    }

    return particles;
}