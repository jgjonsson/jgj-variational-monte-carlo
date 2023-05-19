#include <memory>
#include <cmath>
#include <cassert>

#include "../include/rbm.h"
#include "../include/particle.h"
#include "../include/random.h"

using namespace std;
using namespace arma;

//The following 3 functions are taken from formulas (82), (83), (84) at https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/week13/ipynb/week13.ipynb
vec SimpleRBM::gradient_a_ln_psi(vec x)
{
    return (x - m_a)/m_sigmaSquared;
}

vec SimpleRBM::gradient_b_ln_psi(vec x)
{
    return 1/(1 + exp(-(m_b + 1/m_sigmaSquared*(m_W.t()*x))));
}

mat SimpleRBM::gradient_W_ln_psi(vec x)
{
    return x/m_sigmaSquared*(gradient_b_ln_psi(x).t());
}

/** Calculate derivative of ln(psi) with respect to every parameter
    This is the same as calculating 1/psi * delta psi / delta alpha_i term of eq (80) in
    https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/week13/ipynb/week13.ipynb

    Function first flattens all particle positions to an array x. Then calculates all interesting derivatives.
    And last it flattens the multiple arrays of derivatives to one long result vector,
    containing all the M+N+M*N derivatives w.r.t a, b and W.
    We do it like this because internally to this wave function class (the Boltzmann machine) parameters are
    organized in vectors a and b, and matrix W. But the sampler is built to handle just one vector of generic parameters.

    It's also worth noting that while eq (80) contains expectation values it's the samplers job to collect
    many values of this function, and accumulate in order to calculate the needed averages.
    This function is only concerned about the value for one state.
*/
std::vector<double> SimpleRBM::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    vec x = flattenParticleCoordinatesToVector(particles, m_M);
    vec grad_a = gradient_a_ln_psi(x);
    vec grad_b = gradient_b_ln_psi(x);
    vec grad_W = gradient_W_ln_psi(x);

    std::vector<double> logPsiDerivativeOverParameters = std::vector<double>();

    for (size_t i = 0; i < m_M; i++){
        logPsiDerivativeOverParameters.push_back(grad_a(i));
    }

    for (size_t i = 0; i < m_N; i++){
        logPsiDerivativeOverParameters.push_back(grad_b(i));
    }

    for (size_t i = 0; i < m_M; i++){
        for (size_t j = 0; j < m_N; j++){
            logPsiDerivativeOverParameters.push_back(grad_W(i,j));
        }
    }
    return logPsiDerivativeOverParameters;
}
