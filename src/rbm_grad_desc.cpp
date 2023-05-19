#include <memory>
#include <cmath>
#include <cassert>

#include "../include/rbm.h"
#include "../include/particle.h"
#include "../include/random.h"

using namespace std;
using namespace arma;

//The following 3 functions are taken from forumlas (82), (83), (84) at https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/week13/ipynb/week13.ipynb
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


std::vector<double> SimpleRBM::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    double alpha = m_parameters[0];
    std::vector<double> logPsiDerivativeOverParameters = std::vector<double>();
    double sum = 0.0;
    for (size_t i = 0; i < particles.size(); i++)
    {
        double r2 = 0.0;
        for (size_t j = 0; j < particles[i]->getPosition().size(); j++)
            r2 += particles[i]->getPosition()[j] * particles[i]->getPosition()[j];
        sum += r2;
    }
    logPsiDerivativeOverParameters.push_back(-sum);
    return logPsiDerivativeOverParameters;
}
