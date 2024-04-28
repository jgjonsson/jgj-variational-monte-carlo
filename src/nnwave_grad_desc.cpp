#include <memory>
#include <cmath>
#include <cassert>

//#include <autodiff/forward/dual.hpp>
//#include <autodiff/forward/dual/eigen.hpp>

#include "../include/nn_wave.h"
#include "../include/particle.h"
#include "../include/random.h"

using namespace std;
using namespace autodiff;
using namespace Eigen;
//using namespace arma;

//The following 3 functions are taken from formulas (82), (83), (84) at https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/week13/ipynb/week13.ipynb
/*vec NeuralNetworkWavefunction::gradient_a_ln_psi(vec x)
{
    return (x - m_a)/m_sigmaSquared;
}

vec NeuralNetworkWavefunction::gradient_b_ln_psi(vec x)
{
    return 1/(1 + exp(-(m_b + 1/m_sigmaSquared*(m_W.t()*x))));
}

mat NeuralNetworkWavefunction::gradient_W_ln_psi(vec x)
{
    return x/m_sigmaSquared*(gradient_b_ln_psi(x).t());
}
*/
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
std::vector<double> NeuralNetworkWavefunction::computeLogPsiDerivativeOverParameters(std::vector<std::unique_ptr<class Particle>> &particles)
{
    cout << "Trying to compute log psi derivative over parameters" << endl;
//#include "../include/neural.h"

    VectorXdual xDual = flattenParticleCoordinatesToVectorAutoDiffFormat(particles, m_M);

    std::vector<double> inputs = flattenParticleCoordinatesToVector(particles, m_M);


    auto x1 = m_neuralNetwork.feedForward(inputs);
    cout << "Input.length = " << inputs.size() << endl;
    cout << "x1 = " << x1 << endl;
    VectorXdual inputsDual = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<dual>();

    cout << "inputsDual.length = " << inputsDual.size() << endl;
    auto x2 = m_neuralNetwork.feedForwardDual2(inputsDual);
    cout << "x2 = " << x2 << endl;
    inputsDual = inputsDual.transpose();

    cout << "Trying deivative 1 " << endl;
//    VectorXd gradienten = m_neuralNetwork.getGradient(x);
    auto gradientFunction = m_neuralNetwork.getGradientFunction();
        cout << "Trying deivative 2 " << endl;
        cout << "m_neuralNetwork.parametersDual = " << m_neuralNetwork.parametersDual << endl;
        cout << "inputsDual = " << inputsDual << endl;
    auto theGradient = gradientFunction(m_neuralNetwork.parametersDual, inputsDual);
        cout << "Trying deivative 3 " << endl;
/*
    vec grad_a = gradient_a_ln_psi(x);
    vec grad_b = gradient_b_ln_psi(x);

    mat grad_W = gradient_W_ln_psi(x);
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
    }*/
    cout << "Cuccedded to compute log psi derivative over parameters" << endl;
    std::vector<double> logPsiDerivativeOverParameters;

    for(int i = 0; i < theGradient.size(); i++) {
        dual d = theGradient[i];
        std::cout << "d = " << d << std::endl;
        double d2 = 0.0;//d.val();
        logPsiDerivativeOverParameters.push_back(d2); // .val() is used to get the value of the dual number
    }

    return logPsiDerivativeOverParameters;
}

/** Function for setting all the parameters. Takes one single array as input and populates a, b and W.
    It's important to insert the values in the same order they where taken out in the function right above.
    This function is meant to be called repeatedly during Gradient descent.
    It's also to be used at first initialization together with function generateRandomParameterSet below.
*/
void NeuralNetworkWavefunction::insertParameters(std::vector<double> parameters)
{
    cout << "Trying to insert parameters" << endl;
    assert(parameters.size() == m_M+m_N+m_M*m_N);

    size_t index = 0;

    for (size_t i = 0; i < m_M; i++){
        m_a(i) = parameters[index++];
    }

    for (size_t i = 0; i < m_N; i++){
        m_b(i) = parameters[index++];
    }

    for (size_t i = 0; i < m_M; i++){
        for (size_t j = 0; j < m_N; j++){
            m_W(i,j) = parameters[index++];
        }
    }

    m_parameters = parameters; //Lastly, also store the plain vector of parameters.
    //This is double storing, but enables us to keep using the unmodified functions in Sampler for writing results to stdout.
}

/** Generate random numbers for all parameters. This is meant to be used before the first step of Gradient descent.
*/
std::vector<double> NeuralNetworkWavefunction::generateRandomParameterSet(size_t rbs_M, size_t rbs_N, int randomSeed, double spread)
{
    //Using a normal distribution for initial guess.
    //Based on code in https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/week13/ipynb/week13.ipynb
    //although the spread is parameterized for us to investigate different values. In code example it was 0.001.
    mt19937_64 generator;
    generator.seed(randomSeed);
    normal_distribution<double> distribution(0, spread);

    std::vector<double> parameters = std::vector<double>();
    //size_t numberParameters = rbs_M+rbs_N+rbs_M*rbs_N;
    int inputNodes = rbs_M;
    int hiddenNodes = rbs_N;
    //TODO: To many parameters, because we should not have weights and bias for all three layers. Let it be for now.
    int numberParameters = inputNodes * hiddenNodes + hiddenNodes + 1 + inputNodes + hiddenNodes * inputNodes + 1;
    for (size_t i = 0; i < numberParameters; i++){
        parameters.push_back(distribution(generator));
    }
    return parameters;
}