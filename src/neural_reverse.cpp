#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <Eigen/Dense>

// autodiff include
//#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "../include/neural_reverse.h"

using namespace autodiff;
using namespace Eigen;
using namespace std;

NeuralNetworkReverse::NeuralNetworkReverse(std::vector<double> randNumbers, int inputSize, int hiddenSize)
    : parameters(randNumbers), inputSize(inputSize), hiddenSize(hiddenSize)
{
    parametersDual = Eigen::Map<VectorXd>(randNumbers.data(), randNumbers.size()).cast<var>().array();
}

using ArrayXXvar = Eigen::Array<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>;


inline var relu(var x) {
    return max(var(0.0), x);
}

inline double relu(double x) {
    return std::max(0.0, x);
}

inline var leaky_relu(var x) {
    return max(0.01 * x, x);
}

inline double leaky_relu(double x) {
    return std::max(0.01 * x, x);
}

var feedForwardXvar(const ArrayXvar& parameters, const ArrayXvar& inputs, int inputSize, int hiddenSize) {

    int weightsSize = inputSize * hiddenSize + hiddenSize;
    ArrayXvar inputLayerWeights = parameters.head(inputSize * hiddenSize);
    ArrayXvar hiddenLayerWeights = parameters.segment(inputSize * hiddenSize, hiddenSize);
    ArrayXvar hiddenLayerBiases = parameters.tail(hiddenSize);

    ArrayXvar hiddenOutputs(hiddenLayerBiases.size());
    for(int i = 0; i < hiddenLayerBiases.size(); i++) {
        var output = 0.0;
        for(int j = 0; j < inputs.size(); j++) {
            output += inputLayerWeights[j * hiddenLayerBiases.size() + i] * inputs[j];
        }
        output += hiddenLayerBiases[i];
        hiddenOutputs[i] = tanh(output);
        //hiddenOutputs[i] = leaky_relu(output);
    }

    var finalOutput = 0.0;
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        finalOutput += hiddenLayerWeights[i] * hiddenOutputs[i];
    }
    return finalOutput;
}

double NeuralNetworkReverse::feedForward(std::vector<double> inputs) {
    int weightsSize = inputSize * hiddenSize + hiddenSize;

    std::vector<double> inputLayerWeights(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
    std::vector<double> hiddenLayerWeights(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
    std::vector<double> hiddenLayerBiases(parameters.begin() + weightsSize, parameters.begin() + weightsSize + hiddenSize);

    std::vector<double> hiddenOutputs;
    for(int i = 0; i < hiddenLayerBiases.size(); i++) {
        double output = 0.0;
        for(int j = 0; j < inputs.size(); j++) {
            //output += inputLayerWeights[i * inputs.size() + j] * inputs[j];
            output += inputLayerWeights[j * hiddenLayerBiases.size() + i] * inputs[j];
        }
        output += hiddenLayerBiases[i];
        hiddenOutputs.push_back(tanh(output));
        //hiddenOutputs.push_back(leaky_relu(output));
    }

    double finalOutput = 0.0;
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        finalOutput += hiddenLayerWeights[i] * hiddenOutputs[i];
    }
    return finalOutput;
}

std::vector<double> NeuralNetworkReverse::getTheGradientVectorWrtParameters(std::vector<double> &inputs)
{
    VectorXvar xInputs = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& kalle) {
        return feedForwardXvar(kalle, xInputs, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(parametersDual); // the output variable y
    VectorXd dydx = gradient(y, parametersDual);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}

std::vector<double> NeuralNetworkReverse::getTheGradientVectorWrtInputs(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}

std::vector<double> NeuralNetworkReverse::calculateNumericalGradientParameters(std::vector<double>& inputs) {
    double epsilon = 1e-6; // small number for finite difference
    std::vector<double> gradient(parameters.size());

    for (size_t i = 0; i < parameters.size(); ++i) {
        // Store the original value so we can reset it later
        double originalValue = parameters[i];

        // Evaluate function at p+h
        parameters[i] += epsilon;
        double plusEpsilon = feedForward(inputs);

        // Evaluate function at p-h
        parameters[i] = originalValue - epsilon;
        double minusEpsilon = feedForward(inputs);

        // Compute the gradient
        gradient[i] = (plusEpsilon - minusEpsilon) / (2.0 * epsilon);

        // Reset the parameter to its original value
        parameters[i] = originalValue;
    }

    return gradient;
}

/* Calculating a single derivative of log(psi) with respect to one particle's and one dimension's position
This is used when calculating quantum force for one particle*/
double NeuralNetworkReverse::calculateNumericalDeriviateWrtInput(std::vector<double>& inputs, int inputIndexForDerivative) {
    double epsilon = 1e-6; // small number for finite difference
    // Store the original value so we can reset it later
    double originalValue = inputs[inputIndexForDerivative];

    // Evaluate function at x+h
    inputs[inputIndexForDerivative] += epsilon;
    double plusEpsilon = feedForward(inputs);

    // Evaluate function at x-h
    inputs[inputIndexForDerivative] = originalValue - epsilon;
    double minusEpsilon = feedForward(inputs);

    // Reset the input to its original value
    inputs[inputIndexForDerivative] = originalValue;

    // Compute the derivative
    return (plusEpsilon - minusEpsilon) / (2.0 * epsilon);
}
/*
std::vector<double> NeuralNetworkReverse::getTheLaplacianVectorWrtInputs(std::vector<double> inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXvar dydx = gradient(y, x); // evaluate the gradient vector dy/dx

    std::vector<double> laplacian(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Compute the derivative of the i-th component of the gradient with respect to the i-th input variable
        var second_derivative = derivative(dydx[i], x[i]);
        laplacian[i] = second_derivative.val(); // Extract the value from the var type
    }

    return laplacian;
}
*/

/*
double NeuralNetworkReverse::getTheTotalLaplacian(std::vector<double> inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXvar dydx = gradient(y, x); // evaluate the gradient vector dy/dx

    double totalLaplacian = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Compute the derivative of the i-th component of the gradient with respect to the i-th input variable
        var second_derivative = derivative(dydx[i], x[i]);
        totalLaplacian += second_derivative.val(); // Extract the value from the var type and add it to the total
    }

    return totalLaplacian;
}
*/

//#include <vector>
//#include <functional>

double NeuralNetworkReverse::calculateNumericalLaplacianWrtInput(std::vector<double>& inputs) {
//double laplacian(std::function<double(std::vector<double>)> f, std::vector<double> x, double h = 1e-5) {

    double epsilon = 1e-6; // small number for finite difference

    double fInputs = feedForward(inputs);
    double laplacian = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {

        // Store the original value so we can reset it later
        double originalValue = inputs[i];

        // Evaluate function at p+h
        inputs[i] += epsilon;
        double plusEpsilon = feedForward(inputs);

        // Evaluate function at p-h
        inputs[i] = originalValue - epsilon;
        double minusEpsilon = feedForward(inputs);

        // Compute the gradient
        double second_derivative = (plusEpsilon - 2.0 * fInputs + minusEpsilon) / (epsilon * epsilon);
        laplacian += second_derivative;
        //gradient[i] = (plusEpsilon - minusEpsilon) / (2.0 * epsilon);

        // Reset the parameter to its original value
        inputs[i] = originalValue;
    }
    return laplacian;
/*
    for (size_t i = 0; i < x.size(); i++) {
        std::vector<double> x_plus_h = x;
        std::vector<double> x_minus_h = x;
        x_plus_h[i] += h;
        x_minus_h[i] -= h;
        double second_derivative = (f(x_plus_h) - 2.0 * f(x) + f(x_minus_h)) / (h * h);
        laplacian += second_derivative;
    }
    return laplacian;
    */
}

double NeuralNetworkReverse::getTheLaplacianVectorWrtInputs(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    MatrixXd H = hessian(y, x); // evaluate the Hessian matrix H = d^2y/dx^2

    std::vector<double> laplacian(H.diagonal().data(), H.diagonal().data() + H.diagonal().size());

    double sum = std::accumulate(laplacian.begin(), laplacian.end(), 0.0);
    return sum;
}

double NeuralNetworkReverse::getTheLaplacianVectorWrtInputs2(std::vector<double> &inputs)
{
return 0.0;
/*
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXvar dydx = gradient(y, x); // evaluate the gradient vector dy/dx

    std::vector<double> laplacian(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Compute the derivative of the i-th component of the gradient with respect to the i-th input variable
        var second_derivative = gradient(dydx[i], x[i]);
        laplacian[i] = second_derivative.val(); // Extract the value from the var type
    }

    double sum = std::accumulate(laplacian.begin(), laplacian.end(), 0.0);
    return sum;*/
}

/*
double NeuralNetworkReverse::getTheTotalLaplacian(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXvar dydx = gradient(y, x); // evaluate the gradient vector dy/dx

    VectorXvar dy2dx2 = derivatives(dydx, wrt(x)); // evaluate the gradient vector dy/dx

    return 0.0; */
/*
    VectorXvar d2ydx2(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        d2ydx2[i] = autodiff::derivatives(dydx[i], x[i]);
        //d2ydx2[i] = derivative(dydx[i], x[i]);
    }
    //VectorXvar d2ydx2 = gradient(dydx, x);

    std::vector<double> d2ydx2_vec(d2ydx2.data(), d2ydx2.data() + d2ydx2.size());
    double totalLaplacian = std::accumulate(d2ydx2_vec.begin(), d2ydx2_vec.end(), 0.0);

    return totalLaplacian;*/
//}
/*
std::vector<double> NeuralNetworkReverse::getTheGradientVectorWrtInputs(std::vector<double> inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}
*/