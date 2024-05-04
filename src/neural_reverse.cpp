#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <Eigen/Dense>

// autodiff include
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
    //gradientFunction = getGradientFunction();
}

using ArrayXXvar = Eigen::Array<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>;



var feedForwardXvar(const ArrayXvar& parameters, const ArrayXvar& inputs, int inputSize, int hiddenSize) {
//return sqrt((inputsDual * inputsDual).sum());

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
    }

    double finalOutput = 0.0;
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        finalOutput += hiddenLayerWeights[i] * hiddenOutputs[i];
    }
    return finalOutput;
}

std::vector<double> NeuralNetworkReverse::getTheGradientVectorParameters(std::vector<double> inputs)
{
/*
    VectorXvar xInputs(inputSize);
    for (int i = 0; i < inputSize; i++) {
        xInputs(i) = inputs[i];
    }
*/
    VectorXvar xInputs = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();
    //VectorXvar x = Eigen::Map<VectorXd>(parameters.data(), parameters.size()).cast<var>().array();
/*
    VectorXvar x(parameters.size());
    for (int i = 0; i < parameters.size(); i++) {
        x(i) = parameters[i];
    }
*/
    auto feedForwardWrapper = [&](const VectorXvar& kalle) {
        return feedForwardXvar(kalle, xInputs, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(parametersDual); // the output variable y

    VectorXd dydx = gradient(y, parametersDual);        // evaluate the gradient vector dy/dx

    std::cout << "y = " << y << std::endl;           // print the evaluated output y
    std::cout << "dy/dx = \n" << dydx << std::endl;  // print the evaluated gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}

std::vector<double> NeuralNetworkReverse::getTheGradientVector(std::vector<double> inputs)
{

    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();
    //VectorXvar xParameters = Eigen::Map<VectorXd>(parameters.data(), parameters.size()).cast<var>().array();
    /*
    VectorXvar x(inputSize);
    for (int i = 0; i < inputSize; i++) {
        x(i) = inputs[i];
    }

    VectorXvar xParameters(parameters.size());
    for (int i = 0; i < parameters.size(); i++) {
        xParameters(i) = parameters[i];
    }
*/
    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y

    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::cout << "y = " << y << std::endl;           // print the evaluated output y
    std::cout << "dy/dx = \n" << dydx << std::endl;  // print the evaluated gradient vector dy/dx

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
