#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <Eigen/Dense>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "../include/neural_onelayer.h"

using namespace autodiff;
using namespace Eigen;
using namespace std;

NeuralNetworkOneLayer::NeuralNetworkOneLayer(std::vector<double> parameters, int inputSize, int hiddenSize)
    : parameters(parameters), inputSize(inputSize), hiddenSize(hiddenSize)
{
    parametersVar = Eigen::Map<VectorXd>(parameters.data(), parameters.size()).cast<var>().array();
    weightsSize = inputSize * hiddenSize + hiddenSize;

    inputLayerWeightsVar = parametersVar.head(inputSize * hiddenSize);
    hiddenLayerWeightsVar = parametersVar.segment(inputSize * hiddenSize, hiddenSize);
    hiddenLayerBiasesVar = parametersVar.tail(hiddenSize);

    inputLayerWeightsDouble = std::vector<double>(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
    hiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
    hiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + weightsSize, parameters.begin() + weightsSize + hiddenSize);
}

//using ArrayXXvar = Eigen::Array<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>;


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

var feedForwardXvarOneLayer(const ArrayXvar& parameters, const ArrayXvar& inputs, int inputSize, int hiddenSize) {

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

var feedForwardXvarParametersPrecalculatedOneLayer(const ArrayXvar& inputLayerWeights, const ArrayXvar& hiddenLayerWeights, const ArrayXvar& hiddenLayerBiases, const ArrayXvar& inputs, int inputSize, int hiddenSize) {

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

double NeuralNetworkOneLayer::feedForward(std::vector<double> inputs) {
    int weightsSize = inputSize * hiddenSize + hiddenSize;

    std::vector<double> hiddenOutputs;
    for(int i = 0; i < hiddenLayerBiasesDouble.size(); i++) {
        double output = 0.0;
        for(int j = 0; j < inputs.size(); j++) {
            //output += inputLayerWeightsDouble[i * inputs.size() + j] * inputs[j];
            output += inputLayerWeightsDouble[j * hiddenLayerBiasesDouble.size() + i] * inputs[j];
        }
        output += hiddenLayerBiasesDouble[i];
        hiddenOutputs.push_back(tanh(output));
        //hiddenOutputs.push_back(leaky_relu(output));
    }

    double finalOutput = 0.0;
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        finalOutput += hiddenLayerWeightsDouble[i] * hiddenOutputs[i];
    }
    return finalOutput;
}


std::vector<double> NeuralNetworkOneLayer::getTheGradientVectorWrtParameters(std::vector<double> &inputs)
{
    VectorXvar xInputs = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& parametersDiffVariable) {
        return feedForwardXvarOneLayer(parametersDiffVariable, xInputs, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(parametersVar); // the output variable y
    VectorXd dydx = gradient(y, parametersVar);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}

std::vector<double> NeuralNetworkOneLayer::getTheGradientVectorWrtInputs(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarParametersPrecalculatedOneLayer(inputLayerWeightsVar, hiddenLayerWeightsVar, hiddenLayerBiasesVar, inputsVar, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());
cout << "I am the one layer  << "<< endl;
    return dydx_vec;
}

/* Calculating a single derivative of log(psi) with respect to one particle's and one dimension's position
This is used when calculating quantum force for one particle*/
double NeuralNetworkOneLayer::calculateNumericalDeriviateWrtInput(std::vector<double>& inputs, int inputIndexForDerivative) {
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

double NeuralNetworkOneLayer::laplacianOfLogarithmWrtInputs(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarParametersPrecalculatedOneLayer(inputLayerWeightsVar, hiddenLayerWeightsVar, hiddenLayerBiasesVar, inputsVar, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    Eigen::VectorXd g;
    Eigen::MatrixXd H = hessian(y, x, g); // evaluate the Hessian matrix H = d^2y/dx^2
    double laplacian2 = H.trace(); // evaluate the trace of the Hessian matrix
    double gradientSquared = g.squaredNorm(); // evaluate the squared norm of the gradient vector

    return laplacian2+gradientSquared;
}