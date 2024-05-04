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

// The scalar function for which the gradient is needed
var feedForwardDual(const ArrayXvar& parameters, const ArrayXvar& inputsDual, int inputSize, int hiddenSize) {
return sqrt((inputsDual * inputsDual).sum());
/*
    int weightsSize = inputSize * hiddenSize + hiddenSize;
    VectorXdual inputLayerWeights = parameters.segment(0, inputSize * hiddenSize);
    VectorXdual hiddenLayerWeights = parameters.segment(inputSize * hiddenSize, hiddenSize);
    VectorXdual hiddenLayerBiases = parameters.segment(inputSize * hiddenSize + hiddenSize, hiddenSize);

    // Reshape inputLayerWeights into a matrix
    Eigen::Map<MatrixXdual> inputLayerWeightsMatrix(inputLayerWeights.data(), hiddenSize, inputSize);

    auto hiddenOutputsBeforeActivation = inputLayerWeightsMatrix * inputsDual + hiddenLayerBiases;

    VectorXdual hiddenOutputs(hiddenSize);
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        hiddenOutputs[i] = tanh(hiddenOutputsBeforeActivation[i]);
    }

    dual finalOutput = hiddenOutputs.dot(hiddenLayerWeights);

    return finalOutput;
    */
}
/*
dual NeuralNetworkReverse::feedForwardDual2(VectorXdual inputsDual) {
    int weightsSize = inputSize * hiddenSize + hiddenSize;
    VectorXdual inputLayerWeights = parametersDual.segment(0, inputSize * hiddenSize);
    VectorXdual hiddenLayerWeights = parametersDual.segment(inputSize * hiddenSize, hiddenSize);
    VectorXdual hiddenLayerBiases = parametersDual.segment(inputSize * hiddenSize + hiddenSize, hiddenSize);

    // Reshape inputLayerWeights into a matrix
    Eigen::Map<MatrixXdual> inputLayerWeightsMatrix(inputLayerWeights.data(), hiddenSize, inputSize);
    auto hiddenOutputsBeforeActivation = inputLayerWeightsMatrix * inputsDual + hiddenLayerBiases;

    VectorXdual hiddenOutputs(hiddenSize);
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        hiddenOutputs[i] = tanh(hiddenOutputsBeforeActivation[i]);
    }

    dual finalOutput = hiddenOutputs.dot(hiddenLayerWeights);

    return finalOutput;
}
*/
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
/*
VectorXdual NeuralNetworkReverse::getTheGradient(VectorXdual inputsDual)
{
    auto feedForwardWrapper = [&](VectorXdual kalle) {
        return feedForwardDual(kalle, inputsDual, inputSize, hiddenSize);
    };

    VectorXdual gradde = gradient(feedForwardWrapper, wrt(parametersDual), at(parametersDual));
    return gradde;
}
*/
/*
VectorXdual NeuralNetworkReverse::getTheGradientOnPositions(VectorXdual inputsDual)
{
    auto feedForwardWrapper = [&](VectorXdual inputs2) {
        return feedForwardDual(parametersDual, inputs2, inputSize, hiddenSize);
    };

    VectorXdual theGradient = gradient(feedForwardWrapper, wrt(inputsDual), at(inputsDual));

    return theGradient;
}
*/
/*
VectorXdual NeuralNetworkReverse::getTheGradientOnPositions (std::vector<double> inputs)
//(VectorXdual inputsDual)
{
auto inputsDual = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<dual>();
    auto feedForwardWrapper = [&](VectorXdual kalle) {
        return feedForwardDual(parametersDual, kalle, inputSize, hiddenSize);
    };

    VectorXdual theGradient = gradient(feedForwardWrapper, wrt(inputsDual), at(inputsDual));

    return theGradient;
}*/
/*
//std::vector<double> NeuralNetworkReverse::getTheGradientOnPositions(std::vector<double> inputs)
VectorXdual NeuralNetworkReverse::getTheGradientOnPositions(std::vector<double> inputs)
{
    auto inputsDual = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<dual>();
    auto feedForwardWrapper = [&](VectorXdual kalle) {
        return feedForwardDual(parametersDual, kalle, inputSize, hiddenSize);
    };

    VectorXdual theGradient = gradient(feedForwardWrapper, wrt(inputsDual), at(inputsDual));


return theGradient;*/
/*
    std::vector<double> returnVector(theGradient.size());
    std::transform(theGradient.begin(), theGradient.end(), returnVector.begin(), [](const dual& d) { return d.val; });

    //std::vector<double> graddeVec = Eigen::Map<VectorXd>(gradde.unaryExpr([](const dual& x) { return val(x); }).data(), gradde.size()).cast<double>();
//    VectorXd graddeDouble = gradde.unaryExpr([](const dual& x) { return val(x); });
//    std::vector<double> graddeVec(graddeDouble.data(), graddeDouble.data() + graddeDouble.size());
    return returnVector;
    */
//}

std::vector<double> NeuralNetworkReverse::getTheGradientVector(std::vector<double> inputs)
{
    //
    VectorXvar x(inputSize);
    for (int i = 0; i < inputSize; i++) {
        x(i) = inputs[i];
    }

    VectorXvar xParameters(parameters.size());
    for (int i = 0; i < parameters.size(); i++) {
        xParameters(i) = parameters[i];
    }

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardDual(xParameters, inputsDual, inputSize, hiddenSize);
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
