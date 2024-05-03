#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <Eigen/Dense>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "../include/neural.h"

using namespace autodiff;
using namespace Eigen;
using namespace std;

NeuralNetworkSimple::NeuralNetworkSimple(std::vector<double> randNumbers, int inputSize, int hiddenSize)
    : parameters(randNumbers), inputSize(inputSize), hiddenSize(hiddenSize),
      parametersDual(Eigen::Map<VectorXd>(randNumbers.data(), randNumbers.size()).cast<dual>())
{
    gradientFunction = getGradientFunction();
}

dual feedForwardDual(VectorXdual parameters, VectorXdual inputsDual, int inputSize, int hiddenSize) {
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
}

dual NeuralNetworkSimple::feedForwardDual2(VectorXdual inputsDual) {
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

double NeuralNetworkSimple::feedForward(std::vector<double> inputs) {
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

std::function<VectorXdual(VectorXdual, VectorXdual)> NeuralNetworkSimple::getGradientFunction() {
    return [&](VectorXdual parametersDual, VectorXdual inputsDual) {
        auto feedForwardDual2Wrapper = [&](VectorXdual parametersDual) {
            this->parametersDual = parametersDual;
            return feedForwardDual2(inputsDual);
        };
        dual u;
        return gradient(feedForwardDual2Wrapper, wrt(parametersDual), at(parametersDual), u);
    };
}

VectorXdual NeuralNetworkSimple::getTheGradient(VectorXdual inputsDual)
{
    auto feedForwardWrapper = [&](VectorXdual kalle) {
        return feedForwardDual(kalle, inputsDual, inputSize, hiddenSize);
    };

    VectorXdual gradde = gradient(feedForwardWrapper, wrt(parametersDual), at(parametersDual));
    return gradde;
}

VectorXdual NeuralNetworkSimple::getTheGradientOnPositions(VectorXdual inputsDual)
{
    auto feedForwardWrapper = [&](VectorXdual inputs) {
        return feedForwardDual(parametersDual, inputs, inputSize, hiddenSize);
    };

    VectorXdual theGradient = gradient(feedForwardWrapper, wrt(inputsDual), at(inputsDual));

    return theGradient;
}
/*
VectorXdual NeuralNetworkSimple::getTheGradientOnPositions (std::vector<double> inputs)
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
//std::vector<double> NeuralNetworkSimple::getTheGradientOnPositions(std::vector<double> inputs)
VectorXdual NeuralNetworkSimple::getTheGradientOnPositions(std::vector<double> inputs)
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

std::vector<double> NeuralNetworkSimple::getTheGradientVector(std::vector<double> inputs)
{
    // Convert std::vector<double> to VectorXdual
    VectorXdual inputsDual = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<dual>();

    auto feedForwardWrapper = [&](VectorXdual kalle) {
        return feedForwardDual(kalle, inputsDual, inputSize, hiddenSize);
    };

    VectorXdual gradde = gradient(feedForwardWrapper, wrt(parametersDual), at(parametersDual));

    // Convert VectorXdual back to std::vector<double>
    std::vector<double> graddeVec(gradde.data(), gradde.data() + gradde.size());

    return graddeVec;
}
