#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <Eigen/Dense>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "../include/neural_twolayer.h"

using namespace autodiff;
using namespace Eigen;
using namespace std;

NeuralNetworkTwoLayers::NeuralNetworkTwoLayers(std::vector<double> parameters, int inputSize, int hiddenSize)
    : parameters(parameters), inputSize(inputSize), hiddenSize(hiddenSize)
{
    parametersVar = Eigen::Map<VectorXd>(parameters.data(), parameters.size()).cast<var>().array();
    cacheWeightsAndBiases(parameters);
/*
    //Set up variables for all the sizes of weights and biases.
    int inputLayerWeightsSize = inputSize * hiddenSize;
    int hiddenLayerWeightsSize = hiddenSize * hiddenSize;
    int secondHiddenLayerWeightsSize = hiddenSize;
    int allWeightsSize = inputLayerWeightsSize + hiddenLayerWeightsSize + secondHiddenLayerWeightsSize;
    int hiddenLayerBiasesSize = hiddenSize;
    int secondHiddenLayerBiasesSize = hiddenSize;


   // weightsSize = hiddenSize*hiddenSize;//inputSize * hiddenSize + hiddenSize;
//cout << "Banan " << parameters.size() << " of " << weightsSize << endl;

    inputLayerWeightsDouble = std::vector<double>(parameters.begin(), parameters.begin() + inputLayerWeightsSize);
    hiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputLayerWeightsSize, parameters.begin() + inputLayerWeightsSize + hiddenLayerWeightsSize);
    secondHiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputLayerWeightsSize + hiddenLayerWeightsSize, parameters.begin() + allWeightsSize);
    hiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + allWeightsSize, parameters.begin() + allWeightsSize + secondHiddenLayerWeightsSize);
    secondHiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + allWeightsSize + secondHiddenLayerWeightsSize, parameters.end());

    inputLayerWeightsVar = parametersVar.head(inputLayerWeightsSize);
    hiddenLayerWeightsVar = parametersVar.segment(inputLayerWeightsSize, hiddenLayerWeightsSize);
    secondHiddenLayerWeightsVar = parametersVar.segment(inputLayerWeightsSize + hiddenLayerWeightsSize, secondHiddenLayerWeightsSize);
    cout << "Hax " << endl;
    hiddenLayerBiasesVar = parametersVar.segment(allWeightsSize, hiddenLayerBiasesSize);
    secondHiddenLayerBiasesVar = parametersVar.tail(hiddenSize);
    cout << "Banarna " << endl;*/
}

void NeuralNetworkTwoLayers::cacheWeightsAndBiases(std::vector<double> parameters) {
    int inputLayerWeightsSize = inputSize * hiddenSize;
    int hiddenLayerWeightsSize = hiddenSize * hiddenSize;
    int secondHiddenLayerWeightsSize = hiddenSize;
    int allWeightsSize = inputLayerWeightsSize + hiddenLayerWeightsSize + secondHiddenLayerWeightsSize;
    int hiddenLayerBiasesSize = hiddenSize;
    int secondHiddenLayerBiasesSize = hiddenSize;

    inputLayerWeightsDouble = std::vector<double>(parameters.begin(), parameters.begin() + inputLayerWeightsSize);
    hiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputLayerWeightsSize, parameters.begin() + inputLayerWeightsSize + hiddenLayerWeightsSize);
    secondHiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputLayerWeightsSize + hiddenLayerWeightsSize, parameters.begin() + allWeightsSize);
    hiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + allWeightsSize, parameters.begin() + allWeightsSize + secondHiddenLayerWeightsSize);
    secondHiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + allWeightsSize + secondHiddenLayerWeightsSize, parameters.end());

    inputLayerWeightsVar = parametersVar.head(inputLayerWeightsSize);
    hiddenLayerWeightsVar = parametersVar.segment(inputLayerWeightsSize, hiddenLayerWeightsSize);
    secondHiddenLayerWeightsVar = parametersVar.segment(inputLayerWeightsSize + hiddenLayerWeightsSize, secondHiddenLayerWeightsSize);
    hiddenLayerBiasesVar = parametersVar.segment(allWeightsSize, hiddenLayerBiasesSize);
    secondHiddenLayerBiasesVar = parametersVar.tail(hiddenSize);
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

var feedForwardXvarTwoLayer(const ArrayXvar& parameters, const ArrayXvar& inputs, int inputSize, int hiddenSize) {
cout << "Apa " << endl;
    int inputLayerWeightsSize = inputSize * hiddenSize;
    int hiddenLayerWeightsSize = hiddenSize * hiddenSize;
    int secondHiddenLayerWeightsSize = hiddenSize;
    int allWeightsSize = inputLayerWeightsSize + hiddenLayerWeightsSize + secondHiddenLayerWeightsSize;
    int hiddenLayerBiasesSize = hiddenSize;
    int secondHiddenLayerBiasesSize = hiddenSize;


cout << "inputLayerWeights.size() = " << inputLayerWeightsSize << endl;
cout << "hiddenLayerWeights.size() = " << hiddenLayerWeightsSize << endl;
cout << "secondHiddenLayerWeights.size() = " << secondHiddenLayerWeightsSize << endl;
cout << "hiddenLayerBiases.size() = " << hiddenLayerBiasesSize << endl;
cout << "secondHiddenLayerBiases.size() = " << secondHiddenLayerBiasesSize << endl;

/*
    inputLayerWeightsDouble = std::vector<double>(parameters.begin(), parameters.begin() + inputLayerWeightsSize);
    hiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputLayerWeightsSize, parameters.begin() + inputLayerWeightsSize + hiddenLayerWeightsSize);
    secondHiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputLayerWeightsSize + hiddenLayerWeightsSize, parameters.begin() + allWeightsSize);

    hiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + allWeightsSize, parameters.begin() + allWeightsSize + secondHiddenLayerWeightsSize);
    secondHiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + allWeightsSize + secondHiddenLayerWeightsSize, parameters.end());
    */
    ArrayXvar inputLayerWeights = parameters.head(inputLayerWeightsSize);
    ArrayXvar hiddenLayerWeights = parameters.segment(inputLayerWeightsSize, hiddenLayerWeightsSize);
    ArrayXvar secondHiddenLayerWeights = parameters.segment(inputLayerWeightsSize + hiddenLayerWeightsSize, secondHiddenLayerWeightsSize);
    ArrayXvar hiddenLayerBiases = parameters.segment(allWeightsSize, hiddenLayerBiasesSize);
    ArrayXvar secondHiddenLayerBiases = parameters.tail(hiddenSize);

    /*
    int weightsSize = weightsSize = hiddenSize*hiddenSize;//inputSize * hiddenSize + hiddenSize;
    ArrayXvar inputLayerWeights = parameters.head(inputSize * hiddenSize);
    ArrayXvar hiddenLayerWeights = parameters.segment(inputSize * hiddenSize, hiddenSize*hiddenSize);
    ArrayXvar hiddenLayerBiases = parameters.segment(inputSize * hiddenSize+hiddenSize*hiddenSize, hiddenSize);
    ArrayXvar secondHiddenLayerWeights = parameters.segment(inputSize * hiddenSize+hiddenSize*hiddenSize + hiddenSize, hiddenSize * hiddenSize);
    ArrayXvar secondHiddenLayerBiases = parameters.tail(hiddenSize);
    */
cout << "Apanson " << endl;
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

    ArrayXvar secondHiddenOutputs(hiddenLayerBiases.size());
    for(int i = 0; i < secondHiddenLayerBiases.size(); i++) {
        var output = 0.0;
        for(int j = 0; j < hiddenOutputs.size(); j++) {
            output += hiddenLayerWeights[j * secondHiddenLayerBiases.size() + i] * hiddenOutputs[j];
        }
        output += secondHiddenLayerBiases[i];
        secondHiddenOutputs[i] = tanh(output);
        //secondHiddenOutputs[i] = leaky_relu(output);
    }

    var finalOutput = 0.0;
    for(int i = 0; i < secondHiddenOutputs.size(); i++) {

        //finalOutput += hiddenLayerWeightsDouble[i] * hiddenOutputs[i];
        finalOutput += secondHiddenLayerWeights[i] * secondHiddenOutputs[i];
    }
    return finalOutput;
}

var feedForwardXvarParametersPrecalculatedTwoLayer(const ArrayXvar& inputLayerWeights,
const ArrayXvar& hiddenLayerWeights,
const ArrayXvar& secondHiddenLayerWeights,
const ArrayXvar& hiddenLayerBiases,
const ArrayXvar& secondHiddenLayerBiases,
const ArrayXvar& inputs, int inputSize, int hiddenSize) {

//Print length of all parameters:
cout << "inputLayerWeights.size() = " << inputLayerWeights.size() << endl;
cout << "hiddenLayerWeights.size() = " << hiddenLayerWeights.size() << endl;
cout << "secondHiddenLayerWeights.size() = " << secondHiddenLayerWeights.size() << endl;
cout << "hiddenLayerBiases.size() = " << hiddenLayerBiases.size() << endl;
cout << "secondHiddenLayerBiases.size() = " << secondHiddenLayerBiases.size() << endl;


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

    ArrayXvar secondHiddenOutputs(hiddenLayerBiases.size());
    for(int i = 0; i < secondHiddenLayerBiases.size(); i++) {
        var output = 0.0;
        for(int j = 0; j < hiddenOutputs.size(); j++) {
            output += hiddenLayerWeights[j * secondHiddenLayerBiases.size() + i] * hiddenOutputs[j];
        }
        output += secondHiddenLayerBiases[i];
        secondHiddenOutputs[i] = tanh(output);
        //secondHiddenOutputs[i] = leaky_relu(output);
    }

    var finalOutput = 0.0;
    for(int i = 0; i < secondHiddenOutputs.size(); i++) {
        finalOutput += secondHiddenLayerWeights[i] * secondHiddenOutputs[i];
    }
    return finalOutput;
}

double NeuralNetworkTwoLayers::feedForward(std::vector<double> inputs) {
    int weightsSize = weightsSize = hiddenSize*hiddenSize;//inputSize * hiddenSize + hiddenSize;

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

    std::vector<double> secondHiddenOutputs;
    for(int i = 0; i < secondHiddenLayerBiasesDouble.size(); i++) {
        double output = 0.0;
        for(int j = 0; j < hiddenOutputs.size(); j++) {
            output += hiddenLayerWeightsDouble[j * secondHiddenLayerBiasesDouble.size() + i] * hiddenOutputs[j];
        }
        output += secondHiddenLayerBiasesDouble[i];
        secondHiddenOutputs.push_back(tanh(output));
        //secondHiddenOutputs.push_back(leaky_relu(output));
    }

    double finalOutput = 0.0;
    for(int i = 0; i < secondHiddenOutputs.size(); i++) {
        finalOutput += secondHiddenLayerWeightsDouble[i] * secondHiddenOutputs[i];
    }
    return finalOutput;
}


std::vector<double> NeuralNetworkTwoLayers::getTheGradientVectorWrtParameters(std::vector<double> &inputs)
{
    VectorXvar xInputs = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& parametersDiffVariable) {
        return feedForwardXvarTwoLayer(parametersDiffVariable, xInputs, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(parametersVar); // the output variable y
    VectorXd dydx = gradient(y, parametersVar);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}

std::vector<double> NeuralNetworkTwoLayers::getTheGradientVectorWrtInputs(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarParametersPrecalculatedTwoLayer(inputLayerWeightsVar, hiddenLayerWeightsVar, secondHiddenLayerWeightsVar, hiddenLayerBiasesVar, secondHiddenLayerBiasesVar, inputsVar, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());
cout << "I am the one layer  << "<< endl;
    return dydx_vec;
}

/* Calculating a single derivative of log(psi) with respect to one particle's and one dimension's position
This is used when calculating quantum force for one particle*/
double NeuralNetworkTwoLayers::calculateNumericalDeriviateWrtInput(std::vector<double>& inputs, int inputIndexForDerivative) {
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

double NeuralNetworkTwoLayers::laplacianOfLogarithmWrtInputs(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarParametersPrecalculatedTwoLayer(inputLayerWeightsVar, hiddenLayerWeightsVar, secondHiddenLayerWeightsVar, hiddenLayerBiasesVar, secondHiddenLayerBiasesVar, inputsVar, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    Eigen::VectorXd g;
    Eigen::MatrixXd H = hessian(y, x, g); // evaluate the Hessian matrix H = d^2y/dx^2
    double laplacian2 = H.trace(); // evaluate the trace of the Hessian matrix
    double gradientSquared = g.squaredNorm(); // evaluate the squared norm of the gradient vector

    return laplacian2+gradientSquared;
}

std::vector<double> NeuralNetworkTwoLayers::generateRandomParameterSetTwoLayers(size_t rbs_M, size_t rbs_N, int randomSeed, double spread)
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

    //Number of parameters for one layer
    int numberParameters = inputNodes * hiddenNodes + hiddenNodes * 2;
    //Additional number of parameters for two layers
    numberParameters += hiddenNodes * hiddenNodes + hiddenNodes;

    for (size_t i = 0; i < numberParameters; i++){
        parameters.push_back(distribution(generator));
    }
    return parameters;
}