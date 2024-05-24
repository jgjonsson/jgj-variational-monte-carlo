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

#include "../include/neural_onelayer.h"

using namespace autodiff;
using namespace Eigen;
using namespace std;

NeuralNetworkOneLayer::NeuralNetworkOneLayer(std::vector<double> parameters, int inputSize, int hiddenSize)
    : parameters(parameters), inputSize(inputSize), hiddenSize(hiddenSize)
{
    parametersVar = Eigen::Map<VectorXd>(parameters.data(), parameters.size()).cast<var>().array();
    weightsSize = inputSize * hiddenSize + hiddenSize;
    /*
    inputLayerWeightsVar = parametersVar.head(inputSize * hiddenSize);
    hiddenLayerWeightsVar = parametersVar.segment(inputSize * hiddenSize, hiddenSize);
    hiddenLayerBiasesVar = parametersVar.tail(hiddenSize);

    inputLayerWeightsMatrix = Eigen::Map<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>>(inputLayerWeightsVar.data(), hiddenSize, inputSize).matrix();
*/
    //Eigen::Map<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> inputLayerWeightsMatrix(inputLayerWeights.data(), hiddenSize, inputSize);
    //Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> weightMatrix = inputLayerWeightsMatrix.matrix();
//cout << "I am the one layer  << "<< endl;
    inputLayerWeightsDouble = std::vector<double>(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
    hiddenLayerWeightsDouble = std::vector<double>(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
    hiddenLayerBiasesDouble = std::vector<double>(parameters.begin() + weightsSize, parameters.begin() + weightsSize + hiddenSize);
/*
    std::vector<double> inputLayerWeightsDouble(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
    std::vector<double> hiddenLayerWeightsDouble(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
    std::vector<double> hiddenLayerBiasesDouble(parameters.begin() + weightsSize, parameters.begin() + weightsSize + hiddenSize);
*/
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

/*
double NeuralNetworkOneLayer::getTheLaplacianFromGradient(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    double totalLaplacian = 0.0;
    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarOneLayer(parametersVar, inputsVar, inputSize, hiddenSize);
    };

    var u = feedForwardWrapper(x); // the output variable y

    return totalLaplacian;
}*/
/*
std::vector<double> NeuralNetworkOneLayer::calculateNumericalGradientParameters(std::vector<double>& inputs) {
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
*/
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
    /*auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarOneLayer(parametersVar, inputsVar, inputSize, hiddenSize);
    };
*/
    var y = feedForwardWrapper(x); // the output variable y
    Eigen::VectorXd g;
    Eigen::MatrixXd H = hessian(y, x, g); // evaluate the Hessian matrix H = d^2y/dx^2
    double laplacian2 = H.trace(); // evaluate the trace of the Hessian matrix
    double gradientSquared = g.squaredNorm(); // evaluate the squared norm of the gradient vector

    //cout << "Laplacian2: " << laplacian2 << endl;
    //cout << "Gradient squared " << gradientSquared << endl;
    return laplacian2+gradientSquared;
}
/*
double NeuralNetworkOneLayer::getTheLaplacianVectorWrtInputs2(std::vector<double> &inputs)
{
return 0.0;

    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarOneLayer(parametersVar, inputsVar, inputSize, hiddenSize);
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
    return sum;
}
*/
/*
double NeuralNetworkOneLayer::getTheTotalLaplacian(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarOneLayer(parametersVar, inputsVar, inputSize, hiddenSize);
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
std::vector<double> NeuralNetworkOneLayer::getTheGradientVectorWrtInputs(std::vector<double> inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsVar) {
        return feedForwardXvarOneLayer(parametersVar, inputsVar, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}
*/