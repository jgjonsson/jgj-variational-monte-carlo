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

NeuralNetworkReverse::NeuralNetworkReverse(std::vector<double> parameters, int inputSize, int hiddenSize)
    : parameters(parameters), inputSize(inputSize), hiddenSize(hiddenSize)
{
    parametersDual = Eigen::Map<VectorXd>(parameters.data(), parameters.size()).cast<var>().array();
    weightsSize = inputSize * hiddenSize + hiddenSize;
    inputLayerWeightsVar = parametersDual.head(inputSize * hiddenSize);
    hiddenLayerWeightsVar = parametersDual.segment(inputSize * hiddenSize, hiddenSize);
    hiddenLayerBiasesVar = parametersDual.tail(hiddenSize);

    inputLayerWeightsMatrix = Eigen::Map<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>>(inputLayerWeightsVar.data(), hiddenSize, inputSize).matrix();

    //Eigen::Map<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> inputLayerWeightsMatrix(inputLayerWeights.data(), hiddenSize, inputSize);
    //Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> weightMatrix = inputLayerWeightsMatrix.matrix();

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

var feedForwardVarVectorized(ArrayXvar parameters, ArrayXvar inputsVar, int inputSize, int hiddenSize) {
    int weightsSize = inputSize * hiddenSize + hiddenSize;
    ArrayXvar inputLayerWeights = parameters.segment(0, inputSize * hiddenSize);
    ArrayXvar hiddenLayerWeights = parameters.segment(inputSize * hiddenSize, hiddenSize);
    ArrayXvar hiddenLayerBiases = parameters.segment(inputSize * hiddenSize + hiddenSize, hiddenSize);

    // Reshape inputLayerWeights into a matrix
    Eigen::Map<Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>> inputLayerWeightsMatrix(inputLayerWeights.data(), hiddenSize, inputSize);
    Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> matrasse = inputLayerWeightsMatrix.matrix();
    auto hiddenOutputsBeforeActivation = inputLayerWeightsMatrix.matrix() * inputsVar.matrix() + hiddenLayerBiases.matrix();

    ArrayXvar hiddenOutputs(hiddenSize);
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        hiddenOutputs[i] = tanh(hiddenOutputsBeforeActivation[i]);
    }

    var finalOutput = hiddenOutputs.matrix().dot(hiddenLayerWeights.matrix());

    return finalOutput;
}

var feedForwardVarVectorizedFastest(
        const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& inputLayerWeightsMatrix,
        const ArrayXvar& hiddenLayerWeightsVar,
        const ArrayXvar& hiddenLayerBiasesVar,
        const VectorXvar& inputsVar,
        int inputSize, int hiddenSize) {
    auto hiddenOutputsBeforeActivation = inputLayerWeightsMatrix * inputsVar.matrix() + hiddenLayerBiasesVar.matrix();

    ArrayXvar hiddenOutputs(hiddenSize);
    for(int i = 0; i < hiddenOutputs.size(); i++) {
        hiddenOutputs[i] = tanh(hiddenOutputsBeforeActivation[i]);
    }
    //auto hiddenOutputs = hiddenOutputsBeforeActivation.unaryExpr([](const var& x) { return tanh(x); });

    var finalOutput = hiddenOutputs.matrix().dot(hiddenLayerWeightsVar.matrix());

    return finalOutput;
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

var feedForwardXvarParametersOptimized(const ArrayXvar& inputLayerWeights, const ArrayXvar& hiddenLayerWeights, const ArrayXvar& hiddenLayerBiases, const ArrayXvar& inputs, int inputSize, int hiddenSize) {

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


void NeuralNetworkReverse::backpropagate(std::vector<double> inputs, double targetOutput, double learningRate) {

    int weightsSize = inputSize * hiddenSize + hiddenSize;

    std::vector<double> inputLayerWeights(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
    std::vector<double> hiddenLayerWeights(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
    std::vector<double> hiddenLayerBiases(parameters.begin() + weightsSize, parameters.begin() + weightsSize + hiddenSize);

    // Calculate the outputs of the hidden layer
    std::vector<double> hiddenLayerOutputs(hiddenSize);
    for(int i = 0; i < hiddenLayerBiases.size(); i++) {
        double output = 0.0;
        for(int j = 0; j < inputs.size(); j++) {
            output += inputLayerWeights[j * hiddenLayerBiases.size() + i] * inputs[j];
        }
        output += hiddenLayerBiases[i];
        hiddenLayerOutputs[i] = tanh(output);
    }

    // Feed the inputs forward through the network
    double output = 0.0;
    for(int i = 0; i < hiddenLayerOutputs.size(); i++) {
        output += hiddenLayerWeights[i] * hiddenLayerOutputs[i];
    }

    // Calculate the error of the output
    double outputError = targetOutput - output;
//cout << "outputError: " << outputError << endl;

    // Compute gradient at output layer
    double outputGradient = outputError; // derivative of linear activation function is 1

    // Propagate error back to hidden layer
    std::vector<double> hiddenError(hiddenSize);
    for (int i = 0; i < hiddenSize; i++) {
        hiddenError[i] = outputGradient * hiddenLayerWeights[i];
    }

    // Compute gradient at hidden layer
    std::vector<double> hiddenGradient(hiddenSize);
    for (int i = 0; i < hiddenSize; i++) {
        hiddenGradient[i] = hiddenError[i] * (1 - hiddenLayerOutputs[i] * hiddenLayerOutputs[i]); // derivative of tanh
    }

    // Update weights and biases
    for (int i = 0; i < inputSize * hiddenSize; i++) {
        inputLayerWeights[i] += learningRate * hiddenGradient[i / hiddenSize] * inputs[i % inputSize];
    }
    for (int i = 0; i < hiddenSize; i++) {
        hiddenLayerWeights[i] += learningRate * outputGradient * hiddenLayerOutputs[i];
        hiddenLayerBiases[i] += learningRate * hiddenGradient[i];
    }

    // Map the updated weights and biases back to parameters
    std::copy(inputLayerWeights.begin(), inputLayerWeights.end(), parameters.begin());
    std::copy(hiddenLayerWeights.begin(), hiddenLayerWeights.end(), parameters.begin() + inputSize * hiddenSize);
    std::copy(hiddenLayerBiases.begin(), hiddenLayerBiases.end(), parameters.begin() + weightsSize);
}
/*
void backpropagate(std::vector<double> inputs, double targetOutput, double learningRate) {

    int weightsSize = inputSize * hiddenSize + hiddenSize;

    std::vector<double> inputLayerWeights(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
    std::vector<double> hiddenLayerWeights(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
    std::vector<double> hiddenLayerBiases(parameters.begin() + weightsSize, parameters.begin() + weightsSize + hiddenSize);


    // Feed the inputs forward through the network
    double output = feedForward(inputs);

    // Calculate the error of the output
    double outputError = targetOutput - output;

    // Compute gradient at output layer
    double outputGradient = outputError; // derivative of linear activation function is 1

    // Propagate error back to hidden layer
    std::vector<double> hiddenError(hiddenSize);
    for (int i = 0; i < hiddenSize; i++) {
        hiddenError[i] = outputGradient * hiddenLayerWeights[i];
    }

    // Compute gradient at hidden layer
    std::vector<double> hiddenGradient(hiddenSize);
    for (int i = 0; i < hiddenSize; i++) {
        hiddenGradient[i] = hiddenError[i] * (1 - hiddenLayerOutputs[i] * hiddenLayerOutputs[i]); // derivative of tanh
    }

    // Update weights and biases
    for (int i = 0; i < inputSize * hiddenSize; i++) {
        inputLayerWeights[i] += learningRate * hiddenGradient[i / hiddenSize] * inputs[i % inputSize];
    }
    for (int i = 0; i < hiddenSize; i++) {
        hiddenLayerWeights[i] += learningRate * outputGradient * hiddenLayerOutputs[i];
        //hiddenLayerWeights[i] += learningRate * outputGradient * inputs[i];
        hiddenLayerBiases[i] += learningRate * hiddenGradient[i];
    }

    // Map the updated weights and biases back to parameters
    std::copy(inputLayerWeights.begin(), inputLayerWeights.end(), parameters.begin());
    std::copy(hiddenLayerWeights.begin(), hiddenLayerWeights.end(), parameters.begin() + inputSize * hiddenSize);
    std::copy(hiddenLayerBiases.begin(), hiddenLayerBiases.end(), parameters.begin() + weightsSize);
}
*/
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

std::vector<double> NeuralNetworkReverse::getTheGradientVectorWrtParametersVectorized(std::vector<double> &inputs)
{
    VectorXvar xInputs = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& kalle) {
        return feedForwardVarVectorized(kalle, xInputs, inputSize, hiddenSize);
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
        return feedForwardXvarParametersOptimized(inputLayerWeightsVar, hiddenLayerWeightsVar, hiddenLayerBiasesVar, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}

std::vector<double> NeuralNetworkReverse::getTheGradientVectorWrtInputsVectorized(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardVarVectorizedFastest(inputLayerWeightsMatrix, hiddenLayerWeightsVar, hiddenLayerBiasesVar, inputsDual, inputSize, hiddenSize);
    };

    var y = feedForwardWrapper(x); // the output variable y
    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    std::vector<double> dydx_vec(dydx.data(), dydx.data() + dydx.size());

    return dydx_vec;
}

double NeuralNetworkReverse::getTheLaplacianFromGradient(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    double totalLaplacian = 0.0;
    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
    };

    var u = feedForwardWrapper(x); // the output variable y
    /*
    auto [ux] = derivativesx(u);//, x);//wrt(x));

    auto [uxx] = derivativesx(ux, wrt(x));

    //VectorXvar dydx = gradient(y, x); // evaluate the gradient vector dy/dx

    for (size_t i = 0; i < uxx.size(); ++i) {
        // Compute the derivative of the i-th component of the gradient with respect to the i-th input variable
        //auto derder = uxx[i];
        totalLaplacian += uxx[i];
        //totalLaplacian += derder;//second_derivative.val(); // Extract the value from the var type and add it to the total
    }
*/
    return totalLaplacian;
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

    double epsilon = 3e-4; // small number for finite difference

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
    Eigen::VectorXd g;
    Eigen::MatrixXd H = hessian(y, x, g); // evaluate the Hessian matrix H = d^2y/dx^2
    double laplacian2 = H.trace(); // evaluate the trace of the Hessian matrix
    double gradientSquared = g.squaredNorm(); // evaluate the squared norm of the gradient vector
    //auto [ux] = derivativesx(y, wrt(x));
    //std::vector<double> laplacian(H.diagonal().data(), H.diagonal().data() + H.diagonal().size());

    //double sum = std::accumulate(laplacian.begin(), laplacian.end(), 0.0);

    //cout << "Laplacian2: " << laplacian2 << /*" and summed laplacian " << sum <<*/ endl;
    //cout << "Gradient squared " << gradientSquared << endl;
    return laplacian2;
}

double NeuralNetworkReverse::laplacianOfLogarithmWrtInputs(std::vector<double> &inputs)
{
    VectorXvar x = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<var>().array();

    auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvarParametersOptimized(inputLayerWeightsVar, hiddenLayerWeightsVar, hiddenLayerBiasesVar, inputsDual, inputSize, hiddenSize);
    };
    /*auto feedForwardWrapper = [&](const VectorXvar& inputsDual) {
        return feedForwardXvar(parametersDual, inputsDual, inputSize, hiddenSize);
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
double NeuralNetworkReverse::getTheLaplacianVectorWrtInputs2(std::vector<double> &inputs)
{
return 0.0;

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
    return sum;
}
*/
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