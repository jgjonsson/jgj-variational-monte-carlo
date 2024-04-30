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

/*
class NeuralNetworkSimple {
public:
    VectorXdual parametersDual;
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;
    int outputSize = 1;
    std::function<VectorXdual(VectorXdual, VectorXdual)> gradientFunction;*/

NeuralNetworkSimple::NeuralNetworkSimple(std::vector<double> randNumbers, int inputSize, int hiddenSize)
    : parameters(randNumbers), inputSize(inputSize), hiddenSize(hiddenSize),
      parametersDual(Eigen::Map<VectorXd>(randNumbers.data(), randNumbers.size()).cast<dual>())
{
    gradientFunction = getGradientFunction();
}

    dual feedForwardDual(VectorXdual kalle, VectorXdual inputsDual, int inputSize, int hiddenSize) {
        int weightsSize = inputSize * hiddenSize + hiddenSize;
        VectorXdual inputLayerWeights = kalle.segment(0, inputSize * hiddenSize);
        VectorXdual hiddenLayerWeights = kalle.segment(inputSize * hiddenSize, hiddenSize);
        VectorXdual hiddenLayerBiases = kalle.segment(inputSize * hiddenSize + hiddenSize, hiddenSize);

        // Reshape inputLayerWeights into a matrix
//        std::cout << "Size of inputLayerWeights: " << inputLayerWeights.size() << std::endl;
//        std::cout << "Value of hiddenSize: " << hiddenSize << std::endl;

        Eigen::Map<MatrixXdual> inputLayerWeightsMatrix(inputLayerWeights.data(), hiddenSize, inputSize);

        // Print dimensions
//        std::cout << "Dimensions of inputLayerWeightsMatrix: " << inputLayerWeightsMatrix.rows() << " x " << inputLayerWeightsMatrix.cols() << std::endl;
//        std::cout << "Size of inputsDual: " << inputsDual.size() << std::endl;
//        std::cout << "Size of hiddenLayerBiases: " << hiddenLayerBiases.size() << std::endl;

        auto hiddenOutputsBeforeActivation = inputLayerWeightsMatrix * inputsDual + hiddenLayerBiases;
//        std::cout << "Size of hiddenOutputsBeforeActivation: " << hiddenOutputsBeforeActivation.size() << std::endl;

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
//        std::cout << "Size of inputLayerWeights: " << inputLayerWeights.size() << std::endl;
//        std::cout << "Value of hiddenSize: " << hiddenSize << std::endl;

        Eigen::Map<MatrixXdual> inputLayerWeightsMatrix(inputLayerWeights.data(), hiddenSize, inputSize);

        // Print dimensions
//        std::cout << "Dimensions of inputLayerWeightsMatrix: " << inputLayerWeightsMatrix.rows() << " x " << inputLayerWeightsMatrix.cols() << std::endl;
//        std::cout << "Size of inputsDual: " << inputsDual.size() << std::endl;
//        std::cout << "Size of hiddenLayerBiases: " << hiddenLayerBiases.size() << std::endl;

        auto hiddenOutputsBeforeActivation = inputLayerWeightsMatrix * inputsDual + hiddenLayerBiases;
//        std::cout << "Size of hiddenOutputsBeforeActivation: " << hiddenOutputsBeforeActivation.size() << std::endl;

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
        /*cout << "Differentiating with parameters being: ";
        for(const auto& value : parametersDual) {
            cout << value << " ";
        }*/
        cout << endl;
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
        //VectorXdual getGradient(VectorXdual inputsDual)
    {
        auto feedForwardWrapper = [&](VectorXdual kalle) {
            return feedForwardDual(kalle, inputsDual, inputSize, hiddenSize);
        };

        VectorXdual gradde = gradient(feedForwardWrapper, wrt(parametersDual), at(parametersDual));
    /*
        cout << "hej" << endl;
        VectorXdual gradientSymbolic = gradientFunction(parametersDual, inputsDual);
        return gradientSymbolic;*/

        return gradde;
    }

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


    void NeuralNetworkSimple::printParameters() {
        int weightsSize = inputSize * hiddenSize + hiddenSize;
        int biasesSize = inputSize + hiddenSize;

        std::vector<double> inputLayerWeights(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
        std::vector<double> hiddenLayerWeights(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
        std::vector<double> inputLayerBiases(parameters.begin() + weightsSize, parameters.begin() + weightsSize + inputSize);
        std::vector<double> hiddenLayerBiases(parameters.begin() + weightsSize + inputSize, parameters.begin() + weightsSize + biasesSize);
        double outputNeuronWeight = parameters[weightsSize + biasesSize];
        double outputNeuronBias = parameters[weightsSize + biasesSize + 1];

        std::cout << "Input Layer Weights: ";
        for(const auto& weight : inputLayerWeights) {
            std::cout << weight << " ";
        }
        std::cout << "\nHidden Layer Weights: ";
        for(const auto& weight : hiddenLayerWeights) {
            std::cout << weight << " ";
        }
        std::cout << "\nOutput Neuron Weight: " << outputNeuronWeight;

        std::cout << "\n\nInput Layer Biases: ";
        for(const auto& bias : inputLayerBiases) {
            std::cout << bias << " ";
        }
        std::cout << "\nHidden Layer Biases: ";
        for(const auto& bias : hiddenLayerBiases) {
            std::cout << bias << " ";
        }
        std::cout << "\nOutput Neuron Bias: " << outputNeuronBias << "\n";
    }
void NeuralNetworkSimple::printParameters2() {
    int weightsSize = inputSize * hiddenSize + hiddenSize;
    int biasesSize = inputSize + hiddenSize;

    std::vector<double> inputLayerWeights(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
    std::vector<double> hiddenLayerWeights(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
    std::vector<double> inputLayerBiases(parameters.begin() + weightsSize, parameters.begin() + weightsSize + inputSize);
    std::vector<double> hiddenLayerBiases(parameters.begin() + weightsSize + inputSize, parameters.begin() + weightsSize + biasesSize);
    double outputNeuronWeight = parameters[weightsSize + biasesSize];
    double outputNeuronBias = parameters[weightsSize + biasesSize + 1];

    std::cout << "Input Layer Weights:\n";
    for(int i = 0; i < inputSize; i++) {
        std::cout << "{ ";
        for(int j = 0; j < hiddenSize; j++) {
            std::cout << inputLayerWeights[i * hiddenSize + j] << " ";
        }
        std::cout << "}\n";
    }

    std::cout << "\nHidden Layer Weights:\n{ ";
    for(int i = 0; i < hiddenSize; i++) {
        std::cout << hiddenLayerWeights[i] << " ";
    }
    std::cout << "}\nOutput Neuron Weight: " << outputNeuronWeight;

    std::cout << "\n\nInput Layer Biases:\n{ ";
    for(const auto& bias : inputLayerBiases) {
        std::cout << bias << " ";
    }
    std::cout << "}\nHidden Layer Biases:\n{ ";
    for(const auto& bias : hiddenLayerBiases) {
        std::cout << bias << " ";
    }
    std::cout << "}\nOutput Neuron Bias: " << outputNeuronBias << "\n";
}

