#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using namespace autodiff;
using namespace Eigen;

#include <Eigen/Dense>

using namespace std;
using namespace autodiff;



class NeuralNetwork {
public:
    VectorXdual parametersDual;
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;
    int outputSize = 1;

NeuralNetwork(std::vector<double> randNumbers, int inputSize, int hiddenSize)
    : parameters(randNumbers), inputSize(inputSize), hiddenSize(hiddenSize),
      parametersDual(Eigen::Map<VectorXd>(randNumbers.data(), randNumbers.size()).cast<dual>())
{}
dual feedForwardDual2(VectorXdual inputsDual) {
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

    double feedForward(std::vector<double> inputs) {
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

    std::function<VectorXdual(VectorXdual, VectorXdual)> getGradientFunction() {
        return [&](VectorXdual parametersDual, VectorXdual inputsDual) {
            auto feedForwardDual2Wrapper = [&](VectorXdual parametersDual) {
                this->parametersDual = parametersDual;
                return this->feedForwardDual2(inputsDual);
            };
            dual u;
            return gradient(feedForwardDual2Wrapper, wrt(parametersDual), at(parametersDual), u);
        };
    }

    VectorXd getGradient(VectorXdual inputsDual) {
        dual u;

        auto feedForwardDual2Wrapper = [&](VectorXdual parametersDual) {
            this->parametersDual = parametersDual;
            return this->feedForwardDual2(inputsDual);
        };

        VectorXd g = gradient(feedForwardDual2Wrapper, wrt(parametersDual), at(parametersDual), u);
        //VectorXd g = gradient(feedForwardDual2, wrt(parametersDual), at(inputsDual), u); // evaluate the function value u and its gradient vector g = du/dx
        std::cout << "u = " << u << std::endl;      // print the evaluated output u
        //std::cout << "g = \n" << g << std::endl;    // print the evaluated gradient vector g = du/dx
        return g;
    }

    void backpropagate(std::vector<double> inputs, double targetOutput, double learningRate) {

            int weightsSize = inputSize * hiddenSize + hiddenSize;
            int biasesSize = inputSize + hiddenSize;

            std::vector<double> inputLayerWeights(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
            std::vector<double> hiddenLayerWeights(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
            std::vector<double> inputLayerBiases(parameters.begin() + weightsSize, parameters.begin() + weightsSize + inputSize);
            std::vector<double> hiddenLayerBiases(parameters.begin() + weightsSize + inputSize, parameters.begin() + weightsSize + biasesSize);
            double outputNeuronWeight = parameters[weightsSize + biasesSize];
            double outputNeuronBias = parameters[weightsSize + biasesSize + 1];

        // Feed the inputs forward through the network
        double output = feedForward(inputs);

        // Calculate the error of the output
        double outputError = targetOutput - output;

        // Adjust the weights and biases of the neurons in the output layer
        for(int i = 0; i < hiddenLayerBiases.size(); i++) {
            double error = outputError * hiddenLayerWeights[i];
            hiddenLayerWeights[i] += learningRate * error * inputs[i];
            hiddenLayerBiases[i] += learningRate * error;
        }
        outputNeuronWeight += learningRate * outputError * hiddenLayerBiases.back();
        outputNeuronBias += learningRate * outputError;

        // Calculate the error of the hidden layer
        std::vector<double> hiddenErrors;
        for(int i = 0; i < inputLayerBiases.size(); i++) {
            double error = 0.0;
            for(int j = 0; j < hiddenLayerWeights.size(); j++) {
                error += hiddenLayerWeights[j] * outputError;
            }
            hiddenErrors.push_back(error);
        }

        // Adjust the weights and biases of the neurons in the hidden layer
        for(int i = 0; i < inputLayerBiases.size(); i++) {
            for(int j = 0; j < inputs.size(); j++) {
                inputLayerWeights[i * inputs.size() + j] += learningRate * hiddenErrors[i] * inputs[j];
            }
            inputLayerBiases[i] += learningRate * hiddenErrors[i];
        }

        // Map the updated weights and biases back to parameters
        std::copy(inputLayerWeights.begin(), inputLayerWeights.end(), parameters.begin());
        std::copy(hiddenLayerWeights.begin(), hiddenLayerWeights.end(), parameters.begin() + inputSize * hiddenSize);
        std::copy(inputLayerBiases.begin(), inputLayerBiases.end(), parameters.begin() + inputSize * hiddenSize + hiddenSize);
        std::copy(hiddenLayerBiases.begin(), hiddenLayerBiases.end(), parameters.begin() + inputSize * hiddenSize + hiddenSize + inputSize);
        parameters[inputSize * hiddenSize + hiddenSize + inputSize + hiddenSize] = outputNeuronWeight;
        parameters[inputSize * hiddenSize + hiddenSize + inputSize + hiddenSize + 1] = outputNeuronBias;

        // Map the updated parameters back to parametersDual
        parametersDual = Eigen::Map<VectorXd>(parameters.data(), parameters.size()).cast<dual>();
    }


    void printParameters() {
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
void printParameters2() {
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

};

int main2() {
    srand((unsigned) time(NULL)); // Initialize random seed

    int inputNodes = 4;
    int hiddenNodes = 4;

    std::vector<double> randNumbers;
    //int totalNumbers = 4 * 4 + 4 + 1 + 4 + 4 * 4 + 1; // Total number of weights and biases for the neural network
    int totalNumbers = inputNodes * hiddenNodes + hiddenNodes + 1 + inputNodes + hiddenNodes * inputNodes + 1; // Total number of weights and biases for the neural network
    //double targetOutput = 0.714;
    double targetOutput = 0.612;
    double learningRate = 0.0005;

    for(int i = 0; i < totalNumbers; i++) {
        double randNumber = static_cast<double>(rand()) / RAND_MAX / 1000;
        randNumbers.push_back(randNumber);
    }

    NeuralNetwork neuralNetwork(randNumbers, inputNodes, hiddenNodes);

    std::vector<double> inputs = {0.1, 0.2, 0.3, 0.4};


    for(int i = 0; i < 50000; i++) { // Training for 1000 iterations
        double output = neuralNetwork.feedForward(inputs);
        neuralNetwork.backpropagate(inputs, targetOutput, learningRate);
        if(i % 1000 == 0) { // Print the output every 100 iterations
            std::cout << "Iteration: " << i << " Output: " << output << std::endl;
        }
    }



    neuralNetwork.printParameters2();

    double output = neuralNetwork.feedForward(inputs);
    std::cout << "Output: " << output << std::endl;

VectorXdual inputsDual = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<dual>();


    dual outputDual = neuralNetwork.feedForwardDual2(inputsDual);
    std::cout << "Output Dual: " << outputDual << std::endl;

    auto gradientFunction = neuralNetwork.getGradientFunction();
    auto theGradient = gradientFunction(neuralNetwork.parametersDual, inputsDual);

    //auto theGradient = neuralNetwork.getGradient(inputsDual);
    cout << "Gradient: " << theGradient.transpose() << endl;
//    neuralNetwork.printParameters();
//        std::cout << "Sweet: " << output << std::endl;


    return 0;
}

