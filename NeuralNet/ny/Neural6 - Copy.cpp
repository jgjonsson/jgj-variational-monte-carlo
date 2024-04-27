#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

using namespace std;



class NeuralNetwork {
public:
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;

    NeuralNetwork(std::vector<double> randNumbers, int inputSize, int hiddenSize)
        : parameters(randNumbers), inputSize(inputSize), hiddenSize(hiddenSize) {}

    double feedForward(std::vector<double> inputs) {
        int weightsSize = inputSize * hiddenSize + hiddenSize;
        int biasesSize = inputSize + hiddenSize;

        std::vector<double> inputLayerWeights(parameters.begin(), parameters.begin() + inputSize * hiddenSize);
        std::vector<double> hiddenLayerWeights(parameters.begin() + inputSize * hiddenSize, parameters.begin() + weightsSize);
        std::vector<double> inputLayerBiases(parameters.begin() + weightsSize, parameters.begin() + weightsSize + inputSize);
        std::vector<double> hiddenLayerBiases(parameters.begin() + weightsSize + inputSize, parameters.begin() + weightsSize + biasesSize);
        double outputNeuronWeight = parameters[weightsSize + biasesSize];
        double outputNeuronBias = parameters[weightsSize + biasesSize + 1];

        std::vector<double> hiddenOutputs;
        for(int i = 0; i < inputLayerBiases.size(); i++) {
            double output = 0.0;
            for(int j = 0; j < inputs.size(); j++) {
                output += inputLayerWeights[i * inputs.size() + j] * inputs[j];
            }
            output += inputLayerBiases[i];
            hiddenOutputs.push_back(tanh(output));
        }

        std::vector<double> outputLayerInputs;
        for(int i = 0; i < hiddenLayerBiases.size(); i++) {
            double output = 0.0;
            for(int j = 0; j < hiddenOutputs.size(); j++) {
                output += hiddenLayerWeights[i * hiddenOutputs.size() + j] * hiddenOutputs[j];
            }
            output += hiddenLayerBiases[i];
            outputLayerInputs.push_back(tanh(output));
        }

        double finalOutput = 0.0;
        for(int i = 0; i < outputLayerInputs.size(); i++) {
            finalOutput += outputNeuronWeight * outputLayerInputs[i];
        }
        finalOutput += outputNeuronBias;

        return tanh(finalOutput);
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

int main() {
    srand((unsigned) time(NULL)); // Initialize random seed

    int inputNodes = 4;
    int hiddenNodes = 20;

    std::vector<double> randNumbers;
    //int totalNumbers = 4 * 4 + 4 + 1 + 4 + 4 * 4 + 1; // Total number of weights and biases for the neural network
    int totalNumbers = inputNodes * hiddenNodes + hiddenNodes + 1 + inputNodes + hiddenNodes * inputNodes + 1; // Total number of weights and biases for the neural network
    //double targetOutput = 0.714;
    double targetOutput = 0.612;
    double learningRate = 0.0005;

    for(int i = 0; i < totalNumbers; i++) {
        double randNumber = static_cast<double>(rand()) / RAND_MAX / 100;
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

    double output = neuralNetwork.feedForward(inputs);
    std::cout << "Output: " << output << std::endl;

    neuralNetwork.printParameters();
        std::cout << "Sweet: " << output << std::endl;


    neuralNetwork.printParameters2();

    return 0;
}

