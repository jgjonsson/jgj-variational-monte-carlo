#include <cassert>
#include <cmath>
#include <vector>
#include <random>

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cassert>

#include "../../include/system.h"
#include "../../include/hamiltonian_cyllindric_repulsive.h"
#include "../../include/initialstate.h"
#include "../../include/random.h"
#include "../../include/particle.h"
#include "../../include/sampler.h"

//#include "../../include/neural.h"
#include "../../include/nn_wave.h"
#include "../../include/neural_reverse.h"

// Define the 2D Gaussian function
double gaussian2D(double x, double y) {
    double sigma = 1.0;
    return std::exp(-(x*x + y*y) / (2 * sigma * sigma));
}

int main() {
    // Initialize the neural network with 2 input nodes and 4 hidden nodes
    int inputSize = 2;
    int hiddenSize = 12;
    /*
    std::vector<double> randNumbers(inputSize * hiddenSize + hiddenSize);
    std::generate(randNumbers.begin(), randNumbers.end(), std::rand);
    NeuralNetworkReverse nnr(randNumbers, inputSize, hiddenSize);
*/
	//By having same seed every time we can get known values to directly check for correctness in tests.
    int seed = 2023;
    double parameterGuessSpread = 0.1;  //Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.

    //Start with all parameters as random values
    auto randomParameters = NeuralNetworkWavefunction::generateRandomParameterSet(inputSize, hiddenSize, seed, parameterGuessSpread);
	//auto looseNeuralNetwork = std::make_unique<SimpleRBM>(rbs_M, rbs_N, randomParameters, omega);

	//auto nnr = std::make_unique<NeuralNetworkReverse>(randomParameters, inputSize, hiddenSize);
	NeuralNetworkReverse nnr(randomParameters, inputSize, hiddenSize);

// Define the training data
std::vector<std::vector<double>> inputs(5, std::vector<double>(inputSize));
std::vector<double> targets(inputs.size());

// Initialize random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);


	// Run the training loop
    double learningRate = 0.1;
    for (int epoch = 0; epoch < 100000; epoch++) {

        // Generate new random inputs for each epoch
        for (size_t i = 0; i < inputs.size(); i++) {
            for (size_t j = 0; j < inputSize; j++) {
                inputs[i][j] = dis(gen);
            }
        }
        /*
        // Generate new random inputs for each epoch
        for (auto& input : inputs) {
            for (auto& value : input) {
                value = dis(gen);
            }
        }*/

        // Calculate targets for new inputs
        for (size_t i = 0; i < inputs.size(); i++) {
            targets[i] = gaussian2D(inputs[i][0], inputs[i][1]);
        }
        //cout << "Blah" << inputs.size() << endl;
        //cout << "Target is " << targets[0] <<endl;
//cout << "Training on " << inputs[0][0] <<endl;
        // Train on the new inputs and targets
        for (size_t i = 0; i < inputs.size(); i++) {
        //cout << "Calling backpropagate with " << inputs[i][0] << " and " << inputs[i][1] << " and " << targets[i] << endl;
            nnr.backpropagate(inputs[i], targets[i], learningRate);
        }
    }

    // Check the outputs of the neural network after training on new random inputs
    for (int i = 0; i < 100; i++) {
        std::vector<double> newInput(inputSize);
        for (auto& value : newInput) {
            value = dis(gen);
        }
        double newTarget = gaussian2D(newInput[0], newInput[1]);
        double newOutput = nnr.feedForward(newInput);
        assert(std::fabs(newTarget - newOutput) < 0.1); // The output of the neural network should match the target
    }

/*
    // Define the training data
    std::vector<std::vector<double>> inputs = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    std::vector<double> targets(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        targets[i] = gaussian2D(inputs[i][0], inputs[i][1]);
    }

    // Run the training loop
    double learningRate = 0.1;
    for (int epoch = 0; epoch < 1000; epoch++) {
        for (size_t i = 0; i < inputs.size(); i++) {
            nnr.backpropagate(inputs[i], targets[i], learningRate);
        }
    }

    // Check the outputs of the neural network after training
    for (size_t i = 0; i < inputs.size(); i++) {
        double output = nnr.feedForward(inputs[i]);
        assert(std::fabs(targets[i] - output) < 0.1); // The output of the neural network should match the target
    }
    */

    return 0;
}