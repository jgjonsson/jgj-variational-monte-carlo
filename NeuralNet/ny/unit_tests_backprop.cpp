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
    int hiddenSize = 4;
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

    return 0;
}