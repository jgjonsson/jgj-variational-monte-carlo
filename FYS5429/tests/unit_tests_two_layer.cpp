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

#include "../../include/neural_twolayer.h"
//#include "../../include/nn_wave.h"

using namespace std;

const double closeEnoughTolerance = 0.001;
bool closeEnough(double x, double y)
{
    return fabs(x-y) < closeEnoughTolerance;
}

std::vector<double> calculateNumericalGradientParameters(std::unique_ptr<NeuralNetworkTwoLayers>& looseNeuralNetwork, std::vector<double>& inputs) {
    double epsilon = 1e-6; // small number for finite difference
    std::vector<double> gradient(looseNeuralNetwork->parameters.size());

    for (size_t i = 0; i < looseNeuralNetwork->parameters.size(); ++i) {
        // Store the original value so we can reset it later
        double originalValue = looseNeuralNetwork->parameters[i];

        // Evaluate function at p+h
        looseNeuralNetwork->parameters[i] += epsilon;
        looseNeuralNetwork->cacheWeightsAndBiases(looseNeuralNetwork->parameters);
        double plusEpsilon = looseNeuralNetwork->feedForward(inputs);

        // Evaluate function at p-h
        looseNeuralNetwork->parameters[i] = originalValue - epsilon;
        looseNeuralNetwork->cacheWeightsAndBiases(looseNeuralNetwork->parameters);
        double minusEpsilon = looseNeuralNetwork->feedForward(inputs);

        // Compute the gradient
        gradient[i] = (plusEpsilon - minusEpsilon) / (2.0 * epsilon);

        // Reset the parameter to its original value
        looseNeuralNetwork->parameters[i] = originalValue;
        looseNeuralNetwork->cacheWeightsAndBiases(looseNeuralNetwork->parameters);
    }

    return gradient;
}


std::vector<double> calculateNumericalGradientInputs(std::unique_ptr<NeuralNetworkTwoLayers>& looseNeuralNetwork, std::vector<double>& inputs) {
    double epsilon = 1e-6; // small number for finite difference
    std::vector<double> gradient(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        // Store the original value so we can reset it later
        double originalValue = inputs[i];

        // Evaluate function at x+h
        inputs[i] += epsilon;
        double plusEpsilon = looseNeuralNetwork->feedForward(inputs);

        // Evaluate function at x-h
        inputs[i] = originalValue - epsilon;
        double minusEpsilon = looseNeuralNetwork->feedForward(inputs);

        // Compute the gradient
        gradient[i] = (plusEpsilon - minusEpsilon) / (2.0 * epsilon);

        // Reset the input to its original value
        inputs[i] = originalValue;
    }

    return gradient;
}

/*
std::vector<double> calculateNumericalGradient(std::unique_ptr<NeuralNetworkTwoLayers>& looseNeuralNetwork, std::vector<double>& inputs) {
    double epsilon = 1e-6; // small number for finite difference
    std::vector<double> gradient(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        // Store the original value so we can reset it later
        double originalValue = inputs[i];

        // Evaluate function at x+h
        inputs[i] += epsilon;
        double plusEpsilon = looseNeuralNetwork->feedForward(inputs);

        // Evaluate function at x-h
        inputs[i] = originalValue - epsilon;
        double minusEpsilon = looseNeuralNetwork->feedForward(inputs);

        // Compute the gradient
        gradient[i] = (plusEpsilon - minusEpsilon) / (2.0 * epsilon);

        // Reset the input to its original value
        inputs[i] = originalValue;
    }

    return gradient;
}*/

int main(int argc, char **argv)
{
    // Seed for the random number generator. 
	//By having same seed every time we can get known values to directly check for correctness in tests. 
    int seed = 2023;
    double parameterGuessSpread = 0.1;  //Standard deviation "spread" of the normal distribution that initial parameter guess is randomized as.

    size_t numberOfDimensions = 2;
    size_t numberOfParticles = 2;
	
    double omega = 1.0;                                         // Oscillator frequency.
    double beta = 2.82843;                                      // Frequency ratio
    double hard_core_size = 0.0043 / sqrt(omega);               // Hard core size
    std::vector<double> params{argc > 4 ? stod(argv[4]) : 0.5}; // Variational parameter.
    double stepLength = 0.1;                                    // Metropolis step length.
    size_t MC_reduction = 100;                                  // Number of MC steps to reduce by at intermediate steps
    bool verbose = true;                                        // Verbosity of output


	size_t rbs_M = numberOfParticles*numberOfDimensions;

	size_t rbs_N = 2;//rbs_M;//1; //Only one hidden node is on the extreme small side in practical scenarios. rbs_N = rbs_M would have been more realistic.
	//However in a unit test setting it gives a nice small set of values for unit testing.
	//Also since M != N, we get non square matrix, and might uncover bugs related to matrix dimensionalities in matrix multiplications and such.

	cout << " ------------------------------ " << endl;
	
	auto rng = std::make_unique<Random>(seed);
	// Initialize particles
	auto particles = setupRandomUniformInitialStateWithRepulsion(stepLength, hard_core_size, numberOfDimensions, numberOfParticles, *rng);

	cout << " ------------------------------ " << endl;
	
	auto hamiltonian = std::make_unique<RepulsiveHamiltonianCyllindric>(omega, beta);


    //Start with all parameters as random values
    auto randomParameters = NeuralNetworkTwoLayers::generateRandomParameterSetTwoLayers(rbs_M, rbs_N, seed, parameterGuessSpread);
	//auto looseNeuralNetwork = std::make_unique<SimpleRBM>(rbs_M, rbs_N, randomParameters, omega);

    //Print length of randomParameters
    cout << "Length of randomParameters: " << randomParameters.size() << endl;

	auto looseNeuralNetwork = std::make_unique<NeuralNetworkTwoLayers>(randomParameters, rbs_M, rbs_N);
	
	// Construct an input vector of doubles
    std::vector<double> inputs = {0.1, 0.2, 0.3, 0.4};
    
    // Convert the vector of doubles to VectorXdual
    //VectorXdual inputsDual = Eigen::Map<VectorXd>(inputs.data(), inputs.size()).cast<dual>();
cout << "junit 1" << endl;
    // Call the feedForwardDual2 function
    //dual outputDual = looseNeuralNetwork->feedForwardDual2(inputsDual);
    cout << "junit 2" << endl;
    // Call the feedForward function
    double output = looseNeuralNetwork->feedForward(inputs);
    cout << "junit 3" << endl;
    //cout << "outputDual: " << outputDual << endl;
    cout << "output: " << output << endl;
    // Check if the values are the same
    /*
    if (abs(output - static_cast<double>(outputDual)) < 1e-6) {
        std::cout << "The values are the same." << std::endl;
    } else {
        std::cout << "The values are not the same." << std::endl;
    }
*/
cout << "0" << endl;
auto gradientSymbolicCachedFunctionInputs = looseNeuralNetwork->getTheGradientVectorWrtInputs(inputs);
cout << "1" << endl;
auto gradientSymbolicCachedFunctionParameters = looseNeuralNetwork->getTheGradientVectorWrtParameters(inputs);

    cout << "Params Gradient calculated with automatic diff:                ";
    for(const auto& value : gradientSymbolicCachedFunctionParameters) {
        cout << value << " ";
    }
    cout << endl;

cout << "A" << endl;
    std::vector<double> gradientNumeric = calculateNumericalGradientParameters(looseNeuralNetwork, inputs);
    //std::vector<double> gradientNumeric = looseNeuralNetwork->calculateNumericalGradientParameters(inputs);
    cout << "B" << endl;
    std::vector<double> gradientNumericInputs = calculateNumericalGradientInputs(looseNeuralNetwork, inputs);
cout << "C" << endl;
    cout << "Params Gradient calculated with numerical methods:             ";
    for(const auto& value : gradientNumeric) {
        cout << value << " ";
    }
    cout << endl;

    cout << "Inputs Gradient calculated with automatic diff:                ";
    for(const auto& value : gradientSymbolicCachedFunctionInputs) {
        cout << value << " ";
    }
    cout << endl;

    cout << "Inputs Gradient calculated with numerical methods:             ";
    for(const auto& value : gradientNumericInputs) {
        cout << value << " ";
    }
    cout << endl;

    //Calculate Laplacian of log of wave function
    //double lap = looseNeuralNetwork->laplacianOfLogarithmWrtInputs(inputs);
    double lapTotal = looseNeuralNetwork->laplacianOfLogarithmWrtInputs(inputs);
    //Do the same numerically
    //double lapNumeric = looseNeuralNetwork->calculateNumericalLaplacianWrtInput(inputs);

    //Calculate dot product of gradientNumeric with itself
    //double gradSqr = gradientNumeric * gradientNumeric;
    double gradSqr = std::inner_product(gradientNumericInputs.begin(), gradientNumericInputs.end(), gradientNumericInputs.begin(), 0.0);

    //Print those 2 for comparision
    //cout << "Laplacian calculated with automatic diff: " << lap << endl;
    //cout << "Laplacian calculated with numerical methods: " << lapNumeric << endl;

    cout << "Total laplacian of log of a wave function with automatic diff:" << lapTotal << endl;
    //cout << "Total laplacian of log of a wave function with numerical methods:" << lapNumeric+gradSqr << endl;

/*
	double lap = looseNeuralNetwork->computeLocalLaplasian(particles);
	cout << " ------------------------------ " << endl;

	
	for (size_t i = 0; i < particles.size(); i++)
    {
        auto position = particles[i]->getPosition();
		cout << "Position particle " << i << ": " ;
		auto numDimensions = position.size();
        for (size_t j=0; j<numDimensions; j++)
        {
            cout << position[j] << " ";
        }
		cout << endl;
	}
	//cout << "First particle coordinates " << (particles[0]->getPosition()) << endl;
	double value = looseNeuralNetwork->evaluate(particles);
	cout << " ------------------------------ " << endl;

	cout << "Wave function value " << value << endl;
	cout << "Wave function Laplacian " << lap << endl;

	cout << " ------------------------------ " << endl;

	auto qForce = looseNeuralNetwork->computeQuantumForce(particles, 0);
	auto qForce2 = looseNeuralNetwork->computeQuantumForce(particles, 1);
    cout << "Quantum-force particle 1 (size M*D=" << qForce.size() << "): ";
    for(int i=0; i<qForce.size(); i++){
      cout << qForce[i] << " " ;
    }
    cout << endl;
    cout << "Quantum-force particle 2 (size M*D=" << qForce2.size() << "): ";
    for(int i=0; i<qForce2.size(); i++){
      cout << qForce2[i] << " " ;
    }
    cout << endl;
	cout << " ------------------------------ " << endl;

	auto logPsiDerivatives = looseNeuralNetwork->computeLogPsiDerivativeOverParameters(particles);
    cout << "All derivatives w.r.t parameters, of ln psi (size M+N+M*N=" << logPsiDerivatives.size() << "): ";
    for(int i=0; i<logPsiDerivatives.size(); i++){
      cout << logPsiDerivatives[i] << " " ;
    }
    cout << endl;
	cout << " ------------------------------ " << endl;

    //Now run assert on some selected values. If failed, the program will print out which row in this source code file it happened.
    assert(closeEnough(value, 1.86413));
    assert(closeEnough(lap, -3.84772));
    assert(closeEnough(qForce[0], 0.100514));
    assert(closeEnough(qForce[1], -0.376823));
    assert(closeEnough(qForce2[0], -0.419817));
    assert(closeEnough(qForce2[1], -0.456047));
    assert(closeEnough(logPsiDerivatives[0], -0.106325));  //One derivative related to a
    assert(closeEnough(logPsiDerivatives[rbs_M], 0.487879)); //One derivative related to b
    assert(closeEnough(logPsiDerivatives[logPsiDerivatives.size()-1], 0.135884)); //One derivative related to W
*/
	cout << "All tests passed!" << endl;  //If we reach this line, no assert failed.

    return 0;
}
