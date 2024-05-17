#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <Eigen/Dense>

using namespace autodiff;
using namespace Eigen;

class NeuralNetworkReverse {
public:
    ArrayXvar parametersDual;
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;

    NeuralNetworkReverse(std::vector<double> randNumbers, int inputSize, int hiddenSize);
    double feedForward(std::vector<double> inputs);

    std::vector<double> calculateNumericalGradientParameters(std::vector<double>& inputs);
    double calculateNumericalDeriviateWrtInput(std::vector<double>& inputs, int inputIndexForDerivative);
    //double getTheTotalLaplacian(std::vector<double> &inputs);
    //double getTheLaplacianVectorWrtInputs(std::vector<double> &inputs);

    std::vector<double> getTheGradientVectorWrtInputs(std::vector<double> &inputs);
    std::vector<double> getTheGradientVectorWrtParameters(std::vector<double> &inputs);

    double calculateNumericalLaplacianWrtInput(std::vector<double>& inputs);
    double getTheLaplacianVectorWrtInputs(std::vector<double> &inputs);
    double getTheLaplacianVectorWrtInputs2(std::vector<double> &inputs);
    double getTheLaplacianFromGradient(std::vector<double> &inputs);
    double laplacianOfLogarithmWrtInputs(std::vector<double> &inputs);

    void backpropagate(std::vector<double> inputs, double targetOutput, double learningRate);

private:
    int weightsSize;
    ArrayXvar inputLayerWeights;
    ArrayXvar hiddenLayerWeights;
    ArrayXvar hiddenLayerBiases;

};
