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

class NeuralNetworkOneLayer {
public:
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;

    NeuralNetworkOneLayer(std::vector<double> randNumbers, int inputSize, int hiddenSize);
    double feedForward(std::vector<double> inputs);

    //std::vector<double> calculateNumericalGradientParameters(std::vector<double>& inputs);
    double calculateNumericalDeriviateWrtInput(std::vector<double>& inputs, int inputIndexForDerivative);
    //double getTheTotalLaplacian(std::vector<double> &inputs);
    //double getTheLaplacianVectorWrtInputs(std::vector<double> &inputs);

    std::vector<double> getTheGradientVectorWrtInputs(std::vector<double> &inputs);
    std::vector<double> getTheGradientVectorWrtParameters(std::vector<double> &inputs);

    //double calculateNumericalLaplacianWrtInput(std::vector<double>& inputs);
    //double getTheLaplacianVectorWrtInputs(std::vector<double> &inputs);
    //double getTheLaplacianVectorWrtInputs2(std::vector<double> &inputs);
    //double getTheLaplacianFromGradient(std::vector<double> &inputs);
    double laplacianOfLogarithmWrtInputs(std::vector<double> &inputs);

    static std::vector<double> generateRandomParameterSet(size_t rbs_M, size_t rbs_N, int randomSeed, double spread);

    //void backpropagate(std::vector<double> inputs, double targetOutput, double learningRate);

private:
    //Storing parameters in multiple different forms, to allow fast feed forward calculation and automatic differentiation.
    ArrayXvar parametersVar;

    int weightsSize;
    ArrayXvar inputLayerWeightsVar;
    ArrayXvar hiddenLayerWeightsVar;
    ArrayXvar hiddenLayerBiasesVar;
    //Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> inputLayerWeightsMatrix;

    std::vector<double> inputLayerWeightsDouble;
    std::vector<double> hiddenLayerWeightsDouble;
    std::vector<double> hiddenLayerBiasesDouble;

};
