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

class NeuralNetworkTwoLayers {
public:
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;

    NeuralNetworkTwoLayers(std::vector<double> randNumbers, int inputSize, int hiddenSize);
    double feedForward(std::vector<double> inputs);

    double calculateNumericalDeriviateWrtInput(std::vector<double>& inputs, int inputIndexForDerivative);

    std::vector<double> getTheGradientVectorWrtInputs(std::vector<double> &inputs);
    std::vector<double> getTheGradientVectorWrtParameters(std::vector<double> &inputs);

    double laplacianOfLogarithmWrtInputs(std::vector<double> &inputs);

    static std::vector<double> generateRandomParameterSetTwoLayers(size_t rbs_M, size_t rbs_N, int randomSeed, double spread);

    void cacheWeightsAndBiases(std::vector<double> parameters);

private:
    //Storing parameters in multiple different forms, to allow fast feed forward calculation and automatic differentiation.
    ArrayXvar parametersVar;

    int weightsSize;
    ArrayXvar inputLayerWeightsVar;
    ArrayXvar hiddenLayerWeightsVar;
    ArrayXvar hiddenLayerBiasesVar;
    ArrayXvar secondHiddenLayerWeightsVar;
    ArrayXvar secondHiddenLayerBiasesVar;

    std::vector<double> inputLayerWeightsDouble;
    std::vector<double> hiddenLayerWeightsDouble;
    std::vector<double> hiddenLayerBiasesDouble;
    std::vector<double> secondHiddenLayerWeightsDouble;
    std::vector<double> secondHiddenLayerBiasesDouble;

};
