#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdlib>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <Eigen/Dense>

using namespace autodiff;
using namespace Eigen;

class NeuralNetworkSimple {
public:
    VectorXdual parametersDual;
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;
    std::function<VectorXdual(VectorXdual, VectorXdual)> gradientFunction;
/*
    NeuralNetwork(std::vector<double> randNumbers, int inputSize, int hiddenSize);
    dual feedForwardDual2(VectorXdual inputsDual);
    double feedForward(std::vector<double> inputs);
    auto getGradientFunction();
    auto getGradient(VectorXdual inputsDual);
    void backpropagate(std::vector<double> inputs, double targetOutput, double learningRate);
    void printParameters();
    void printParameters2();
    */
    NeuralNetworkSimple(std::vector<double> randNumbers, int inputSize, int hiddenSize);
    dual feedForwardDual2(VectorXdual inputsDual);
    double feedForward(std::vector<double> inputs);

    VectorXdual getTheGradient(VectorXdual inputsDual);
    std::vector<double> getTheGradientVector(std::vector<double> inputs);
    void backpropagate(std::vector<double> inputs, double targetOutput, double learningRate);
    void printParameters();
    void printParameters2();
private:
    std::function<VectorXdual(VectorXdual, VectorXdual)> getGradientFunction();
};

#endif // NEURALNETWORK_H
