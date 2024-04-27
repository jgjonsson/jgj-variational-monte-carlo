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

class NeuralNetwork {
public:
    VectorXdual parametersDual;
    std::vector<double> parameters;
    int inputSize;
    int hiddenSize;
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
    NeuralNetwork(std::vector<double> randNumbers, int inputSize, int hiddenSize);
    dual feedForwardDual2(VectorXdual inputsDual);
    double feedForward(std::vector<double> inputs);
    std::function<VectorXdual(VectorXdual, VectorXdual)> getGradientFunction();
    VectorXdual getGradient(VectorXdual inputsDual);
    void backpropagate(std::vector<double> inputs, double targetOutput, double learningRate);
    void printParameters();
    void printParameters2();
};

#endif // NEURALNETWORK_H
