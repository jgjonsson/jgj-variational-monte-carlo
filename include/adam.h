#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H

#include <vector>
#include <cmath>

class AdamOptimizer {
private:
    std::vector<double> m; // first moment vector
    std::vector<double> v; // second moment vector
    double beta1; // exponential decay rate for the first moment estimates
    double beta2; // exponential decay rate for the second moment estimates
    double epsilon; // small constant for numerical stability
    double learning_rate; // learning rate

public:
    AdamOptimizer(int paramSize, double learningRate);

    std::vector<double> adamOptimization(std::vector<double> params, std::vector<double>& gradient, int count);
};

#endif // ADAMOPTIMIZER_H
