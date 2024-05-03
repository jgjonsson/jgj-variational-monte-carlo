#include "../include/adam.h"

AdamOptimizer::AdamOptimizer(int paramSize, double learningRate) : m(paramSize, 0), v(paramSize, 0), beta1(0.9), beta2(0.999), epsilon(1e-8), learning_rate(learningRate) {}

std::vector<double> AdamOptimizer::adamOptimization(std::vector<double> params, std::vector<double>& gradient, int count) {
    double total_change = 0.0;
    // Update the parameter using Adam optimizer
    for (size_t param_num = 0; param_num < params.size(); ++param_num)
    {
        // Update biased first and second moment estimates
        m[param_num] = beta1 * m[param_num] + (1 - beta1) * gradient[param_num];
        v[param_num] = beta2 * v[param_num] + (1 - beta2) * gradient[param_num] * gradient[param_num];
        // Compute bias-corrected first and second moment estimates
        double m_hat = m[param_num] / (1 - pow(beta1, count + 1));
        double v_hat = v[param_num] / (1 - pow(beta2, count + 1));
        // Update parameters
        params[param_num] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        total_change += fabs(learning_rate * m_hat / (sqrt(v_hat) + epsilon));
    }
    return params;
}
/*
AdamOptimizer::AdamOptimizer(int paramSize, double learningRate) : m(paramSize, 0), v(paramSize, 0), beta1(0.9), beta2(0.999), epsilon(1e-8), learning_rate(learningRate) {}

    std::vector<double> adamOptimization(std::vector<double> params, std::vector<double>& gradient, int count) {
        double total_change = 0.0;
        // Update the parameter using Adam optimizer
        for (size_t param_num = 0; param_num < params.size(); ++param_num)
        {
            // Update biased first and second moment estimates
            m[param_num] = beta1 * m[param_num] + (1 - beta1) * gradient[param_num];
            v[param_num] = beta2 * v[param_num] + (1 - beta2) * gradient[param_num] * gradient[param_num];
            // Compute bias-corrected first and second moment estimates
            double m_hat = m[param_num] / (1 - pow(beta1, count + 1));
            double v_hat = v[param_num] / (1 - pow(beta2, count + 1));
            // Update parameters
            params[param_num] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            total_change += fabs(learning_rate * m_hat / (sqrt(v_hat) + epsilon));
        }
        return params;
    }
;
*/