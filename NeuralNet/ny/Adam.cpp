#include <iostream>
#include <cmath>
#include <Eigen/Dense>

// Simple quadratic function f(x) = x^2
double quadratic_function(double x) {
    return x * x;
}

// Derivative of the quadratic function: f'(x) = 2 * x
double quadratic_derivative(double x) {
    return 2 * x;
}

// Plain Gradient Descent
double plain_gradient_descent(double initial_x, double learning_rate, int num_iterations) {
    double x = initial_x;

    for (int i = 0; i < num_iterations; ++i) {
        double gradient = quadratic_derivative(x);
        x -= learning_rate * gradient;
    }

    return x;
}

// Adam Optimizer
double adam_optimizer(double initial_x, double learning_rate, int num_iterations) {
    double x = initial_x;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    Eigen::VectorXd m(1);
    Eigen::VectorXd v(1);
    m.setZero();
    v.setZero();
    int t = 0;

    for (int i = 0; i < num_iterations; ++i) {
        double gradient = quadratic_derivative(x);
        t++;
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * gradient * gradient;
        double m_hat = m / (1 - std::pow(beta1, t));
        double v_hat = v / (1 - std::pow(beta2, t));
        x -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    return x;
}

int main() {
    double initial_x = 10.0; // Initial guess
    double learning_rate = 0.1;
    int num_iterations = 100;

    std::cout << "Plain Gradient Descent:" << std::endl;
    double result_plain = plain_gradient_descent(initial_x, learning_rate, num_iterations);
    std::cout << "Minimum found: " << result_plain << ", Value: " << quadratic_function(result_plain) << std::endl;

    std::cout << "\nAdam Optimizer:" << std::endl;
    double result_adam = adam_optimizer(initial_x, learning_rate, num_iterations);
    std::cout << "Minimum found: " << result_adam << ", Value: " << quadratic_function(result_adam) << std::endl;

    return 0;
}
