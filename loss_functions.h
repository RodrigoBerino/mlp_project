#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <vector>
#include <cmath>

/**
 * @brief Functor para a função de custo MSE (Mean Squared Error).
 */
template <typename T>
struct MSE {
    T operator()(const std::vector<T>& predicted, const std::vector<T>& target) const {
        T sum = 0;
        for (size_t i = 0; i < predicted.size(); ++i) {
            T diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / static_cast<T>(predicted.size());
    }

    std::vector<T> derivative(const std::vector<T>& predicted, const std::vector<T>& target) const {
        std::vector<T> grads(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i) {
            grads[i] = static_cast<T>(2) * (predicted[i] - target[i]) / static_cast<T>(predicted.size());
        }
        return grads;
    }
};

#endif // LOSS_FUNCTIONS_H
