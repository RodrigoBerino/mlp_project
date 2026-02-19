#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>
#include <algorithm>

/**
 * @brief Functor para a função de ativação Sigmoid.
 */
template <typename T>
struct Sigmoid {
    T operator()(T x) const {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    }

    T derivative(T x) const {
        T s = (*this)(x);
        return s * (static_cast<T>(1) - s);
    }
};

/**
 * @brief Functor para a função de ativação ReLU (Rectified Linear Unit).
 */
template <typename T>
struct ReLU {
    T operator()(T x) const {
        return std::max(static_cast<T>(0), x);
    }

    T derivative(T x) const {
        return x > 0 ? static_cast<T>(1) : static_cast<T>(0);
    }
};

/**
 * @brief Functor para a função de ativação Tanh (Tangente Hiperbólica).
 */
template <typename T>
struct Tanh {
    T operator()(T x) const {
        return std::tanh(x);
    }

    T derivative(T x) const {
        T t = std::tanh(x);
        return static_cast<T>(1) - t * t;
    }
};

#endif // ACTIVATION_FUNCTIONS_H
