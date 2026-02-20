#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>
#include <algorithm>

// objetos que podem ser chamados como se fossem funções.
// tecnicamente, é qualquer classe que sobrescreve o operador.
// programação genérica e templates

/*
podem armazenar estado interno, funcionam muito bem com templates, 
podem ser otimizados melhor pelo compilador, 
são mais rápidos que std::function na maioria dos casos,
são a base das lambdas 


sigmoid e backpropagation para o resultadp*/


/**
 * @brief Functor como tempalte para a função de ativação Sigmoid.
 */
template <typename T>

struct Sigmoid {
    //calcula a ativação
    T operator()(T x) const
    {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    }
    //calcula a derivada (para backpropagation)
    T derivative(T x) const
    {
        T s = (*this)(x);
        return s * (static_cast<T>(1) - s);
    }
};

/**
 * @brief Functor para a função de ativação ReLU (Rectified Linear Unit).
 */
template <typename T>

struct ReLU {
    T operator()(T x) const
    {
        return std::max(static_cast<T>(0), x);
    }

    T derivative(T x) const
    {
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
