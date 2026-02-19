#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <random>
#include <cmath>
#include "activation_functions.h"

/**
 * @brief Classe que representa uma camada da MLP.
 * @tparam T Tipo numérico (float, double).
 * @tparam Activation Functor da função de ativação.
 */
template <typename T, typename Activation>
class Layer {
public:
    Layer(size_t input_size, size_t output_size)
        : weights(output_size, std::vector<T>(input_size)),
          biases(output_size),
          outputs(output_size),
          inputs(input_size),
          z_values(output_size),
          weight_grads(output_size, std::vector<T>(input_size)),
          bias_grads(output_size) {
        initialize_weights();
    }

    /**
     * @brief Realiza a propagação direta (forward pass) na camada.
     */
    const std::vector<T>& forward(const std::vector<T>& input_data) {
        inputs = input_data;
        Activation activation;

        for (size_t i = 0; i < weights.size(); ++i) {
            T sum = 0;
            for (size_t j = 0; j < weights[i].size(); ++j) {
                sum += weights[i][j] * inputs[j];
            }
            z_values[i] = sum + biases[i];
            outputs[i] = activation(z_values[i]);
        }
        return outputs;
    }

    /**
     * @brief Inicializa pesos e bias aleatoriamente.
     */
    void initialize_weights() {
        std::random_device rd;
        std::mt19937 generator(rd());
        // Inicialização de Xavier/Glorot para melhor convergência
        T limit = std::sqrt(static_cast<T>(6) / (weights[0].size() + weights.size()));
        std::uniform_real_distribution<T> distribution(-limit, limit);

        for (auto& row : weights) {
            for (auto& w : row) {
                w = distribution(generator);
            }
        }
        for (auto& b : biases) {
            b = static_cast<T>(0);
        }
    }

    std::vector<std::vector<T>> weights;
    std::vector<T> biases;
    std::vector<T> outputs;
    std::vector<T> inputs;
    std::vector<T> z_values;
    std::vector<std::vector<T>> weight_grads;
    std::vector<T> bias_grads;
};

#endif // LAYER_H
