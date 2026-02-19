#ifndef MLP_H
#define MLP_H

#include <vector>
#include <memory>
#include <stdexcept>
#include "layer.h"
#include "loss_functions.h"

/**
 * @brief Classe principal da Multi-Layer Perceptron.
 * @tparam T Tipo numérico (float, double).
 * @tparam HiddenActivation Functor da função de ativação das camadas ocultas.
 * @tparam OutputActivation Functor da função de ativação da camada de saída.
 * @tparam Loss Functor da função de custo.
 */
template <typename T, typename HiddenActivation, typename OutputActivation, typename Loss>
class MLP {
public:
    MLP(const std::vector<size_t>& layers_config) {
        if (layers_config.size() < 2) {
            throw std::invalid_argument("A configuração da rede deve ter pelo menos uma camada de entrada e uma de saída.");
        }

        // Cria as camadas ocultas
        for (size_t i = 0; i < layers_config.size() - 2; ++i) {
            hidden_layers.push_back(std::make_unique<Layer<T, HiddenActivation>>(layers_config[i], layers_config[i+1]));
        }
        // Cria a camada de saída
        output_layer = std::make_unique<Layer<T, OutputActivation>>(layers_config[layers_config.size() - 2], layers_config.back());
    }

    /**
     * @brief Realiza a propagação direta por toda a rede.
     */
    std::vector<T> forward(const std::vector<T>& input) {
        std::vector<T> current_output = input;
        for (auto& layer : hidden_layers) {
            current_output = layer->forward(current_output);
        }
        return output_layer->forward(current_output);
    }

    /**
     * @brief Realiza o backpropagation e atualiza os pesos.
     */
    void train(const std::vector<T>& input, const std::vector<T>& target, T learning_rate) {
        // 1. Forward pass para obter as saídas atuais
        std::vector<T> output = forward(input);

        // 2. Calcular o erro (delta) na camada de saída
        Loss loss_func;
        OutputActivation out_act;
        std::vector<T> loss_grads = loss_func.derivative(output, target);
        std::vector<T> output_deltas(output.size());

        for (size_t i = 0; i < output.size(); ++i) {
            output_deltas[i] = loss_grads[i] * out_act.derivative(output_layer->z_values[i]);
        }

        // 3. Retropropagar o erro para as camadas ocultas
        std::vector<T> next_layer_deltas = output_deltas;
        const auto* next_layer_weights = &output_layer->weights;
        HiddenActivation hidden_act;

        for (int i = static_cast<int>(hidden_layers.size()) - 1; i >= 0; --i) {
            auto& current_layer = hidden_layers[i];
            std::vector<T> current_deltas(current_layer->outputs.size());

            for (size_t j = 0; j < current_layer->outputs.size(); ++j) {
                T error = 0;
                for (size_t k = 0; k < next_layer_weights->size(); ++k) {
                    error += next_layer_deltas[k] * (*next_layer_weights)[k][j];
                }
                current_deltas[j] = error * hidden_act.derivative(current_layer->z_values[j]);
            }
            
            // Atualiza os pesos da camada atual
            for (size_t j = 0; j < current_layer->weights.size(); ++j) {
                for (size_t k = 0; k < current_layer->weights[j].size(); ++k) {
                    current_layer->weights[j][k] -= learning_rate * current_deltas[j] * current_layer->inputs[k];
                }
                current_layer->biases[j] -= learning_rate * current_deltas[j];
            }

            next_layer_deltas = current_deltas;
            if (i > 0) {
                next_layer_weights = &hidden_layers[i-1]->weights;
            }
        }

        // 4. Atualizar os pesos da camada de saída
        for (size_t i = 0; i < output_layer->weights.size(); ++i) {
            for (size_t j = 0; j < output_layer->weights[i].size(); ++j) {
                output_layer->weights[i][j] -= learning_rate * output_deltas[i] * output_layer->inputs[j];
            }
            output_layer->biases[i] -= learning_rate * output_deltas[i];
        }
    }

private:
    std::vector<std::unique_ptr<Layer<T, HiddenActivation>>> hidden_layers;
    std::unique_ptr<Layer<T, OutputActivation>> output_layer;
};

#endif // MLP_H
