#include <iostream>
#include <vector>
#include <iomanip>
#include "mlp.h"
#include "activation_functions.h"
#include "loss_functions.h"

int main() {
    // configuração da rede: 2 entradas, 4 neurônios na camada oculta, 1 saída
    std::vector<size_t> config = {2, 4, 1};

    // instanciação da MLP usando templates e functors
    // Tipo: double, Ativação Oculta: Tanh, Ativação Saída: Sigmoid, Custo: MSE
    MLP<double, Tanh<double>, Sigmoid<double>, MSE<double>> my_mlp(config);

    // Conjunto de dados XOR
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    std::cout << "--- Treinando MLP para o problema XOR ---" << std::endl;
    
    int epochs = 10000;
    double learning_rate = 0.1;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        double total_loss = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            my_mlp.train(inputs[i], targets[i], learning_rate);
            
            auto output = my_mlp.forward(inputs[i]);
            MSE<double> mse;
            total_loss += mse(output, targets[i]);
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoca " << std::setw(5) << epoch 
                      << " | Erro Medio: " << std::fixed << std::setprecision(6) << total_loss / 4.0 << std::endl;
        }
    }

    std::cout << "\n--- Resultados Finais ---" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = my_mlp.forward(inputs[i]);
        std::cout << "Entrada: [" << inputs[i][0] << ", " << inputs[i][1] << "] "
                  << "Alvo: [" << targets[i][0] << "] "
                  << "Predicao: [" << std::fixed << std::setprecision(4) << output[0] << "]" << std::endl;
    }

    return 0;
}
