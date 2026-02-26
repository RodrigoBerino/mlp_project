#include "MainWindow.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "CsvReader.h"
#include "activation_functions.h"
#include "loss_functions.h"
#include "mlp.h"

MainWindow::InferenceResult MainWindow::run(const std::string& csv_path, const std::string& activation_name) const {
    CsvReader reader;
    auto data = reader.read(csv_path);

    if (activation_name == "relu") {
        return run_with_activation<ReLU<double>>(data);
    }
    if (activation_name == "sigmoid") {
        return run_with_activation<Sigmoid<double>>(data);
    }
    if (activation_name == "tanh") {
        return run_with_activation<Tanh<double>>(data);
    }

    throw std::runtime_error("Funcao de ativacao invalida. Use: relu, sigmoid ou tanh.");
}

template <typename HiddenActivation>
MainWindow::InferenceResult MainWindow::run_with_activation(const std::vector<std::vector<double>>& input_data) const {
    if (input_data.empty() || input_data.front().empty()) {
        throw std::runtime_error("Nao ha dados suficientes para inferencia.");
    }

    const std::size_t input_size = input_data.front().size();
    const std::size_t hidden_size = std::max<std::size_t>(4, input_size * 2);

    std::vector<std::size_t> config = {input_size, hidden_size, 1};
    MLP<double, HiddenActivation, Sigmoid<double>, MSE<double>> mlp(config);

    // Treinamento leve para tornar a inferencia mais estavel,
    // usando a media da linha como alvo sintetico em [0, 1].
    constexpr int epochs = 150;
    constexpr double learning_rate = 0.05;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& row : input_data) {
            double avg = 0.0;
            for (double value : row) {
                avg += value;
            }
            avg /= static_cast<double>(row.size());
            avg = std::clamp(avg, 0.0, 1.0);
            mlp.train(row, {avg}, learning_rate);
        }
    }

    InferenceResult result;
    result.rows = input_data.size();
    result.columns = input_size;
    result.outputs.reserve(input_data.size());

    for (const auto& row : input_data) {
        auto output = mlp.forward(row);
        result.outputs.push_back(output.front());
    }

    return result;
}

// Instanciacao explicita das variantes usadas.
template MainWindow::InferenceResult MainWindow::run_with_activation<ReLU<double>>(const std::vector<std::vector<double>>& input_data) const;
template MainWindow::InferenceResult MainWindow::run_with_activation<Sigmoid<double>>(const std::vector<std::vector<double>>& input_data) const;
template MainWindow::InferenceResult MainWindow::run_with_activation<Tanh<double>>(const std::vector<std::vector<double>>& input_data) const;
