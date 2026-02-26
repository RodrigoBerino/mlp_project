#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <string>
#include <vector>

class MainWindow {
public:
    struct InferenceResult {
        std::size_t rows{};
        std::size_t columns{};
        std::vector<double> outputs;
    };

    InferenceResult run(const std::string& csv_path, const std::string& activation_name) const;

private:
    template <typename HiddenActivation>
    InferenceResult run_with_activation(const std::vector<std::vector<double>>& input_data) const;
};

#endif
