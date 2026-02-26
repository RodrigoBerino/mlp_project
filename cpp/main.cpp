#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "MainWindow.h"

namespace {

std::string escape_json(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

std::string get_arg(int argc, char* argv[], const std::string& key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == key) {
            return argv[i + 1];
        }
    }
    return {};
}

void print_usage() {
    std::cerr << "Uso: mlp_inference --csv <arquivo.csv> --activation <relu|sigmoid|tanh>\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string csv_path = get_arg(argc, argv, "--csv");
        const std::string activation = get_arg(argc, argv, "--activation");

        if (csv_path.empty() || activation.empty()) {
            print_usage();
            return 1;
        }

        MainWindow window;
        const auto result = window.run(csv_path, activation);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "{\"rows\":" << result.rows
            << ",\"columns\":" << result.columns
            << ",\"outputs\":[";

        for (std::size_t i = 0; i < result.outputs.size(); ++i) {
            if (i > 0) {
                oss << ',';
            }
            oss << result.outputs[i];
        }
        oss << "]}";

        std::cout << oss.str() << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "{\"error\":\"" << escape_json(ex.what()) << "\"}" << std::endl;
        return 2;
    }
}
