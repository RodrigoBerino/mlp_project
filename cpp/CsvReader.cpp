#include "CsvReader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<double> CsvReader::parse_line(const std::string& line, std::size_t line_number) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        if (cell.empty()) {
            throw std::runtime_error("CSV invalido: celula vazia na linha " + std::to_string(line_number) + ".");
        }

        try {
            std::size_t idx = 0;
            double value = std::stod(cell, &idx);
            if (idx != cell.size()) {
                throw std::runtime_error("CSV invalido: valor nao numerico na linha " + std::to_string(line_number) + ".");
            }
            row.push_back(value);
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("CSV invalido: valor nao numerico na linha " + std::to_string(line_number) + ".");
        } catch (const std::out_of_range&) {
            throw std::runtime_error("CSV invalido: valor fora do intervalo na linha " + std::to_string(line_number) + ".");
        }
    }

    if (row.empty()) {
        throw std::runtime_error("CSV invalido: linha vazia encontrada na linha " + std::to_string(line_number) + ".");
    }

    return row;
}

std::vector<std::vector<double>> CsvReader::read(const std::string& file_path) const {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Nao foi possivel abrir o arquivo CSV: " + file_path);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    std::size_t line_number = 0;
    std::size_t expected_columns = 0;

    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        auto row = parse_line(line, line_number);
        if (expected_columns == 0) {
            expected_columns = row.size();
        } else if (row.size() != expected_columns) {
            throw std::runtime_error("CSV invalido: numero de colunas inconsistente na linha " + std::to_string(line_number) + ".");
        }
        data.push_back(std::move(row));
    }

    if (data.empty()) {
        throw std::runtime_error("CSV invalido: o arquivo nao contem dados.");
    }

    return data;
}
