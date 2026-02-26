#ifndef CSV_READER_H
#define CSV_READER_H

#include <string>
#include <vector>

class CsvReader {
public:
    std::vector<std::vector<double>> read(const std::string& file_path) const;

private:
    static std::vector<double> parse_line(const std::string& line, std::size_t line_number);
};

#endif
