#include "data/load_data.hpp"
#include <glog/logging.h>
#include <armadillo>
#include <fstream>
#include <string>
#include <utility>

namespace kuiper_infer {

std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream& file, char split_char) {
    bool load_ok = file.good();
    file.clear();
    size_t fn_rows = 0;
    size_t fn_cols = 0;
    const std::ifstream::pos_type start_pos = file.tellg();

    std::string token;
    std::string line_str;
    std::stringstream line_stream;

    while (file.good() && load_ok) {
        std::getline(file, line_str);
        if (line_str.empty()) {
            break;
        }

        line_stream.clear();
        line_stream.str(line_str);
        size_t line_cols = 0;

        std::string row_token;
        while (line_stream.good()) {
            std::getline(line_stream, row_token, split_char);
            ++line_cols;
        }
        if (line_cols > fn_cols) {
            fn_cols = line_cols;
        }

        ++fn_rows;
    }
    file.clear();
    file.seekg(start_pos);
    return { fn_rows, fn_cols };
}
}  // namespace kuiper_infer