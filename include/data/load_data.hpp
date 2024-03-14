#ifndef KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_

#include <glog/logging.h>
#include <armadillo>
#include <string>

namespace kuiper_infer {

/**
 * @brief CSV data loader
 *
 * Provides utility to load CSV data into Armadillo matrices.
 */
class CSVDataLoader {
public:
    /**
     * @brief Loads CSV file into matrix
     *
     * Loads data from a CSV file into an Armadillo matrix.
     *
     * @param file_path Path to CSV file
     * @param split_char Delimiter character
     * @return Matrix containing loaded data
     */
    template <typename T>
    static arma::Mat<T> LoadData(const std::string& file_path, char split_char = ',');

private:
    /**
     * @brief Gets matrix dimensions from CSV
     *
     * Gets number of rows and cols for matrix based on CSV file.

     * @param file CSV file stream
     * @param split_char Delimiter character
     * @return Pair of rows and cols
    */
    static std::pair<size_t, size_t> GetMatrixSize(std::ifstream& file, char split_char);
};

template <typename T>
arma::Mat<T> CSVDataLoader::LoadData(const std::string& file_path, const char split_char) {
    arma::Mat<T> data;
    if (file_path.empty()) {
        LOG(ERROR) << "CSV file path is empty: " << file_path;
        return data;
    }

    std::ifstream in(file_path);
    if (!in.is_open() || !in.good()) {
        LOG(ERROR) << "File open failed: " << file_path;
        return data;
    }

    std::string line_str;
    std::stringstream line_stream;

    const auto& [rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
    data.zeros(rows, cols);

    size_t row = 0;
    while (in.good()) {
        std::getline(in, line_str);
        if (line_str.empty()) {
            break;
        }

        std::string token;
        line_stream.clear();
        line_stream.str(line_str);

        size_t col = 0;
        while (line_stream.good()) {
            std::getline(line_stream, token, split_char);
            try {
                if (std::is_same_v<T, float>) {
                    data.at(row, col) = std::stof(token);
                }
                else if (std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>) {
                    data.at(row, col) = std::stoi(token);
                }
                else {
                    LOG(FATAL) << "Unsupported data type \n";
                }
            }
            catch (std::exception& e) {
                DLOG(ERROR) << "Parse CSV File meet error: " << e.what() << " row:" << row
                    << " col:" << col;
            }
            col += 1;
            CHECK(col <= cols) << "There are excessive elements on the column";
        }

        row += 1;
        CHECK(row <= rows) << "There are excessive elements on the row";
    }
    return data;
}

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
