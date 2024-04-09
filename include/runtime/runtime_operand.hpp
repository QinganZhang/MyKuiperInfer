/**
 * @file runtime_operand.hpp
 * @brief
*/

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include <memory>
#include <string>
#include <vector>
#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace kuiper_infer {
    
/**
 * @brief Base for runtime graph operand
 *
 * Template base class representing an operand (input/output) in a
 * graph. Contains operand name, shape, data vector, and data type.
 *
 * @tparam T Operand data type (float, int, etc.)
 */

template <typename T>
struct RuntimeOperandBase {
    explicit RuntimeOperandBase() = default;

    explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes,
        std::vector<std::shared_ptr<Tensor<T>>> datas, RuntimeDataType type)
        : name(std::move(name)), shapes(std::move(shapes)), datas(std::move(datas)), type(type) {
    }

    size_t size() const;

    /**
     * @brief Name of the operand
     * 比如当前operand是输入operand，则此时name是输出当前operand的节点的name
    */
    std::string name;

    /// Shape of the operand
    std::vector<int32_t> shapes;

    /// Vector containing operand data
    std::vector<std::shared_ptr<Tensor<T>>> datas;

    /// Data type of the operand
    RuntimeDataType type = RuntimeDataType::kTypeUnknown;
};

template<typename T>
size_t RuntimeOperandBase<T>::size() const{
    if(shape.empty(())) return 0;
    size_t size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
    return size;
}

using RuntimeOperand = RuntimeOperandBase<float>;

using RuntimeOperandQuantized = RuntimeOperandBase<int8_t>;

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
