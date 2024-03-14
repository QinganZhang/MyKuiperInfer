#ifndef KUIPER_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
#define KUIPER_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
/**
 * @brief Runtime data types for operator attributes
 *
 * Enumerates the data types supported for operator attributes like
 * weights and biases.
 */
enum class RuntimeDataType {
    kTypeUnknown = 0,
    kTypeFloat32 = 1,
    kTypeFloat64 = 2,
    kTypeFloat16 = 3,
    kTypeInt32 = 4,
    kTypeInt64 = 5,
    kTypeInt16 = 6,
    kTypeInt8 = 7,
    kTypeUInt8 = 8,
};
#endif  // KUIPER_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
