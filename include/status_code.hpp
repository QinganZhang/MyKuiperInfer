/**
 * @file status_code.hpp
 * @brief
 */

#ifndef KUIPER_INFER_INCLUDE_COMMON_HPP_
#define KUIPER_INFER_INCLUDE_COMMON_HPP_
namespace kuiper_infer {

enum class RuntimeParameterType {
  kParameterUnknown = 0,
  kParameterBool = 1,
  kParameterInt = 2,

  kParameterFloat = 3,
  kParameterString = 4,
  kParameterIntArray = 5,
  kParameterFloatArray = 6,
  kParameterStringArray = 7,
};

enum class StatusCode {
  kUnknownCode = -1,
  kSuccess = 0,

  kInferInputsEmpty = 1,
  kInferOutputsEmpty = 2,
  kInferParameterError = 3,
  kInferDimMismatch = 4,

  kFunctionNotImplement = 5,
  kParseWeightError = 6,
  kParseParameterError = 7,
  kParseNullOperator = 8,
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_COMMON_HPP_
