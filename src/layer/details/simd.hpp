#ifndef KUIPER_INFER_INCLUDE_MATH_ARMA_SSE
#define KUIPER_INFER_INCLUDE_MATH_ARMA_SSE
#include "activation.hpp"

namespace kuiper_infer {

namespace activation {

/**
 * @brief SIMD只在激活函数中使用到了，还不如将simd中的内容直接放在activation的文件中
*/
ActivationFunc ApplySSEActivation(ActivationType act_type);

}  // namespace activation

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_INCLUDE_MATH_ARMA_SSE
