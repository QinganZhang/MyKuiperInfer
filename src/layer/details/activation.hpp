#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP
#include "data/tensor.hpp"
#include "status_code.hpp"

namespace kuiper_infer {

namespace activation {

enum class ActivationType {
    kActivatetionUnknown = -1,
    kActivationRelu = 0,
    kActivationSilu = 1,
    kActivationSigmoid = 2,
    kActivationHardSwish = 3,
    kActivationHardSigmoid = 4,
    kActivationRelu6 = 5,
};

std::string ActivationTypeToString(ActivationType type);

using ActivationFunc = std::function<void(f_tensor_sptr, f_tensor_sptr)>;
/**
 * @brief 返回特定的激活函数（即返回function wrapper）
 * @details 这个函数原来是在smid.h文件中，考虑到只有激活函数这里才使用SIMD，因此移到这里
*/
ActivationFunc ApplySSEActivation(ActivationType act_type);

/**
 * @brief 真正进行激活运算的函数
 * @details 里面首先调用ActivationTypeToString判断激活函数类型，
 * 然后调用ApplySSEActivation得到具体的运算逻辑
 * （得到一个选定的激活函数的Forward(intput, output)的函数指针），
 * 最后尝试SIMD运行
*/
StatusCode ActivationForward(ActivationType type,
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs);

}  // namespace activation

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_ACTIVATION_HPP