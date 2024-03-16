#ifndef KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {

class ReluLayer : public NonParamLayer {
public:
    ReluLayer() : NonParamLayer("Relu") {}

    StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
        std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    /**
     * @brief 创建算子的函数，实际使用时经常作为函数指针传入注册函数中
    */
    static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer<float>>& relu_layer);
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
