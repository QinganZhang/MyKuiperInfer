#include "runtime/runtime_operator.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {

/**
 * @brief use pnnx_operators to init kuiperInfer_operators, 
 *  just convert kuiperInfer_operators's input operands into the counterpart shape of pnnx_operands's input operands
 * 将kuiperInfer_operators中节点的输入操作数init成 pnnx_operators中节点的输入操作数相同的维度，不要求数值相同
*/
void RuntimeOperatorUtils<float>::InitOperatorInput(
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
    if (operators.empty()) {
        LOG(ERROR) << "Operators for init input shapes is empty!";
        return;
    }

    for (const auto& op : operators) {
        if (op->input_operands_map.empty()) {
            continue;
        }
        else {
            const std::map<std::string, std::shared_ptr<RuntimeOperand>>& input_operands_map =
                op->input_operands_map;
            // 初始化operator的输入空间
            for (const auto& [_, input_operand] : input_operands_map) { // for each pair{input_name, input_operand}
                if (!input_operand) {
                    continue;
                }
                const auto& type = input_operand->type;
                auto& input_operand_datas = input_operand->datas;  // 需要初始化的输入空间, input_operand_datas 's type is std::vector<std::shared_ptr<Tensor<float>>>
                CHECK(type == RuntimeDataType::kTypeFloat32) << "The graph only support float32 yet!";
                const auto& input_operand_shape = input_operand->shapes;

                CHECK(!input_operand_shape.empty());
                const int32_t batch = input_operand_shape.at(0);
                CHECK(batch > 0) << "Dynamic batch size is not supported!";
                CHECK(input_operand_shape.size() == 2 || input_operand_shape.size() == 4 ||
                    input_operand_shape.size() == 3)
                    << "Unsupported tensor shape sizes: " << input_operand_shape.size();

                if (!input_operand_datas.empty()) { /// 后面运行时，检查operand的形状是否与operand中存储的数据的形状相同
                    CHECK_EQ(input_operand_datas.size(), batch); // input_operand_datas中有input_operand_datas.size()个指向Tensor<float>的指针
                }
                else { /// 第一次运行时，input_operand_datas是空的，所以先在数组中准备好batch个指向空Tensor<float>的指针
                    input_operand_datas.resize(batch);
                }
            }
        }
    }
}


/**
 * @brief create a Tensor<float> in the shape of operand_shapes
 * @return return std::shared_ptr<Tensor<float>>
*/
static f_tensor_sptr CreateTensor(const std::vector<int32_t>& operand_shapes) {
    switch (operand_shapes.size()) {
        case 4:
            return TensorCreate<float>(operand_shapes[1], operand_shapes[2], operand_shapes[3]);
        case 3:
            return TensorCreate<float>(operand_shapes[1], operand_shapes[2]);
        case 2:
            return TensorCreate<float>(operand_shapes[1]);
        default:
            LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
            return nullptr;
    }
}

/**
 * @brief transform output_tensor into the shape of operand_shapes
 * 
*/
static void CheckAndReshapeTensor(f_tensor_sptr& output_tensor,
    const std::vector<int32_t>& operand_shapes) {
    const std::vector<uint32_t>& tensor_shapes = output_tensor->shapes();
    switch (operand_shapes.size()) {
        case 4:
            if (tensor_shapes[0] != operand_shapes[1] || tensor_shapes[1] != operand_shapes[2] ||
                tensor_shapes[2] != operand_shapes[3]) {
                output_tensor->Reshape({ (uint32_t)operand_shapes[1], (uint32_t)operand_shapes[2],
                                        (uint32_t)operand_shapes[3] });
            }
            break;
        case 3:
            if (tensor_shapes[0] != 1 || tensor_shapes[1] != operand_shapes[1] ||
                tensor_shapes[2] != operand_shapes[2]) {
                output_tensor->Reshape({ (uint32_t)operand_shapes[1], (uint32_t)operand_shapes[2] });
            }
            break;
        case 2:
            if (tensor_shapes[0] != 1 || tensor_shapes[1] != operand_shapes[1] || tensor_shapes[2] != 1) {
                output_tensor->Reshape({ (uint32_t)operand_shapes[1] });
            }
            break;
        default:
            LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
            break;
    }
}

/**
 * @brief use pnnx_operators to init kuiperInfer_operators, 
 *  just convert kuiperInfer_operators's output operands into the counterpart shape of pnnx_operands's output operands
 * 将kuiperInfer_operators中节点的输出操作数init成 pnnx_operators中节点的输出操作数相同的维度
*/
void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator*>& pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) 
    {
    CHECK(!pnnx_operators.empty() && !operators.empty() && pnnx_operators.size() == operators.size());
    CHECK(pnnx_operators.size() == operators.size());
    for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
        const std::vector<pnnx::Operand*> operands = pnnx_operators[i]->outputs;
        if (operands.empty()) continue;
        if (operands.size() > 1) {
            LOG(FATAL) << "Only support one node one output yet! Every node shoud have one output operand now";
        }

        pnnx::Operand* operand = operands[0]; // pnnx operator 's only output operand
        CHECK(operand != nullptr && !operand->shape.empty()) << "Operand output is null or empty!";
        std::vector<int32_t> operand_shapes;
        std::copy_if(operand->shape.begin(), operand->shape.end(), std::back_inserter(operand_shapes),
            [](int32_t dim) { return dim > 0; });
        const auto& runtime_op = operators[i];
        const auto& output_tensors = runtime_op->output_operand; // KuiperInfer's only output operand 
        CHECK((operand_shapes.size() == 2 || operand_shapes.size() == 4 || operand_shapes.size() == 3))
            << "Unsupported shape sizes: " << operand_shapes.size();

        const int32_t batch = operand_shapes[0];
        CHECK_EQ(operand->type, 1) << "The type of pnnx operand is not float32";
        if (!output_tensors) { // output_tensors == nullptr
            std::vector<f_tensor_sptr> output_operand_datas;
            for (uint32_t j = 0; j < batch; ++j) {
                output_operand_datas.push_back(CreateTensor(operand_shapes));
            }
            runtime_op->output_operand =
                std::make_shared<RuntimeOperand>(operand->name + "_output", operand_shapes,
                    output_operand_datas, RuntimeDataType::kTypeFloat32);
        }
        else {
            CHECK(batch == output_tensors->datas.size());
            CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
            CHECK(output_tensors->shapes == operand_shapes);
            for (uint32_t b = 0; b < batch; ++b) {
                f_tensor_sptr output_tensor = output_tensors->datas[b];
                CheckAndReshapeTensor(output_tensor, operand_shapes);
            }
        }
    }
}

}  // namespace kuiper_infer
