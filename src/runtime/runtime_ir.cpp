#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "runtime/runtime_ir.hpp"
#include "utils/time/time_logging.hpp"

namespace kuiper_infer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {
}

void RuntimeGraph::set_bin_path(const std::string& bin_path) { this->bin_path_ = bin_path; }

void RuntimeGraph::set_param_path(const std::string& param_path) { this->param_path_ = param_path; }

const std::string& RuntimeGraph::param_path() const { return this->param_path_; }

const std::string& RuntimeGraph::bin_path() const { return this->bin_path_; }

static bool IsQuantizeOp(const pnnx::Operator* op) { return false; }

/**
 * @brief 根据PNNX中的节点，构建相同的KuiperInfer节点
 * @details
 * 初始化结果：对于每个节点，将PNNX节点的attribute和parameter复制（其实是移动）到KuiperInfer的节点中
 * 将KuiperInfer节点的输入输出operand都创建为与原来PNNX节点输入输出相同大小
*/
bool RuntimeGraph::Init() {
    if (this->bin_path_.empty() || this->param_path_.empty()) {
        LOG(ERROR) << "The bin path or param path is empty";
        return false;
    }

    this->graph_ = std::make_unique<pnnx::Graph>();
    int32_t load_result = this->graph_->load(param_path_, bin_path_);
    if (load_result != 0) {
        LOG(ERROR) << "Can not find the param path or bin path: " << param_path_ << " " << bin_path_;
        return false;
    }

    std::vector<pnnx::Operator*> operators = this->graph_->ops;
    if (operators.empty()) {
        LOG(ERROR) << "Can not read the layers' define";
        return false;
    }

    operators_.clear();
    for (const pnnx::Operator* op : operators) { /// 将PNNX中的Operator逐个导入到KuierInfer的Operator中
        if (!op) {
            LOG(ERROR) << "Meet the empty node in the model";
            continue;
        }
        else {
            if (!IsQuantizeOp(op)) {
                std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
                // 初始化节点的名称
                runtime_operator->name = op->name;
                runtime_operator->type = op->type;

                // 初始化节点中的input，即使用op->inputs去初始化runtime_operator中的相关信息，
                // 但是此时没有为输入输出operand分配空间，在Build时才分配
                InitGraphOperatorsInput(op->inputs, runtime_operator);

                // 记录输出operand中的名称
                InitGraphOperatorsOutput(op->outputs, runtime_operator);

                // 初始化节点中的attribute(权重)
                InitGraphAttrs(op->attrs, runtime_operator);

                // 初始化节点中的parameter
                InitGraphParams(op->params, runtime_operator);
                this->operators_.push_back(runtime_operator);
            }
            else {
                LOG(FATAL) << "UnSupported quantize operator in the model " << op->name
                    << " type: " << op->type;
            }
        }
    }

    graph_state_ = GraphState::NeedBuild;
    return true;
}

void RuntimeGraph::Build() {
    if (graph_state_ == GraphState::Complete) {
        LOG(INFO) << "Model has been built already!";
        return;
    }

    if (graph_state_ == GraphState::NeedInit) {
        bool init_graph = Init();
        LOG_IF(FATAL, !init_graph || graph_state_ == GraphState::NeedInit) << "Init graph failed!";
    }

    CHECK(graph_state_ >= GraphState::NeedBuild)
        << "Graph status error, current state is " << int32_t(graph_state_);
    LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty, may be no init";

    // 构建节点关系
    CreateNodeRelation();

    // 节点拓扑排序，此时operators_按照逆拓扑顺序进行排列
    ReverseTopoSort();

    // 初始化节点的输入和输出空间
    RuntimeOperatorUtils<float>::InitOperatorInput(operators_);
    RuntimeOperatorUtils<float>::InitOperatorOutput(graph_->ops, operators_);

    graph_state_ = GraphState::Complete;
    if (graph_ != nullptr) {
        graph_.reset();
        graph_ = nullptr;
    }
}

/**
 * @brief
 * @details 注意调用过程：
 * 1. 节点(RuntimeOperator类)中的算子(Layer类)调用算子基类的Forward()方法
 * 2. 在算子基类中准备输入输出，然后runtime_operator->layer->Forward，先指回节点类，然后通过多态，
 * 运行时指向特定派生类（特定算子）带参数版本的Forward，真正执行算子
*/
template <typename T>
StatusCode ExecuteLayer(const std::shared_ptr<Layer<T>>& layer, const std::string& op_name,
    const std::string& op_type, bool is_debug) {
    CHECK(layer != nullptr);
    StatusCode status;
    if (is_debug) {
        utils::LayerTimeLogging layer_time_logging(op_name, op_type);
        status = layer->Forward();
    }
    else {
        status = layer->Forward();
    }
    return status;
}

void RuntimeGraph::Forward(bool debug) {
    // 检查当前的执行图是否已经初始化完毕
    if (graph_state_ < GraphState::Complete) {
        LOG(FATAL) << "Graph need be build!"
            << ", current state is " << int32_t(graph_state_);
    }

    if (debug) {
        utils::LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
    }

    for (const auto& current_op : operators_) {
        // 节点按照逆拓扑顺序进行执行
        current_op->has_forward = false;
        CHECK_GT(current_op->forward_index, 0);

        if (is_input_op(current_op->name) || is_output_op(current_op->name)) {
            current_op->has_forward = true;
            continue;
        }

        CHECK(current_op->layer != nullptr)
            << "The layer corresponding to the op " << current_op->name
            << " is empty, indicating that it may not have been created.";

        // current_op->layer->Forward()
        StatusCode status = ExecuteLayer(current_op->layer, current_op->name, current_op->type, debug);

        CHECK(status == StatusCode::kSuccess)
            << current_op->layer->layer_name()
            << " layer forward failed, error code: " << int32_t(status);

        current_op->has_forward = true;
        PropagateLayerOutputs(current_op, current_op->output_operand->datas); // 前面是当前节点，后面是当前节点计算后的输出Tensor
    }

    if (debug) {
        utils::LayerTimeLogging::SummaryLogging();
    }

    for (const auto& op : operators_) {
        LOG_IF(FATAL, !op->has_forward) << "The operator: " << op->name << " has not been forward yet!";
    }
}

template <typename T>
std::shared_ptr<Layer<T>> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperatorBase<T>>& op) {
    LOG_IF(FATAL, !op) << "Operator is empty!";
    auto layer = LayerRegisterer::CreateLayer(op);
    LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
    return layer;
}

/**
 * @brief 初始化kuiperInfer节点中的input
 * @details
 * 根据PNNX节点op的输入operand inputs，构建kuiperInfer节点runtime_operator的输入operand
 * 对于PNNX节点的多个输入操作数inputs，对于每个操作数input，创建KuiperInfer的节点runtime_operator，并将input的属性赋值给runtime_operator对应的输入Operand
*/
template <typename T>
void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
    if (inputs.empty()) {
        return;
    }
    CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
    for (const pnnx::Operand* input : inputs) {
        if (!input) {
            continue;
        }

        std::vector<int32_t> dims;
        // 当前的input张量是节点producer的输出
        const pnnx::Operator* producer = input->producer;


        for (int32_t dim : input->shape) {
            dims.push_back(dim);
        }
        CHECK(!dims.empty());

        // runtime_operand是当前节点的输入，根据PNNX节点的输入构建而来
        std::shared_ptr<RuntimeOperandBase<T>> runtime_operand =
            std::make_shared<RuntimeOperandBase<T>>();
        runtime_operand->name = producer->name; /// 比如产生这个operand的节点是conv_1，则该输入operand的name也是conv_1
        runtime_operand->shapes = dims;
        runtime_operator->input_operands_map.insert({ producer->name, runtime_operand });
        runtime_operator->input_operands_seq.push_back(runtime_operand);

        switch (input->type) {
            case 1: {
                    runtime_operand->type = RuntimeDataType::kTypeFloat32;
                    break;
                }
            case 7: {
                    runtime_operand->type = RuntimeDataType::kTypeInt8;
                    break;
                }
            default: {
                    LOG(FATAL) << "Unknown input operand type: " << input->type;
                }
        }
    }
}


/**
 * @brief 记录输出operand中的名称
 * @details
 * 根据PNNX节点op的输出operand outputs，在kuiperInfer节点runtime_operator的输出operand中记录使用该operand的节点名称，即记录当前节点的后继节点
*/
template <typename T>
void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
    if (outputs.empty()) {
        return;
    }
    CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
    for (const pnnx::Operand* output : outputs) {
        if (!output) {
            continue;
        }
        const auto& consumers = output->consumers;
        for (const auto& c : consumers) {
            runtime_operator->output_names.push_back(c->name);
        }
    }
}

template <typename T>
void RuntimeGraph::InitGraphParams(
    const std::map<std::string, pnnx::Parameter>& params,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
    if (params.empty()) {
        return;
    }
    CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
    for (const auto& [name, parameter] : params) {
        const int32_t type = parameter.type;
        switch (type) {
            case int32_t(RuntimeParameterType::kParameterUnknown): {
                    std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }

            case int32_t(RuntimeParameterType::kParameterBool): {
                    std::shared_ptr<RuntimeParameterBool> runtime_parameter =
                        std::make_shared<RuntimeParameterBool>(parameter.b);
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }

            case int32_t(RuntimeParameterType::kParameterInt): {
                    std::shared_ptr<RuntimeParameterInt> runtime_parameter =
                        std::make_shared<RuntimeParameterInt>(parameter.i);
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }

            case int32_t(RuntimeParameterType::kParameterFloat): {
                    std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
                        std::make_shared<RuntimeParameterFloat>(parameter.f);
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }

            case int32_t(RuntimeParameterType::kParameterString): {
                    std::shared_ptr<RuntimeParameterString> runtime_parameter =
                        std::make_shared<RuntimeParameterString>(parameter.s);
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }

            case int32_t(RuntimeParameterType::kParameterIntArray): {
                    std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
                        std::make_shared<RuntimeParameterIntArray>(parameter.ai);
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }

            case int32_t(RuntimeParameterType::kParameterFloatArray): {
                    std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
                        std::make_shared<RuntimeParameterFloatArray>(parameter.af);
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }
            case int32_t(RuntimeParameterType::kParameterStringArray): {
                    std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
                        std::make_shared<RuntimeParameterStringArray>(parameter.as);
                    runtime_operator->params.insert({ name, runtime_parameter });
                    break;
                }
            default: {
                    LOG(FATAL) << "Unknown parameter type: " << type;
                }
        }
    }
}

template <typename T>
void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
    if (attrs.empty()) {
        return;
    }
    CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
    for (const auto& [name, attr] : attrs) {
        switch (attr.type) {
            case 1: { // float32
                    std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>(
                        attr.shape, RuntimeDataType::kTypeFloat32, attr.data);
                    runtime_operator->attribute.insert({ name, runtime_attribute });
                    break;
                }
            default: {
                    LOG(FATAL) << "Unknown attribute type: " << attr.type;
                }
        }
    }
}


// 将当前节点的输出，传播到当前节点后继节点的输入中
template <typename T>
void RuntimeGraph::PropagateLayerOutputs(
    const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
    const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_datas) {
    
    // For each next operator of current operator
    for (const auto& [_, output_op] : current_op->output_operators_map) { // current_op的后继节点

        // 对于当前节点的每一个后继节点output_op，得到其输入Operands的map
        const auto& next_input_operands_map = output_op->input_operands_map; 

        const auto& next_input_op_iter = next_input_operands_map.find(current_op->name);

        // 在后继节点的输入Operands的map中，找到了当前节点的名字
        if (next_input_op_iter != next_input_operands_map.end()) {

            // 后继节点的输入Operand中保存的Tensors
            std::vector<tensor_sptr<T>>& next_input_datas = next_input_op_iter->second->datas;

            // Copy current op output data to next op input data
            for (uint32_t i = 0; i < next_input_datas.size(); ++i) {
                // 从当前节点的输出Tensors中，取出指向第i维Tensor的指针
                const tensor_sptr<T>& layer_output_data = layer_output_datas.at(i);
                if (next_input_datas.at(i) != nullptr) {
                    CHECK(next_input_datas.at(i)->shapes() == layer_output_data->shapes());
                }
                // 检查输入输出形状相同后，将后继节点的对应于当前节点的输入Operand，将对应维度的数据成员（即Tensor）指向当前节点对应的Tensor
                // 即整个过程没有出现数据复制，只是将后继节点中指向输入Tensor的指针也指向了当前节点输出Tensor
                // 这与构建时，为输出Operand开辟空间，而不为输入Operand开辟空间相一致
                next_input_datas.at(i) = layer_output_data;
            }
        }
    }
}

/**
 * @brief 构建得到逆拓扑排序，输入节点现在排在第一个，输出节点现在是最后一个
 * @details 拓扑排序是遍历时后面的节点排在前面
*/
void RuntimeGraph::ReverseTopoSort() {
    // 构建拓扑顺序
    for (const auto& op : operators_) {
        // 根据输入节点构建拓扑排序，外层for循环防止出现非连通分支
        if (op != nullptr && !op->has_forward) {
            int32_t current_forward_idx = 0;
            this->ReverseTopoSortInternal(op, current_forward_idx);
        }
    }

    // 按照ReverseTopoSort的排序，forward_index小的节点在靠近输出的位置，forward_index大的节点在靠近输入的位置
    // 根据拓扑顺序调整节点的执行顺序，调整为拓扑逆序顺序，即按照forward_index从大到小的规则进行排序，即靠近输入的节点往前排，靠近输出的节点往后排
    std::sort(operators_.begin(), operators_.end(), [](const auto& op1, const auto& op2) {
        return op1->forward_index > op2->forward_index;
        });

    int32_t forward_index = 1;
    for (const auto& op : operators_) {
        op->forward_index = forward_index;
        forward_index += 1;
    }
}

/**
 * @brief 从root_op开始拓扑排序，最终将拓扑顺序记录在每个节点的forward_index属性中
 * @details forward_index小的节点是后面靠近输出的节点，forward_index大的节点是靠近输入的节点
 * @param root_op
 * @param current_forward_idx 记录拓扑排序中节点在访问中的顺序
*/
template <typename T>
void RuntimeGraph::ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_op,
    int32_t& current_forward_idx) {
    if (!root_op) {
        LOG(INFO) << "Current operator is nullptr";
        return;
    }
    /// 当前节点的输入操作数是空的（当前节点没有上一个节点），而且还没访问过当前节点
    if (root_op->input_operands_map.empty() && !root_op->has_forward) {
        this->input_ops_.push_back(root_op);
    }
    /// 后继节点的名字列表为空，而且还没访问过当前节点
    if (root_op->output_names.empty() && !root_op->has_forward) {
        this->output_ops_.push_back(root_op);
    }

    root_op->has_forward = true;
    const auto& next_ops = root_op->output_operators_map;
    for (const auto& [_, op] : next_ops) {
        if (op != nullptr && !op->has_forward) { // op是root_op的后继节点
            this->ReverseTopoSortInternal(op, current_forward_idx);
        }
    }

    for (const auto& [_, op] : next_ops) {
        CHECK_EQ(op->has_forward, true);
    }
    root_op->forward_index = current_forward_idx;
    current_forward_idx += 1;
}

/**
 * @brief 构建节点关系
 * @details
 * 首先构建每个节点的output_operators，即每个节点可以根据后继节点的名字找到该后继节点（map{name: 后继节点}）
 * 然后对于每个节点，构建layer
*/
void RuntimeGraph::CreateNodeRelation() {
    // 构建图关系
    for (const auto& current_op : this->operators_) {
        // 对于图中的每个节点，首先获取当前节点的后继节点的names
        // 然后再遍历所有节点，找到对应的后继节点，将{name: 后继节点}更新到当前节点的output_operators中
        const std::vector<std::string>& output_names = current_op->output_names;
        for (const auto& kOutputName : output_names) {
            for (const auto& output_op : this->operators_) {
                if (output_op != current_op && output_op->name == kOutputName) {
                    // output_op 是 当前节点的一个后继节点
                    current_op->output_operators_map.insert({ kOutputName, output_op });
                }
            }
        }

        // 除了输入和输出节点，对于每个节点，都创建layer
        if (current_op->type != "pnnx.Input" && current_op->type != "pnnx.Output") {
            auto layer = RuntimeGraph::CreateLayer(current_op);
            if (layer) {
                current_op->layer = layer;
                layer->set_runtime_operator(current_op);
            }
            else {
                LOG(FATAL) << "Layer " << current_op->name << " create failed!";
            }
        }
    }
}

RuntimeGraph::GraphState RuntimeGraph::graph_state() const { return this->graph_state_; }

void RuntimeGraph::set_inputs(const std::string& input_name, const std::vector<f_tensor_sptr>& inputs) {
    CHECK(this->graph_state_ == GraphState::Complete);
    std::shared_ptr<RuntimeOperator> input_op;
    for (auto op : this->input_ops_) {
        if (op->name == input_name) {
            input_op = op;
            break;
        }
    }
    CHECK(input_op != nullptr) << "Can not find the input operator: " << input_name;
    PropagateLayerOutputs(input_op, inputs);
}

std::vector<f_tensor_sptr> RuntimeGraph::get_outputs(const std::string& output_name) const {
    CHECK(this->graph_state_ == GraphState::Complete);
    std::shared_ptr<RuntimeOperator> output_op;
    for (auto op : this->output_ops_) {
        if (op->name == output_name) {
            output_op = op;
        }
    }

    CHECK(output_op != nullptr) << "Can not find the output operator: " << output_name;
    std::vector<f_tensor_sptr> outputs;
    for (const auto& input_operand : output_op->input_operands_seq) {
        std::copy(input_operand->datas.begin(), input_operand->datas.end(),
            std::back_inserter(outputs));
    }
    return outputs;
}

bool RuntimeGraph::is_input_op(const std::string& op_name) const {
    for (auto op : this->input_ops_) {
        CHECK(op != nullptr);
        if (op->name == op_name) {
            return true;
        }
    }
    return false;
}

bool RuntimeGraph::is_output_op(const std::string& op_name) const {
    for (auto op : this->output_ops_) {
        CHECK(op != nullptr);
        if (op->name == op_name) {
            return true;
        }
    }
    return false;
}

}  // namespace kuiper_infer
