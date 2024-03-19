## Tensor

### 背景

在推理框架中，只需要进行模型结构的加载、模型权重的加载，然后进行前向运算，整个过程不需要反向传播。

> 有时模型结构和权重信息会放在一个文件中，比如ONNX格式；但也可以将模型结构和权重文件分开存放，比如该项目用到的PNNX格式

其中模型结构涉及到计算图的构建，模型权重会在构建时加载到Tensor类中

### 设计

Tensor一般保存的是三维的数组，最简单的方法就是`std::vector<std::vector<std::vector<float>>>`，但是这种方法非常不利于**数据的访问（尤其是内存不连续的问题，访问会变慢） 、修改以及查询，特别是在扩容的时候非常不方便。不能满足使用需求**。因此最后基于Armadillo类中的`arma::Cube`来进行封装，一个Cube由多个Mat组成，实现了效率与工作量两方面的折中。（需要注意的是，Armadillo类中的`arma::Cube<T>`是以列优先方式存储数据，而PyTorch中导出的文件是以行优先方式存储数据，读取模型权重文件到Tensor类中时，中间要进行一步转置）

模型权重文件是通过PyTorch导出的，但是PyTorch直接导出的文件具有特定的格式，不方便解析。因此PyTorch首先读取模型的权重文件，然后将权重文件转换为numpy管理，再保存为本地csv文件格式，这样方便解析和读取。



## 计算图

### 背景

从PyTorch导出的模型结构的保存方法有两种，一种是将模型结构和模型权重保存在一起，比如ONNX，另一种是将模型结构和模型权重分开保存，比如PNNX，这里使用后一种方法，理由如下：

- ONNX计算图过于细碎，不易理解和阅读
    - 算子过于细碎，有助于兼容更多的框架
    - 而PNNX导出的算子可以保持完整的大算子不被拆分
- ONNX计算图过于细碎，也不利于推理的优化

`PNNX`是`PyTorch Neural Network Exchange`的缩写，能够将`PyTorch`模型文件直接导出为高效、简洁的计算图，作为一种中间格式，PNNX可以进行一些图优化、算子融合的工作，它有以下几个特点：

- 用模板匹配（`pattern matching`）的方法将匹配到的子图用对应等价的大算子替换掉，例如可以将上图子图中的多个小算子（可能是在`TorchScript`中被拆分的）重新替换为`LayerNorm`算子。或者在对`PyTorch`模型导出时，也可以自定义某个`nn.Module`不被拆分；
- 在`PyTorch`中编写的简单算术表达式在转换为`PNNX`后，会保留表达式的整体结构，而不会被拆分成许多小的加减乘除算子。例如表达式`add(mul(@0, @1),add(@2, @3))`不会被拆分为两个`add`算子和一个`mul`算子，而是会生成一个表达式算子`Expression` ;
- `PNNX`项目中有大量图优化的技术，包括了算子融合，常量折叠和移除，公共表达式消除等技术。
    * 算子融合优化是一种针对深度学习神经网络的优化策略，通过将多个相邻的计算算子合并为一个算子来减少计算量和内存占用。
    * 常量折叠是将**在编译时期间将表达式中的常量计算出来，然后将结果替换为一个等价的常量**，以减少模型在运行时的计算量。
    * 常量移除就是将计算图中不需要的常数（**计算图推理的过程中未使用**）节点删除，从而减少计算图的文件和加载后的资源占用大小。
    * 公共表达式消除优化是一种针对计算图中重复计算的优化策略，**它可以通过寻找并合并重复计算的计算节点，减少模型的计算量和内存占用。**公共子表达式检测是指**查找计算图中相同的子表达式**，公共子表达式消除是指**将这些重复计算的计算节点合并为一个新的计算节点**，从而减少计算和内存开销。

### 设计

PNNX计算图中有两个核心的部分，`Operand`（操作数）和 `Operator`（节点），整个计算图`Graph`主要就是针对操作数和节点的管理。

#### PNNX计算图核心结构

##### `Operand`操作数

```cpp
class Operand
{
public:
    void remove_consumer(const Operator* c);
    Operator* producer;		// 产生这个操作数的节点，即这个producer输出了当前这个Operand
    std::vector<Operator*> consumers;	// 使用这个操作数的节点，即当前这个Operand是comsumers中每个的输入
    
    int type;
    std::vector<int> shape;

    std::string name;
    std::map<std::string, Parameter> params;
};
```

##### `Operator`节点

```cpp
class Operator // 计算图中的运算符（算子）
{
public:
    std::vector<Operand*> inputs;	// 该算子需要的输入操作数
    std::vector<Operand*> outputs;	// 该算子计算得到的输出操作数

    std::string type;
    std::string name;

    std::vector<std::string> inputnames;	// 
    std::map<std::string, Parameter> params; // 该运算符的参数，比如conv中的stride，padding，kernel_size等
    std::map<std::string, Attribute> attrs;	 // 该运算符的权重，比如conv中的weight，bias
};
```

`Parameter`参数

```cpp
class Parameter{
    int type; 
}
```

`Attribute`权重

```cpp
class Attribute{
    int type; 
    std::vector<int> shape;
}
```

##### `Graph`计算图

```cpp
class Graph // 管理计算图中的运算符和和操作数
{
    std::vector<Operator*> ops;		// 运算符（算子）的集合
    std::vector<Operand*> operands; // 操作数的集合
};
```

#### 对PNNX中间格式进行封装

##### 对`Operand`的封装：`RuntimeOperand`

```cpp
template <typename T>
struct RuntimeOperandBase {
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
```

##### 对`Operator`的封装：`RuntimeOperator`

```cpp
template <typename T>
struct RuntimeOperatorBase {
    /// Execution order index of this operator，记录拓扑排序中节点的执行顺序
    int32_t forward_index = -1;

    /// Whether this operator has run in current execution，在拓扑排序中判断当前节点是否已经遍历过
    bool has_forward = false;

    /// Name of the operator,全局唯一,比如Conv_1
    std::string name;

    /// Type of the operator, such as Convolution
    std::string type;

    /// Layer for this operator,节点对应的算子，负责完成具体计算
    std::shared_ptr<Layer<T>> layer;

    /// Names of output operators, 当前节点的后继节点的名字
    std::vector<std::string> output_names;

    /// Output operand, 注意只有一个输出operand
    std::shared_ptr<RuntimeOperandBase<T>> output_operand;

    /// Output operators mapped by output name, 当前节点的后继节点的按名访问映射
    std::map<std::string, std::shared_ptr<RuntimeOperatorBase<T>>> output_operators;

    /// Input operands in sequence
    std::vector<std::shared_ptr<RuntimeOperandBase<T>>> input_operands_seq;

    /**
     * @brief Input operands mapped by provider name
     * <上一个节点的名字，当前节点的输入Operand>的map
    */
    std::map<std::string, std::shared_ptr<RuntimeOperandBase<T>>> input_operands;

    /// Operator parameters, such kernel_size, stride for conv
    std::map<std::string, std::shared_ptr<RuntimeParameter>> params;

    /// Operator attributes like weights and bias
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
};
```

对`Parameter`的封装：`RuntimeParameter`

```cpp
struct RuntimeParameter{
    RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
}
```

对`Attribute`的封装：`RuntimeAttribute`

```cpp
struct RuntimeAttribute {
	/// Attribute data，节点中的权重信息，注意保存的是二进制数据（所以是char类型）
    std::vector<char> weight_data;
    std::vector<int32_t> shape;
    RuntimeDataType type = RuntimeDataType::kTypeUnknown;
}
```

##### 对`Graph`的封装：`RuntimeGraph`

```cpp
class RuntimeGraph{
private:
    std::string bin_path_;
    std::string param_path_;
    std::unique_ptr<pnnx::Graph> graph_;

    GraphState graph_state_ = GraphState::NeedInit;
    
    /// 整个图的输入节点，这些节点的input_operands都是空的，直接将计算图的输入作为input_operands
    std::vector<std::shared_ptr<RuntimeOperator>> input_ops_; 
    
    /// 整个图的输出节点，这些节点的output_names都是空的，直接其output_operand作为整个计算图的输出
    std::vector<std::shared_ptr<RuntimeOperator>> output_ops_; 
    std::vector<std::shared_ptr<RuntimeOperator>> operators_;
}
```

整体计算图的[UML类图](http://www.plantuml.com/plantuml/dsvg/dLTFRzn45B_xKup4WHQI7k1cHQlI8b0br5QG0pThTZt9Ml6EtPbnqYXOGcqHAcfAfAGgj18j4bK0YQG73kasQRvCl4vFV0MUPsOdunsleyqbYjzxytlpVk_FlBtA1MOY6yJUXwXuCIn__txqvFLeSwzykptwxO7NYp7dQ95Gdh25nGxQy12QHo4ME40-mco0_UjPbu1AAXXU2tWVfuHNQYv2trybFG5diuYAJpy9HCVBFOtwTKP5D22V2S6YRYQ81FyOhP5ekI-2oiS4Hg-FMKVQI1zrfSPNUh6U4kIFFNmEK8iWyVAwvzkOuG4HX85dzvJHwgD0qEZbGN5ytoP8szvA2SCql8OvqIvSXQOF_5307OV68NYwZCwYugWbhegKIGC50zIGC4W5Mp39whHnWRmSUUubKJ8_hzFjDKQ2Dd9lGdBbodSTr404JGH1CLaIAA4eZkWAyfjeMOeJB3hdqmknElmQJPtWfOIjOHO64Gt9NV2gEPIwFkQ1reEccoGncf8KYpr77EDrmhJS9lyha4lkfLMIYJGJ0K0xQRK0QxQ99jM5Rgmezoo0ys20qp42_6ixcsl1jPdKo16rBYLnBm9BmrPKJkwqbNGxLbgrAPyrMjFS6WMR8JGZa9bU09Hx3fUwezm0Matj2xOQsXTeD5m0GwX95yRfMdEalM-DeMbSstwfoT3txJu0kbK_WB6I8rEtSwWusYuURfPxJuj_jifd3ugjn-NRNVLPR6sMkwyzGaDKUPp1houqy2pw18AAr-4FCTwLrW2Z6P5Dnm7-hOc048XJ4gBkklOcO_TGMvpmENhq4A8ztJj-_tHljdkv_GhEq0rg1MfVAcN9QXGIfgurcoUdXy4J9smL136AQXujvDOv_oN3QTyUy5garuvkH8CKYtxW3EHvDGQQj4YgboiOeybeT2SZierJIk_wllJIiKsvm89K1wJgc-fBgSTnEnlEENEre-5DvSmEeCxhpulwlq4ZSUcoVlhaxBkMHN4eHQj9528f-qAza9SgepDD3icpyoB7YqgQBNP67hXtIygO-xx5ANWVFVdtvE3lulaxw8JJTzl53wzEVxfNtdlpt-6Zq-DVJuRFrITetMf_V_zezCj--TSGjARgMdr3FjgS_pYSvjmUQirkPR7Vlddpsz7Bm-BmPz3LTkiKzrRmCf4Mii9LuPl9lu50PkqW4HKWo8sXcVmRJNPceNJtoKuDHZzA5m6PkZUT_p2VXQc1gbaKPZsvLOsBvYEjwYMKPmF29fSeOpUEICTIrQK1iSI9BGQvNHRQ2C3DHQJta-5kkVFYTF-LraCMrPm6hSdqL2npSuLrT5pk_5d3kbJyyGmAxhABQH9adcox81pEMvEXvt43UpIKv9ajiit809s_qCvU0Ne9csvkobqds_0Yqsh09XNc69hKwD3EY3PlZQ8ZRIatRQ8dJmjd-RWq6FoghMf3U_6HzV9uI4lj0dM5L0xdAGCuz_GFDhNlgHz26xWBzE6UoFL4LtE-UiuoVZq9oJGhcMO-fsk2EuS-KB-0vFHpdcFGCdL9io0PcpM2LngV9o-lJMaLXUyxG01_Sl0OUaXlXVjlYk4ED9AQoxLDuofyP8FuVm00)及[Plantuml代码](MyKuiperInfer/notes/uml/ComputeGraph_Structure.wsd)

![](http://www.plantuml.com/plantuml/dsvg/dLTFRzn45B_xKup4WHQI7k1cHQlI8b0br5QG0pThTZt9Ml6EtPbnqYXOGcqHAcfAfAGgj18j4bK0YQG73kasQRvCl4vFV0MUPsOdunsleyqbYjzxytlpVk_FlBtA1MOY6yJUXwXuCIn__txqvFLeSwzykptwxO7NYp7dQ95Gdh25nGxQy12QHo4ME40-mco0_UjPbu1AAXXU2tWVfuHNQYv2trybFG5diuYAJpy9HCVBFOtwTKP5D22V2S6YRYQ81FyOhP5ekI-2oiS4Hg-FMKVQI1zrfSPNUh6U4kIFFNmEK8iWyVAwvzkOuG4HX85dzvJHwgD0qEZbGN5ytoP8szvA2SCql8OvqIvSXQOF_5307OV68NYwZCwYugWbhegKIGC50zIGC4W5Mp39whHnWRmSUUubKJ8_hzFjDKQ2Dd9lGdBbodSTr404JGH1CLaIAA4eZkWAyfjeMOeJB3hdqmknElmQJPtWfOIjOHO64Gt9NV2gEPIwFkQ1reEccoGncf8KYpr77EDrmhJS9lyha4lkfLMIYJGJ0K0xQRK0QxQ99jM5Rgmezoo0ys20qp42_6ixcsl1jPdKo16rBYLnBm9BmrPKJkwqbNGxLbgrAPyrMjFS6WMR8JGZa9bU09Hx3fUwezm0Matj2xOQsXTeD5m0GwX95yRfMdEalM-DeMbSstwfoT3txJu0kbK_WB6I8rEtSwWusYuURfPxJuj_jifd3ugjn-NRNVLPR6sMkwyzGaDKUPp1houqy2pw18AAr-4FCTwLrW2Z6P5Dnm7-hOc048XJ4gBkklOcO_TGMvpmENhq4A8ztJj-_tHljdkv_GhEq0rg1MfVAcN9QXGIfgurcoUdXy4J9smL136AQXujvDOv_oN3QTyUy5garuvkH8CKYtxW3EHvDGQQj4YgboiOeybeT2SZierJIk_wllJIiKsvm89K1wJgc-fBgSTnEnlEENEre-5DvSmEeCxhpulwlq4ZSUcoVlhaxBkMHN4eHQj9528f-qAza9SgepDD3icpyoB7YqgQBNP67hXtIygO-xx5ANWVFVdtvE3lulaxw8JJTzl53wzEVxfNtdlpt-6Zq-DVJuRFrITetMf_V_zezCj--TSGjARgMdr3FjgS_pYSvjmUQirkPR7Vlddpsz7Bm-BmPz3LTkiKzrRmCf4Mii9LuPl9lu50PkqW4HKWo8sXcVmRJNPceNJtoKuDHZzA5m6PkZUT_p2VXQc1gbaKPZsvLOsBvYEjwYMKPmF29fSeOpUEICTIrQK1iSI9BGQvNHRQ2C3DHQJta-5kkVFYTF-LraCMrPm6hSdqL2npSuLrT5pk_5d3kbJyyGmAxhABQH9adcox81pEMvEXvt43UpIKv9ajiit809s_qCvU0Ne9csvkobqds_0Yqsh09XNc69hKwD3EY3PlZQ8ZRIatRQ8dJmjd-RWq6FoghMf3U_6HzV9uI4lj0dM5L0xdAGCuz_GFDhNlgHz26xWBzE6UoFL4LtE-UiuoVZq9oJGhcMO-fsk2EuS-KB-0vFHpdcFGCdL9io0PcpM2LngV9o-lJMaLXUyxG01_Sl0OUaXlXVjlYk4ED9AQoxLDuofyP8FuVm00)

### 根据PNNX的`Graph`构建KuiperInfer的`RuntimeGraph`

1. 加载模型结构和权重信息的文件，得到PNNX的Graph

2. 根据PNNX Graph中的节点信息，为KuiperInfer构建相同信息的节点（`RuntimeGraph::Init()`）。对于PNNX Graph中的每个节点Operator，构造一个对应的RuntimeOperator，进行以下初始化

    - 根据Operator的输入Operand，初始化RuntimeOperator中相关信息（注意只是将输入Operand的属性复制过来，没有将对应数据复制）。`RuntimeGraph::InitGraphOperatorsInput`
    - 根据Operator的输出Operand，初始化RuntimeOperator中相关信息（注意只是将输出Operand的属性复制过来，没有将对应数据复制）。`RuntimeGraph::InitGraphOperatorsOutput`
    - 将Operator的权重Attribute，属性复制，数据内容移动过来（`std::move`）。`RuntimeGraph::InitGraphAttrs`
    - 将Operator的参数Parameter，属性复制，数据内容移动过来（`std::move`）。`RuntimeGraph::InitGraphParams`

3. 根据PNNX Graph中的图结构，为KuiperInfer构建相同的图结构（`RuntimeGraph::Build()`）。

    - 首先构建节点关系，主要包含两个方面：

        - 找到当前节点的后继节点，更新每个RuntimeOperator的output_operators信息（即找到当前节点的后继节点有哪些，这样才能构建计算图）。`RuntimeGraph::CreateNodeRelation`

        - 为每个节点创建算子（除了输入和输出算子）。在`RuntimeGraph::CreateLayer`中，调用基于工厂模式设计的`LayerRegisterer::CreateLayer`，返回创建的算子，赋值给节点的`layer`属性

            > 之前也没有显示注册算子，是什么时候添加的呢？在刚开始运行推理框架时，算子的注册过程都是全局变量（在对应算子的实现文件最后面），一开始就已经将所有算子注册了

    - 然后根据后继节点信息，得到计算图的逆拓扑排序（当然先拓扑排序，然后再resever）,此时`RuntimeGraph::operators_`中节点按照逆拓扑顺序进行排列

        > 逆拓扑排序的结果是，靠近计算图输入的节点排的靠前，靠近计算图输出的节点排的靠后

    - 最后为每个节点的输出Operand分配空间（`RuntimeOperatorUtils<float>::InitOperatorOutput`），但是输入Operand不用分配空间（`RuntimeOperatorUtils<float>::InitOperatorInput`），输入Operand可以复用前一个节点输出Operand的空间。

        > 这两个函数的代码有点没太看懂

4. 图结构和权重信息转换和构建完毕



## 算子

上面计算图中最核心的数据结构就是节点`RuntimeOperator`，其中封装了输入和输出的操作数，封装了参数和权重，维护了图的结构，除此之外还需要有算子的计算过程，可以将其抽象为Layer（这里将`Layer`称为算子，`RuntimeOperator`称为节点），Layer获取`RuntimeOperator`的输入操作数，基于Layer的派生类中的多态特性，对输出操作数进行计算，将结果放在输出操作数中，完成算子的计算过程。

### 设计

#### 层次设计

##### 算子基类`Layer`

```cpp
template <typename T>
class Layer{
public:
	virtual StatusCode Forward();
    virtual StatusCode Forward(
        const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    	std::vector<std::shared_ptr<Tensor<float>>>& outputs
    );
    
protected:
    std::string layer_name_;
    std::weak_ptr<RuntimeOperator> runtime_operator_; // 算子对应的节点，因为算子需要访问节点中保存的输入操作数
}
```

Layer是虚基类，只有Layer虚基类实现了不带参的Forward方法，每个实现的派生类算子都需要重写带参的Forward方法，实现各个算子的计算过程，不带参的Forward方法中会基于多态特性来调用带参的Forward方法

```cpp
# layer_input_datas 是当前节点输入Operands中的Tensors的数组，即算子的输入
# output_operand_datas 是指向当前节点输出Operand的指针，其datas成员是指向Tensors的数组
# 带参的Forward将输入进行计算，将结果放在输出中，因为函数传参传的是引用，所以会修改输出的值
StatusCode status = runtime_operator->layer->Forward(layer_input_datas, output_operand_datas->datas);
```

###### 不带参算子`NonParamLayer`

###### 带参算子`ParamLayer`

很多算子在初始化时需要一些参数（比如卷积的stride），这些参数封装在节点的attribute数据成员中，在初始化算子的过程中，需要使用入参节点的信息来进行初始化，初始化使用的方法可以进行复用，因此具体的带参算子可以继承自`ParamLayer`

##### 算子注册类`LayerRegisterer`

使用单例模式，在算子注册类`LayerRegisterer`中创建一个private的全局唯一的注册表`register`，这个注册表是一个map类型，key是算子类型（`std::string`类型），val是一个函数指针，所指向的函数完成一个算子的创建过程。在`LayerRegisterer::RegisterCreator`中，使用单例模式获取全局注册表，然后向全局注册表中添加{算子名称，算子创建过程的函数}，就向全局注册表中添加了算子。

这里详细介绍一下这个函数指针，`LayerRegisterer::Creator`就是一个函数指针，所指向的函数第一个参数是`const std::shared_ptr<RuntimeOperator>& op` ，表示从这个节点中读取相关信息（比如参数、weight、bias等）；第二个参数是`std::shared_ptr<Layer<float>>& layer`，表示一个待创建的算子（即`layer`传入时是指向空，而调用该函数完成后指向创建的节点）

然后还使用了工厂模式。单例模式确保了只有一个全局注册表实例，并且可以在代码的任何地方访问该注册表，并向注册表添加算子。在向注册表添加算子之后，工厂模式则负责根据算子的类型返回相应的算子实例。在`LayerRegisterer::CreateLayer`中，根据算子名字从注册表中得到创建算子的函数，有函数，还有节点中保存的相关信息，就可以初始化一个节点，返回一个算子实例。

##### 表达式类`ExpressionLayer`

算子与表达式的区别在于输入，算子的输入是一个操作数，这一个操作数在Forward传参时，将操作数中的datas属性（`std::vector<std::shared_ptr<Tensor<T>>>`类型）传入；而表达式的输入有两个（或多个）操作数，在Forward传参时，这两个输入操作数的datas都拼接放入到入参inputs（`std::vector<std::shared_ptr<Tensor<T>>>`类型）中。因此，虽然带参Forward函数形式是相同的，但是inputs参数中未必只有一个操作数的数据。

如何进行表达式的计算呢？大体过程与算子类似，但是其中多了几步，下面从头开始捋一下：

1. 与算子注册相同，表达式类`ExpressionLayer`一开始就添加到全局注册表中（见`layer/details/expression.cpp`）

2. 在前面根据PNNX的`Graph`构建KuiperInfer的`RuntimeGraph`的第三步中，在构建图结构的过程中需要为每个节点创建算子；表达式作为一种特殊的算子，只需要保存表达式字符串（比如`mul(@0,@1)`）这个属性即可，使用这个字符串构造表达式类`ExpressionLayer`的一个实例，添加到节点的`layer`属性中

3. 在表达式运行时，大致过程与算子类似，都是调用各自重写的带参Forward方法，不同的是算子类的输入inputs只有一个Tensors，表达式类的输入inputs有多个Tensors。在具体执行的过程逻辑中，需要根据表达式的含义，对输入inputs中多个Tensors进行相应运算，写回到输出Outputs中。（见`expression.cpp`中的`ExpressionLayer::Forward`函数）

    > 到底如何根据字符串表达式，对多个Tensors进行运算呢？这里举个例子：比如`std::vector<std::shared_ptr<Tensor<T>>> a`和`std::vector<std::shared_ptr<Tensor<T>>> b`相加
    >
    > - 首先对表达式进行词法解析，将字符串分成一个一个的token（`ExpressionParser::Tokenizer`）
    > - 然后对这些token进行语法解析，转换成一棵语法树（`ExpressionParser::Generate`中调用`ExpressionParser::Generate_`）
    > - 输出语法树的后缀表达式，即表达式的逆波兰表达式（在`ExpressionParser::Generate`中调用`ReversePolish`）
    > - 使用栈结构，遇到数据类型的token就入栈，遇到运算符号类型的出栈两次，计算完再入栈
    >
    > 即抽象表达式->词法解析->语法解析->语法树后序遍历得到逆波兰表达式->用栈计算，本来应该这样计算的，但是可以进行一些优化（这也是代码中实际上的过程）：
    >
    > - 词法解析，但是注意表达式的形式（`mul(@1,add(@2,@3))`，而不是`1*(2+3)`），词法解析后，tokens中的token，是表达式的前缀遍历：`mul ( 1 , add ( 2 , 3 ) )` 
    > - 在遍历过程中，逆序遍历，同样栈计算
    >
    > 参考：https://github.com/zjhellofss/KuiperInfer/issues/33#issuecomment-1718600527

，首先在表达式外部，将a和b都添加拼接到一个`std::vector<std::shared_ptr<Tensor<T>>> inputs`中。然后由于之前已经注册过表达式类`ExpressionLayer`（与算子注册相同，都是添加到全局注册表中），而且构建KuiperInfer图结构时，已经为节点添加过算子

##### Overview

整体算子结构的[UML类图](http://www.plantuml.com/plantuml/dsvg/xLTVRzDM57_tfxZI94X7WWHxix88w62gDlw88l4ONU9Ri71iw_gwq6MPiee0XV27KOP6KGSF7TK4j1qJj6bBlqok4vxu2ZlsSHmxJgC6RJpjpRtdtDyvlz_vSviZZg1Sk6M3V4zd69yKt2q9bpD5sK_Qhn_BL_SxlpsHlpVRQvjAoQ2EWtuLXP03f48lE8BJagYI4nQ_GhcM6ICgcHHKkCwufR7Tl7JJTeMJ9POh_8_KfI-8uKSfchJCYc1qXAQg0AAR5mChTqsXWyco6QV2uf7F5KOl5st1ysVHutJeK52gYca9HWms9OWrXInKDGK4yAtrkvsL9IGlLpPaSxxzv5hdrcnxTaDXfaO275yVx_4pUMyM7KDnJbdc6SffEO0dbQgIF3XuyCS2XevdOn93DoyJ5ItvASEYeX04U3hN7o2E7aXnXUaNBmadKc2QbO177XMxH3dJ0ZtECTEnkjY5Gd3rWav7lbTLFfqE4l7UBClhZRv-Eiikweu8ILMcT6PeLP8ZuvnHmT38PyZNc6gPslbOHsJH3OfC4fShqDf2BrK93-y3rpx4BvCtGfn4DsR6CYNBT0H9AfFki7tPb4jSWCA-CPo6RbRT739ZWzDAkJ8UvXTF22M0s7WMUkppCQSjNUi3mZ11Mdjb1KZ3UIKZuDoA6PivFh5qRs72VYCb2v_FvS9MIWzgvAMglLf2N5bVbcirER_IMgzXMtMUByTdT1XIA9KFSsFRUxJgDMvy-ENcnwr6--LBm6kj_mcpX-Vo_WES_z3QRFHPG__tTNbpGMusOLzWh-yrbrulG3eVjsuYZDnu-t5hhhCpBnSMvPr7hVUt8xozm34Ue_Xq6SMAP6gjl_BURSfRpp1tp1SIznQlGMZlrmsvTG_eoZjly0gwZKOhrcx1Ncssjfx9-bAqJf7cQ9Yn1WFCK9mqjRaQBC-e3h1jvEBjU8W1kuVnyvpQDYwXU4-hk_ZX22MDiSw2eKmFwfXI9SfWE1o1k4c1hNcM6MxY1V0FSFsogm8OzJywcFnUNzg8v85mEKoOS6AMMCQoipZDVNVsn-6Hu0WF28WoYdSgzf24_wqrdGnP3TkHmpCRz6Ealwec4r18O1iuIJZMC4poZzdMzgFMzavxVXc703nREy-zdrTm6k9p1Wi06npMGDIyYPw0LWSW7mAkDHVakoLPVuVxp1_UrJMvUbtMttfp3sK3bjCg0p743DnIy8lGaj2dGL3BJ3ozHwFZ4wvPwZwyK_3jd-NzfoKpLHH1jOeEY9WbYJ_7_6VXg6qRCvEJntjWcTxdEPXWaBelaaBY1s6J2N3Dag6TJFh-lulnKjLq7i7ZyNgnVNUrrRpUkRqaD_xoTchUtvie7UW9ChQs7tfpBphF6nzgcvtt3te5Y6cMonLw3T7TAYVfrJEMPS2uVxOv_NzE2SaHcseQg5u4YwkRugl3HK7GWEKeEleFB6trLxFMTrxrT--uPKudTkynwZ3_q39Sl_p9ydx6sfBuZweS6RVA0CnE1_YPdw9__eRwJO8MWI_gR5ETvxlyiv4I-4lXgAOTqwcJW8RZziAwhB-HQujUxOMl68gqgq03rQi49PXm3bqzz1tZ9ZDEMMUmZ1Zj0fGENKnTJ9GFZrl6-U0dJoQUop0DFfruBk3oxV6J9zFiqOO9eONoCeKTMQQYfyKth3IQ6lyX5ym5qZqyKKxpGJko7SJd22Jjber_0000)及[PlantUML代码](MyKuiperInfer/notes/uml/Layer_Structure.wsd)

![](http://www.plantuml.com/plantuml/dsvg/xLTVRzDM57_tfxZI94X7WWHxix88w62gDlw88l4ONU9Ri71iw_gwq6MPiee0XV27KOP6KGSF7TK4j1qJj6bBlqok4vxu2ZlsSHmxJgC6RJpjpRtdtDyvlz_vSviZZg1Sk6M3V4zd69yKt2q9bpD5sK_Qhn_BL_SxlpsHlpVRQvjAoQ2EWtuLXP03f48lE8BJagYI4nQ_GhcM6ICgcHHKkCwufR7Tl7JJTeMJ9POh_8_KfI-8uKSfchJCYc1qXAQg0AAR5mChTqsXWyco6QV2uf7F5KOl5st1ysVHutJeK52gYca9HWms9OWrXInKDGK4yAtrkvsL9IGlLpPaSxxzv5hdrcnxTaDXfaO275yVx_4pUMyM7KDnJbdc6SffEO0dbQgIF3XuyCS2XevdOn93DoyJ5ItvASEYeX04U3hN7o2E7aXnXUaNBmadKc2QbO177XMxH3dJ0ZtECTEnkjY5Gd3rWav7lbTLFfqE4l7UBClhZRv-Eiikweu8ILMcT6PeLP8ZuvnHmT38PyZNc6gPslbOHsJH3OfC4fShqDf2BrK93-y3rpx4BvCtGfn4DsR6CYNBT0H9AfFki7tPb4jSWCA-CPo6RbRT739ZWzDAkJ8UvXTF22M0s7WMUkppCQSjNUi3mZ11Mdjb1KZ3UIKZuDoA6PivFh5qRs72VYCb2v_FvS9MIWzgvAMglLf2N5bVbcirER_IMgzXMtMUByTdT1XIA9KFSsFRUxJgDMvy-ENcnwr6--LBm6kj_mcpX-Vo_WES_z3QRFHPG__tTNbpGMusOLzWh-yrbrulG3eVjsuYZDnu-t5hhhCpBnSMvPr7hVUt8xozm34Ue_Xq6SMAP6gjl_BURSfRpp1tp1SIznQlGMZlrmsvTG_eoZjly0gwZKOhrcx1Ncssjfx9-bAqJf7cQ9Yn1WFCK9mqjRaQBC-e3h1jvEBjU8W1kuVnyvpQDYwXU4-hk_ZX22MDiSw2eKmFwfXI9SfWE1o1k4c1hNcM6MxY1V0FSFsogm8OzJywcFnUNzg8v85mEKoOS6AMMCQoipZDVNVsn-6Hu0WF28WoYdSgzf24_wqrdGnP3TkHmpCRz6Ealwec4r18O1iuIJZMC4poZzdMzgFMzavxVXc703nREy-zdrTm6k9p1Wi06npMGDIyYPw0LWSW7mAkDHVakoLPVuVxp1_UrJMvUbtMttfp3sK3bjCg0p743DnIy8lGaj2dGL3BJ3ozHwFZ4wvPwZwyK_3jd-NzfoKpLHH1jOeEY9WbYJ_7_6VXg6qRCvEJntjWcTxdEPXWaBelaaBY1s6J2N3Dag6TJFh-lulnKjLq7i7ZyNgnVNUrrRpUkRqaD_xoTchUtvie7UW9ChQs7tfpBphF6nzgcvtt3te5Y6cMonLw3T7TAYVfrJEMPS2uVxOv_NzE2SaHcseQg5u4YwkRugl3HK7GWEKeEleFB6trLxFMTrxrT--uPKudTkynwZ3_q39Sl_p9ydx6sfBuZweS6RVA0CnE1_YPdw9__eRwJO8MWI_gR5ETvxlyiv4I-4lXgAOTqwcJW8RZziAwhB-HQujUxOMl68gqgq03rQi49PXm3bqzz1tZ9ZDEMMUmZ1Zj0fGENKnTJ9GFZrl6-U0dJoQUop0DFfruBk3oxV6J9zFiqOO9eONoCeKTMQQYfyKth3IQ6lyX5ym5qZqyKKxpGJko7SJd22Jjber_0000)

整个推理框架的总体结构Overview的[UML类图](http://www.plantuml.com/plantuml/svg/xLfjRnJNzN-_d-9AVobP9zuWpBiH_rN08THA80YeKgcgHkETQ--8tPd9F1WxxbOkn823JQ2Oqc1aJ5jgMWLiIXJC-i5VPcTsUSLNwBdtpDwPkpFZPQqch_h6dhddtD-vv-6US-wT_TZrDCVpwpNocM_Or1ap9wYZNl0Sl-Bv3YM5zycvccPwlpxpUMVZMVZotjkT1rNFirtbo14R26pgHnMhVkJhQadNd4j7AZNDTKkweKqwMhq0quD7pVRZkV3UPhZqlBrzDrpzfi1vo4LgkfPp_6APp1O8AH5TypILA4HpwfeowezJ8762errMtQfcKvUHNKzNb2bQyM2kRvZUiH7LAnSQXGZrl6zwHfsUy3p76FSzoi4_99UfCLdrL2QZ1wDIrPmov-4oPAB0PtHlngP4wKhu1A0kmWY8bYIVqq1nwb479ATd2IfEPGOArxQ2_Qj90qMyNu0_D5C_gRaq4f4cCFEX71DmkwgveBmvIMAjN9Al5pTyBmF2W6cehjgUSrmugtoGBKYXGGhfHHAhL8hSNSZKXCmsUgQHZpxwVPba6CAqmASrEk4HooGrmxn4b2opJDGipIltDo_echQle4W6dh2SovgZguQfqscG6jcGqwgQgqPqe8nRLetaE4LOBsEufirG1ySbutyEuzpsdCloFTltL0QTzfe0b7Wjj4m6T6P4fUS2kAlMDRkB08z9OSFvk98b8sJ3J2oAunu6rjGFH7NfLx9n1i6osJPr1zLMREu8HHD9Q40WaRtABBEbROaxAgr0FrQnm7wCOW6PZCU1aSTa9hLKAcTJKtjQJ85zsjc4FVfZi7KrU34dxyPV6RfEpL4sT1GIHRRCtrOvr-z0TiQq0b1f9Y5FqiZX98taIHh9bNGC9953Y3gM9-fOLWOkPMgVPU7y6103WaVMCCkA-Q4d1WHFIecKfRfXL-XN1j26eWv8_1DPZ8gIJAi5GlX_nepZHAJ1OHBitWqMbjhFxmVFRhSVV1FSljNUNiVNuFPIUtspG27NCea3o8to5aZsgXIKgit07oga8rK0am6HHG87_2zw528mR5ETZCz4qhZk1PAh9xw6YmkWRUVv_flxpmTRLphzWp0oPYOKJIxATgmfY7G7btLeILWIG99BdIaOg153afX4FgXjEEreTbNQpJWKTR7ZXgdQcbTLK_MzX5KbYqeuUPB128fl6b_vb5S5spId5OLBAITPNK_pA9l0IHVO6vAxUgSQvZvbkepm60U70y3vqUjXvO-PXbSSYZefNafF-aPDvqDJbg4JFlOfDbn5nXHJ6k0yoI-iMg7Jf66iTfBuT4W9TaL9s1IaGwrgRVqhMDc3dT3PknlCF-byUwLzvVNRdSNE_lUjvWg-0ZKTxM-kxeP_sMZ_imbK4NK7oaP_v0g_timJdangsM9vi9yxU_OtuTfEiFCTqAIZcgrLBccJb7Fm2CV05ybxbgr7jBIyAOQ1Xc0KDMCsYq-SJzB7ax844vMomu3ajk7z-xnOpGY8vIBTh_0clHiq7qIaYcMwVfqwlJsvu8iqwkh2ubA098Qjv32CIs61J00k5iAbjvhhxUL7dOqd4HsyY7aQgCBJkLXYvOYrkzzUVYfX5OE_Fu206qfZ2WU97ayEYBJDyvs1MIlBUgREZISE9NsO4AZV-UXG4O0zVJwx27V9sI2QkrM0zHB4DD9BYLIB3Irg3Ph3pWytmOAPvm2RDRgXuS0b2OyDbAAGHEKf41oL0ZGzQBBad9U06ppRO1Xrld_0hePUR6CjnSuEJzgpsZWOLAju3P7ZMNvd1SvtHosTvd5nnxl3KPIugKd_XqTidWzJ1Hge3YvRj4o2vpBLBcKMB519iXogeIWf9C1lRTs2RONQ1nklW-Oox2rCrLBp8KmKldmQN5liCT5vEccutUkbsGc3rlG6bCnHXseUHQ8pG_u_Eq2H3TPDfCcph9CmAWtYHFEPuZBY1_AC24SoS76eo5A9H_fL-VU6Swuui5illpV47InMrLr3SIXpMMbMwaJh4miI9W3PE1Pnl2Vt5088nV1blQjMW7ImNh8434MM9bscM7wYQXHanyV7RxbdaBWK9VObo0tcdyB1EbZUQ6tDOLXr7Z-De-vWfCXK3ERwBciF5g-_-TCDw3RQF_m0UAmN-UuAZWUtxs14HzJcOWyruj-R3swi1DjDKQdUTIrnBC1otkxSG9ZWzKzlTnOw-yl1ocgmTA-rTrFWzOF3fuIA3v-YiOIPMblFmbVRmRThk7Pn0eB-24I7VtuDVHEe6oozn2d8DfXQYN3ZcQBPsbc3fb8a25HpCCn4W04c6YT3jOMvyE62y0Dacs3rPb94HkwXpfUGpssMXEIOLdBuS0lr4aLamQPCRzGXTfcS3IUsW23XcUt-6eGQwX4j-wnbyghBcTcTZZp0RgGGAFD68RbxcG0mLA-093gl8VYbmLNVvK9TP8OodpfkQ0FFQnGEhmxUjk7D6rfQkdEIZHVfMvLluqZ6l6hgvaqQIU6CzCSPIU6CP4n8B5PsSocrfkHzbon7khqIa1XXvsdDbxpF1b04dVPWpqW19xR7kxOdOXjz69MVCHD6p0ejMlOmrekUEUyduN4B3H0yiEq6uKx9pb_mVwqHa4Ksv8GNcrzHPFa5mccPgHhh1cJhu1zNMhltyBmZJioj_SVX7pSmtIOJELGOpA1GPqHsb482SYa00GIRrbm9NZq9vbzXmMJLOVD5i7ajcFyfNBWRB4BrswHW2sy6PXNmYK0hQqo1FIrJwkrEYE5FVRFIxUmcu9cD7MUzIt52zRYpL1TiE4oaro6MN4xOTctcmeLJCLWnVcIVxfYbPKa5aXo8Oim3rc5oO22JL4ySAwOewONmzkJvQllMPgjvhNFpIV3wt-7-NFZZDjeE30fsREtU3HVMEuyNtynjT_Rkn1w8iVc71n4H1V4afOYOlLlOpcZJvooh1XNdVnlv5zl8U1kgws1r5IZTx_W42UWErJM-fcagwmvqUHpFbQEMEIKbPtO7p0QjciySBhlt5_GjIHvLuUXarG6OuZLyyG_Y_74WtmMm1V9oEzkQ4V_wOLXO0jk84xf-qj1S2Hg6sojRmVpBuCLgEBSU_ruYrq2Prfl5wrdtwFJHpwXZqjfPwnow4QMngrXnWpjM7ncrQb_oVaYwkZtG1eTMlCHrENtgp9cqzaZ31N4A_if4cQXhNcoBJsXbC6lyZBhWMc2v8sFrj3v8Hsspz87j0DrzPsyFwbtmuWgwFjBoedM9cb7GU-nPnQiw95napvp6VncasXRi1J01UqZywA6H13lBiluipZBzUkyl9930QlOmrWJvd7nSO9I9m3qrRJlKTGsUX5rXQxxUL8_C0SenWmgR29doZjIOI3IFrgdfPNwCEKrDbjkeMeoqo3pUnqldEl9NSB8ZVvDWD06OFkldOhBkNZO47drGJXu9PwSEXxtSYY9Moqw76qlmuwxE-j_2rVppPHeYLefZF1COxFY__2YySHi7mxKxm10-d8EFYDX6lPcx5hxyQp2_AUAqvmDSCdvJGT6dLFVrhitNd_MjhLV7yizS74JOZBdTQbhaxzrqIWZR5PBgwEgKEy1i7uCz_7hjFm00)及[PlantUML代码](MyKuiperInfer/notes/uml/overview_Structure.wsd)

![](http://www.plantuml.com/plantuml/svg/xLfjRnJNzN-_d-9AVobP9zuWpBiH_rN08THA80YeKgcgHkETQ--8tPd9F1WxxbOkn823JQ2Oqc1aJ5jgMWLiIXJC-i5VPcTsUSLNwBdtpDwPkpFZPQqch_h6dhddtD-vv-6US-wT_TZrDCVpwpNocM_Or1ap9wYZNl0Sl-Bv3YM5zycvccPwlpxpUMVZMVZotjkT1rNFirtbo14R26pgHnMhVkJhQadNd4j7AZNDTKkweKqwMhq0quD7pVRZkV3UPhZqlBrzDrpzfi1vo4LgkfPp_6APp1O8AH5TypILA4HpwfeowezJ8762errMtQfcKvUHNKzNb2bQyM2kRvZUiH7LAnSQXGZrl6zwHfsUy3p76FSzoi4_99UfCLdrL2QZ1wDIrPmov-4oPAB0PtHlngP4wKhu1A0kmWY8bYIVqq1nwb479ATd2IfEPGOArxQ2_Qj90qMyNu0_D5C_gRaq4f4cCFEX71DmkwgveBmvIMAjN9Al5pTyBmF2W6cehjgUSrmugtoGBKYXGGhfHHAhL8hSNSZKXCmsUgQHZpxwVPba6CAqmASrEk4HooGrmxn4b2opJDGipIltDo_echQle4W6dh2SovgZguQfqscG6jcGqwgQgqPqe8nRLetaE4LOBsEufirG1ySbutyEuzpsdCloFTltL0QTzfe0b7Wjj4m6T6P4fUS2kAlMDRkB08z9OSFvk98b8sJ3J2oAunu6rjGFH7NfLx9n1i6osJPr1zLMREu8HHD9Q40WaRtABBEbROaxAgr0FrQnm7wCOW6PZCU1aSTa9hLKAcTJKtjQJ85zsjc4FVfZi7KrU34dxyPV6RfEpL4sT1GIHRRCtrOvr-z0TiQq0b1f9Y5FqiZX98taIHh9bNGC9953Y3gM9-fOLWOkPMgVPU7y6103WaVMCCkA-Q4d1WHFIecKfRfXL-XN1j26eWv8_1DPZ8gIJAi5GlX_nepZHAJ1OHBitWqMbjhFxmVFRhSVV1FSljNUNiVNuFPIUtspG27NCea3o8to5aZsgXIKgit07oga8rK0am6HHG87_2zw528mR5ETZCz4qhZk1PAh9xw6YmkWRUVv_flxpmTRLphzWp0oPYOKJIxATgmfY7G7btLeILWIG99BdIaOg153afX4FgXjEEreTbNQpJWKTR7ZXgdQcbTLK_MzX5KbYqeuUPB128fl6b_vb5S5spId5OLBAITPNK_pA9l0IHVO6vAxUgSQvZvbkepm60U70y3vqUjXvO-PXbSSYZefNafF-aPDvqDJbg4JFlOfDbn5nXHJ6k0yoI-iMg7Jf66iTfBuT4W9TaL9s1IaGwrgRVqhMDc3dT3PknlCF-byUwLzvVNRdSNE_lUjvWg-0ZKTxM-kxeP_sMZ_imbK4NK7oaP_v0g_timJdangsM9vi9yxU_OtuTfEiFCTqAIZcgrLBccJb7Fm2CV05ybxbgr7jBIyAOQ1Xc0KDMCsYq-SJzB7ax844vMomu3ajk7z-xnOpGY8vIBTh_0clHiq7qIaYcMwVfqwlJsvu8iqwkh2ubA098Qjv32CIs61J00k5iAbjvhhxUL7dOqd4HsyY7aQgCBJkLXYvOYrkzzUVYfX5OE_Fu206qfZ2WU97ayEYBJDyvs1MIlBUgREZISE9NsO4AZV-UXG4O0zVJwx27V9sI2QkrM0zHB4DD9BYLIB3Irg3Ph3pWytmOAPvm2RDRgXuS0b2OyDbAAGHEKf41oL0ZGzQBBad9U06ppRO1Xrld_0hePUR6CjnSuEJzgpsZWOLAju3P7ZMNvd1SvtHosTvd5nnxl3KPIugKd_XqTidWzJ1Hge3YvRj4o2vpBLBcKMB519iXogeIWf9C1lRTs2RONQ1nklW-Oox2rCrLBp8KmKldmQN5liCT5vEccutUkbsGc3rlG6bCnHXseUHQ8pG_u_Eq2H3TPDfCcph9CmAWtYHFEPuZBY1_AC24SoS76eo5A9H_fL-VU6Swuui5illpV47InMrLr3SIXpMMbMwaJh4miI9W3PE1Pnl2Vt5088nV1blQjMW7ImNh8434MM9bscM7wYQXHanyV7RxbdaBWK9VObo0tcdyB1EbZUQ6tDOLXr7Z-De-vWfCXK3ERwBciF5g-_-TCDw3RQF_m0UAmN-UuAZWUtxs14HzJcOWyruj-R3swi1DjDKQdUTIrnBC1otkxSG9ZWzKzlTnOw-yl1ocgmTA-rTrFWzOF3fuIA3v-YiOIPMblFmbVRmRThk7Pn0eB-24I7VtuDVHEe6oozn2d8DfXQYN3ZcQBPsbc3fb8a25HpCCn4W04c6YT3jOMvyE62y0Dacs3rPb94HkwXpfUGpssMXEIOLdBuS0lr4aLamQPCRzGXTfcS3IUsW23XcUt-6eGQwX4j-wnbyghBcTcTZZp0RgGGAFD68RbxcG0mLA-093gl8VYbmLNVvK9TP8OodpfkQ0FFQnGEhmxUjk7D6rfQkdEIZHVfMvLluqZ6l6hgvaqQIU6CzCSPIU6CP4n8B5PsSocrfkHzbon7khqIa1XXvsdDbxpF1b04dVPWpqW19xR7kxOdOXjz69MVCHD6p0ejMlOmrekUEUyduN4B3H0yiEq6uKx9pb_mVwqHa4Ksv8GNcrzHPFa5mccPgHhh1cJhu1zNMhltyBmZJioj_SVX7pSmtIOJELGOpA1GPqHsb482SYa00GIRrbm9NZq9vbzXmMJLOVD5i7ajcFyfNBWRB4BrswHW2sy6PXNmYK0hQqo1FIrJwkrEYE5FVRFIxUmcu9cD7MUzIt52zRYpL1TiE4oaro6MN4xOTctcmeLJCLWnVcIVxfYbPKa5aXo8Oim3rc5oO22JL4ySAwOewONmzkJvQllMPgjvhNFpIV3wt-7-NFZZDjeE30fsREtU3HVMEuyNtynjT_Rkn1w8iVc71n4H1V4afOYOlLlOpcZJvooh1XNdVnlv5zl8U1kgws1r5IZTx_W42UWErJM-fcagwmvqUHpFbQEMEIKbPtO7p0QjciySBhlt5_GjIHvLuUXarG6OuZLyyG_Y_74WtmMm1V9oEzkQ4V_wOLXO0jk84xf-qj1S2Hg6sojRmVpBuCLgEBSU_ruYrq2Prfl5wrdtwFJHpwXZqjfPwnow4QMngrXnWpjM7ncrQb_oVaYwkZtG1eTMlCHrENtgp9cqzaZ31N4A_if4cQXhNcoBJsXbC6lyZBhWMc2v8sFrj3v8Hsspz87j0DrzPsyFwbtmuWgwFjBoedM9cb7GU-nPnQiw95napvp6VncasXRi1J01UqZywA6H13lBiluipZBzUkyl9930QlOmrWJvd7nSO9I9m3qrRJlKTGsUX5rXQxxUL8_C0SenWmgR29doZjIOI3IFrgdfPNwCEKrDbjkeMeoqo3pUnqldEl9NSB8ZVvDWD06OFkldOhBkNZO47drGJXu9PwSEXxtSYY9Moqw76qlmuwxE-j_2rVppPHeYLefZF1COxFY__2YySHi7mxKxm10-d8EFYDX6lPcx5hxyQp2_AUAqvmDSCdvJGT6dLFVrhitNd_MjhLV7yizS74JOZBdTQbhaxzrqIWZR5PBgwEgKEy1i7uCz_7hjFm00)

### 算子开发流程

1. 写算子
    - 根据是否含参数，继承`NonParamLayer`或者`ParamLayer`，因为如果含参数，设置weight和bias的过程是可以复用的
    - 在具体算子类中，必须实现两个函数
        - 带参的Forward函数，是算子执行的具体逻辑，输入Tensors在计算之后，写入到输出Tensors中。函数签名为：`StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,std::vector<std::shared_ptr<Tensor<float>>>& outputs) override`
        - 根据节点的信息（比如参数和权重），创建算子的函数，使用时经常作为函数指针传入到注册函数中。函数签名为：`static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer<float>>& layer);`
2. 注册算子
    - 在`LayerRegisterer::RegisterCreator`中，使用单例模式获取全局注册表，然后向全局注册表中添加{算子名称，算子创建过程的函数}
    - 在对应算子的实现文件中，重写完算子的Forward函数之后，顺便将其注册。
        - 比如relu算子在`relu.cpp`中重写完Forward函数之后，紧接着进行了注册：`LayerRegistererWrapper kReluCreateInstance(ReluLayer::CreateInstance, "nn.ReLU");`
3. 创建算子实例
    - 在`LayerRegisterer::CreateLayer`中，因为算子已经注册到全局注册表，所以可以得到该创建该算子的函数（拿到了函数指针），根据节点中的信息（比如参数、权重等），创建一个算子并返回该算子

### 计算图的执行过程

在`RuntimeGraph::Forward(bool)`中，节点按照逆拓扑顺序进行遍历（此时`RuntimeGraph::operators_`中节点已经按照逆拓扑顺序排好）

- 每个节点调用其指向算子的`Forward()`方法，执行算子的计算过程，得到输出操作数，这个过程在`runtime_ir.cpp`中的`ExecuteLayer`函数中

    - 算子的计算过程：当前节点op调用其算子的`Forward()`方法，此时进入了Layer虚基类的`Forward()`方法，首先从节点op中得到输入操作数和输出操作数，然后因为算子与节点关联，所以在Layer类中有指向op的指针，`op->layer`是指向Layer虚基类的指针，但是由于多态特性，此时`op->layer`的动态类型是指向特定算子的指针，因此调用带参的`Forward()`方法，就进入了具体算子的计算过程

- 将当前节点的输出，传播到当前节点后继节点的输入中，对应函数是`RuntimeGraph::PropagateLayerOutputs`，这个函数很重要，其中数据结构比较复杂，而且进行的只是指针的修改，而没有真的将前一个节点的输出复制到后一个节点的输入

    ```cpp
    template <typename T>
    void RuntimeGraph::PropagateLayerOutputs(
        const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
        const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_datas) {
        for (const auto& [_, output_op] : current_op->output_operators_map) { // current_op的后继节点
    
            // 对于当前节点的每一个后继节点output_op，得到其输入Operands的map
            const auto& next_input_operands_map = output_op->input_operands_map; 
            const auto& next_input_op_iter = next_input_operands_map.find(current_op->name);
            // 在后继节点的输入Operands的map中，找到了当前节点的名字
            if (next_input_op_iter != next_input_operands_map.end()) {
                // 后继节点的输入Operand中保存的Tensors
                std::vector<tensor_sptr<T>>& next_input_datas = next_input_op_iter->second->datas;
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
    ```

- 重复上述过程



