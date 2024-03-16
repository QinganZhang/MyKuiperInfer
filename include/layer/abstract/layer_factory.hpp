#ifndef KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#include <map>
#include <memory>
#include <string>
#include "layer.hpp"
#include "runtime/runtime_operator.hpp"

namespace kuiper_infer {

/**
 * @brief 算子注册类
*/
class LayerRegisterer {
private:
    /**
     * @brief 函数指针，所指向的函数第一个参数是节点（包含类型、参数、权重等信息），
     * 第二个参数是一个待创建的算子（即指向算子的空指针）
     * 这个函数指针指向的函数表示一个算子的创建过程，
    */
    typedef StatusCode(*Creator)(const std::shared_ptr<RuntimeOperator>& op,
        std::shared_ptr<Layer<float>>& layer);
    
    /**
     * @brief 注册表类型
    */
    typedef std::map<std::string, Creator> CreateRegistry;

    /**
     * @brief 全局唯一的注册表，key是算子的类型，value是算子的初始化过程
    */
    static CreateRegistry* registry_;
    // static std::map<std::string, Creator>* registry_; // 上面的等价定义

public:
    friend class LayerRegistererWrapper;
    friend class RegistryGarbageCollector;

    /**
     * @brief Registers a layer creator function
     *
     * Registers a layer creation function for the given layer type.
     * The creation function generates a layer instance from runtime operator.
     * 将字符串layer_type和函数指针插入到注册表registry中，在全局注册表中添加一个算子的构建方式，即向推理框架添加了一个算子
     *
     * @param layer_type The name of the layer type
     * @param creator Function to create the layer
     */
    static void RegisterCreator(const std::string& layer_type, const Creator& creator);

    /**
     * @brief Creates a layer object
     *
     * Calls the registered creator function for the given runtime operator
     * to create a layer instance.
     * 创建算子，因为节点中已经存在相关的信息了，而且注册表中也有对应的算子初始化方式（函数指针）
     * 为节点创建算子即为使用节点中相关信息为算子赋值
     * @param op The runtime operator
     * @return A shared pointer to the created layer object
     */
    static std::shared_ptr<Layer<float>> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);

    /**
     * @brief Gets the layer registry，单例模式
     *
     * @return Pointer to the layer creator registry
     */
    static CreateRegistry* Registry();

    /**
     * @brief Gets registered layer types
     * 返回注册表中算子的类型
     *
     * @return A vector of registered layer type names
     */
    static std::vector<std::string> layer_types(); 
};

/**
 * @brief Layer registry wrapper
 *
 * Helper class to register a layer creator function.
 * Automatically calls LayerRegisterer::RegisterCreator.
 */
class LayerRegistererWrapper {
public:
    // LayerRegisterer::RegisterCreator(layer_type, creator)
    explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator,
        const std::string& layer_type) {
        LayerRegisterer::RegisterCreator(layer_type, creator);
    }

    /**
     * @brief 
    */
    template <typename... Ts>
    explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator,
        const std::string& layer_type, const Ts&... other_layer_types)
        : LayerRegistererWrapper(creator, other_layer_types...) {
        LayerRegisterer::RegisterCreator(layer_type, creator);
    }

    explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator) {}
};

/**
 * @brief Garbage collector for layer registry
 *
 * Destructor that cleans up the LayerRegisterer registry
 * when program exits. Deletes registry pointer if allocated.
 */
class RegistryGarbageCollector {
public:
    ~RegistryGarbageCollector() {
        if (LayerRegisterer::registry_ != nullptr) {
            delete LayerRegisterer::registry_;
            LayerRegisterer::registry_ = nullptr;
        }
    }
    friend class LayerRegisterer;

private:
    RegistryGarbageCollector() = default;
    RegistryGarbageCollector(const RegistryGarbageCollector&) = default;
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
