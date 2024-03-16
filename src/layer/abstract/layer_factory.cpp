#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {
LayerRegisterer::CreateRegistry* LayerRegisterer::registry_ = nullptr;

void LayerRegisterer::RegisterCreator(const std::string& layer_type, const Creator& creator) {
    CHECK(!layer_type.empty());
    CHECK(creator != nullptr);
    CreateRegistry* registry = Registry();
    CHECK_EQ(registry->count(layer_type), 0)
        << "Layer type: " << layer_type << " has already registered!";
    registry->insert({ layer_type, creator });
}

LayerRegisterer::CreateRegistry* LayerRegisterer::Registry() {
    if (registry_ == nullptr) {
        registry_ = new CreateRegistry();
        static RegistryGarbageCollector c;
    }

    CHECK(registry_ != nullptr) << "Global layer register init failed!";
    return registry_;
}

std::shared_ptr<Layer<float>> LayerRegisterer::CreateLayer(
    const std::shared_ptr<RuntimeOperator>& op) {
    CreateRegistry* registry = Registry();
    const std::string& layer_type = op->type;
    LOG_IF(FATAL, registry->count(layer_type) <= 0) << "Can not find the layer type: " << layer_type;
    const auto& creator = registry->find(layer_type)->second; // 函数指针

    LOG_IF(FATAL, !creator) << "Layer creator is empty!";
    std::shared_ptr<Layer<float>> layer;
    const auto& status = creator(op, layer);
    LOG_IF(FATAL, status != StatusCode::kSuccess)
        << "Create the layer: " << layer_type << " failed, error code: " << int32_t(status);
    return layer;
}

std::vector<std::string> LayerRegisterer::layer_types() {
    std::set<std::string> layer_types_unique;
    CreateRegistry* registry = Registry();
    for (const auto& [layer_type, _] : *registry) {
        layer_types_unique.insert(layer_type);
    }
    std::vector<std::string> layer_types(layer_types_unique.begin(), layer_types_unique.end());
    return layer_types;
}
}  // namespace kuiper_infer
