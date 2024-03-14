#ifndef KUIPER_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
#define KUIPER_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace kuiper_infer {

namespace utils {

using Time = std::chrono::steady_clock;

/**
 * @brief Timing state for a layer type
 *
 * Contains timing information for a specific layer, including the
 * execution duration, layer name, and type.
 *
 * Used to collect per layer execution time statistics. Thread safe.
 */
struct LayerTimeState {
    /**
     * @brief Construct and start time logging
     *
     * Starts timing for the layer execution.
     *
     * Records the start time and initializes layer name and type.
     *
     * @param layer_name Name of the layer
     * @param layer_type Type of the layer
     */
    explicit LayerTimeState(long duration_time, std::string layer_name, std::string layer_type)
        : duration_time_(duration_time),
        layer_name_(std::move(layer_name)),
        layer_type_(std::move(layer_type)) {
    }

    /// Duration in ms
    long duration_time_;

    /**
     * @brief Mutex for thread safety
     *
     * Locks access to duration_time_ across threads.
     */
    std::mutex time_mutex_;

    /// Name of the layer
    std::string layer_name_;

    /// Type of the layer
    std::string layer_type_;
};

// map layer name to layerTimeState
using LayerTimeStatesCollector = std::map<std::string, std::shared_ptr<LayerTimeState>>; 

using PtrLayerTimeStatesCollector = std::shared_ptr<LayerTimeStatesCollector>;

/**
 * @brief Singleton for layer timing stats collector
 *
 * Implements a singleton pattern to provide thread-safe access
 * to a unique instance of the layer timing collector.
 *
 * The PtrLayerTimeStatesCollector object containing the map of
 * layer timing information can be accessed via the singleton.
 */
class LayerTimeStatesSingleton {
public:
    LayerTimeStatesSingleton() = default;

    /**
     * @brief Initializes the singleton instance
     *
     * Creates the singleton collector instance.
     */
    static void LayerTimeStatesCollectorInit();

    /**
     * @brief Gets the singleton instance
     *
     * @return The singleton collector instance
     */
    static PtrLayerTimeStatesCollector SingletonInstance();

private:
    static std::mutex mutex_;
    static PtrLayerTimeStatesCollector time_states_collector_;
};

/**
 * @brief Logs execution time for a layer
 *
 * Helper class to log the start and end time when executing a layer.
 * Computes the layer execution duration and collects the timing
 * statistics per layer type via a singleton.
 */
class LayerTimeLogging {
public:
    /**
     * @brief Construct and start time logging
     *
     * Starts timing for the layer execution.
     *
     * Records the start time and initializes layer name and type.
     *
     * @param layer_name Name of the layer
     * @param layer_type Type of the layer
     */
    explicit LayerTimeLogging(std::string layer_name, std::string layer_type);

    /**
     * @brief Stop time log and record duration
     *
     * Records execution duration to singleton collector.
     */
    ~LayerTimeLogging();

    /**
     * @brief Prints summary of layer times
     *
     * Prints collected per layer type execution times.
     */
    static void SummaryLogging();

private:
    std::string layer_name_;
    std::string layer_type_;
    std::chrono::steady_clock::time_point start_time_;
};
}  // namespace utils
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
