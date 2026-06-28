#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "memory_management/enums.hpp"
#include "memory_management/utils.hpp"

/**
 * @brief Owns tensor storage and exposes pipeline-stage tensor views.
 *
 * The memory manager allocates configured tensor groups, creates stage-specific
 * contexts, and performs copies or unified-memory prefetches between groups.
 */
class MemoryManager {
    public:

        /**
         * @brief Construct with pipeline tensor group configuration.
         * @param groupLists Preprocessing, inference, and postprocessing groups.
         */
        explicit MemoryManager(TensorGroupConfig groupLists);

        /**
         * @brief Allocate one tensor group from tensor specifications.
         * @param tensorSpecs Tensor specifications keyed by model tensor name.
         * @param group Group to allocate.
         */
        void allocateTensorsForGroup(const TensorSpecMap& tensorSpecs, TensorGroup group);

        /**
         * @brief Allocate all configured groups matching the tensor specs' IO mode.
         * @param tensorSpecs Input or output tensor specifications.
         */
        void allocateAllTensors(const TensorSpecMap& tensorSpecs);

        /**
         * @brief Create the tensor contexts consumed by the pipeline.
         * @return Context containing preprocessing, inference, postprocessing, and transfer views.
         */
        PipelineTensorContext createPipelineTensorContext();

        /**
         * @brief Execute tensor transfers or readiness operations.
         * @param transfers Transfer descriptors.
         * @param TensorKeys Tensor names to copy or prefetch.
         * @param stream CUDA stream for asynchronous operations.
         */
        static void transferTensors(
            const std::vector<TensorTransfer>& transfers,
            const std::vector<std::string>& TensorKeys,
            cudaStream_t stream
        );

        /**
         * @brief Access mutable tensor views for an allocated group.
         */
        TensorViewMap& getTensorViewsFromGroup(TensorGroup group);

        /**
         * @brief Access const tensor views for an allocated group.
         */
        const TensorViewMap& getTensorViewsFromGroup(TensorGroup group) const;

    private:
        template <typename TensorMapType>
        void allocateBuffers(TensorMapType& tensorMap, const TensorSpecMap& tensorSpecs) {
            using TensorType = typename TensorMapType::mapped_type;

            for (const auto& [name, info] : tensorSpecs) {
                auto [it, inserted] = tensorMap.emplace(
                    name,
                    TensorType(info.dtype, info.shape, info.mode)
                );

                if (!inserted) {
                    throw std::runtime_error(
                        "Duplicate tensor name encountered when allocating memory: " + name
                    );
                }
            }
        }

        bool supportsGroup(TensorGroup group) const;
        bool hasAllocatedGroup(TensorGroup group) const;
        TensorGroup selectFirstAllocatedGroup(const TensorGroupList& groups, const std::string& label) const;
        TensorGroup selectInferenceInputGroup() const;
        TensorGroup selectInferenceOutputGroup() const;
        std::vector<TensorGroup> allConfiguredGroups() const;
        std::vector<TensorTransfer> makeTransfers(TensorGroup sourceGroup, TensorGroup targetGroup);
        DeviceType getTensorViewDevice(const TensorViewMap& tensorMap) const;
        void appendUnifiedReadiness(
            std::vector<TensorTransfer>& transfers,
            TensorGroup group,
            TensorViewMap& bufferViews,
            DeviceType targetDevice
        );

        HostTensorMap m_hostInputs;
        PinnedHostTensorMap m_pinnedInputs;
        CudaTensorMap m_deviceInputs;
        UnifiedTensorMap m_unifiedInputs;

        HostTensorMap m_hostOutputs;
        PinnedHostTensorMap m_pinnedOutputs;
        CudaTensorMap m_deviceOutputs;
        UnifiedTensorMap m_unifiedOutputs;

        HostTensorMap m_hostPostProcessOutputs;
        CudaTensorMap m_devicePostProcessOutputs;

        std::unordered_map<TensorGroup, TensorViewMap> m_TensorViewsByGroup;
        TensorGroupConfig m_groupLists;
};
