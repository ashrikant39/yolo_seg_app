#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "memory_management/enums.hpp"
#include "memory_management/utils.hpp"

class MemoryManager {
    public:

        explicit MemoryManager(GroupLists groupLists);

        void allocateTensorGroup(const TensorInfoMap& tensorInfos, TensorGroupType group);
        void allocateAllGroups(const TensorInfoMap& tensorInfos);

        PipelineTensorContext createPipelineTensorContext();

        static void transferTensors(
            const std::vector<TensorTransfer>& transfers,
            const std::vector<std::string>& tensorKeys,
            cudaStream_t stream
        );

        TensorViewMap& getTensorViewsFromGroup(TensorGroupType group);
        const TensorViewMap& getTensorViewsFromGroup(TensorGroupType group) const;

    private:
        template <typename TensorMapType>
        void allocateTensors(TensorMapType& tensorMap, const TensorInfoMap& tensorInfos) {
            using TensorType = typename TensorMapType::mapped_type;

            for (const auto& [name, info] : tensorInfos) {
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

        bool supportsGroup(TensorGroupType group) const;
        bool hasAllocatedGroup(TensorGroupType group) const;
        TensorGroupType selectFirstAllocatedGroup(const TensorGroupList& groups, const std::string& label) const;
        TensorGroupType selectInferenceInputGroup() const;
        TensorGroupType selectInferenceOutputGroup() const;
        std::vector<TensorGroupType> allConfiguredGroups() const;
        std::vector<TensorTransfer> makeTransfers(TensorGroupType sourceGroup, TensorGroupType targetGroup);
        DeviceType getTensorMapDevice(const TensorViewMap& tensorMap) const;
        void appendUnifiedReadiness(
            std::vector<TensorTransfer>& transfers,
            TensorGroupType group,
            TensorViewMap& tensors,
            DeviceType targetDevice
        );

        HostTensorMap m_hostInputs;
        PinnedTensorMap m_pinnedInputs;
        DeviceTensorMap m_deviceInputs;
        UnifiedMemoryTensorMap m_unifiedInputs;

        HostTensorMap m_hostOutputs;
        PinnedTensorMap m_pinnedOutputs;
        DeviceTensorMap m_deviceOutputs;
        UnifiedMemoryTensorMap m_unifiedOutputs;

        HostTensorMap m_hostPostProcessOutputs;
        DeviceTensorMap m_devicePostProcessOutputs;

        std::unordered_map<TensorGroupType, TensorViewMap> m_tensorViewMaps;
        GroupLists m_groupLists;
};
