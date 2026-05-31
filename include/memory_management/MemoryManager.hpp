#pragma once

#include <unordered_map>

#include "core/tensor.hpp"
#include "memory_management/MemoryManagerUtils.hpp"


class MemoryManager {
    
    public:
        MemoryManager(const SupportedTensorGroups& supportedGroups);
        bool allocateTensors(const TensorInfoMap& tensorInfos, const TensorGroup& group);
        TensorViewMap getTensorViewsFromGroup(TensorGroup group);

    private:

        template <typename TensorMapType>
        void allocateTensors(TensorInfoMap& tensorInfos, TensorMapType& tensorMap) {

            using MapType = std::remove_reference_t<TensorMapType>;
            using TensorType = typename MapType::mapped_type;

            for(const auto& [name, info] : tensorInfos){

                auto [it, inserted] = tensorMap.emplace(
                name,
                TensorType(info.dtype, info.shape, info.mode)
                );

                if (!inserted) {
                    throw std::runtime_error("Duplicate tensor name encountered when allocating memory: " + std::string(name));
                }
            }
        }

        TensorMap m_hostInputs, m_hostOutputs;
        PinnedTensorMap m_pinnedHostInputs, m_pinnedHostOutputs;
        CudaTensorMap m_deviceInputs, m_deviceOutputs;
        UnifiedMemoryTensorMap m_unifiedMemInputs, m_unifiedMemOutputs;

        CudaTensorMap m_devicePostProcessOutputs;
        TensorMap m_hostPostProcessOutputs;

        SupportedTensorGroups m_supportedGroups;
};