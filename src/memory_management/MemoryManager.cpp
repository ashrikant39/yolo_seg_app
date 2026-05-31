#include "memory_management/MemoryManager.hpp"

MemoryManager::MemoryManager(const SupportedTensorGroups& supportedGroups):
    m_supportedGroups(supportedGroups) {}

MemoryManager::allocateTensors(const TensorInfoMap& tensorInfos, const TensorGroup& group) {
    
    auto it = std::find(m_supportedGroups.begin(), m_supportedGroups.end(), group);

    if (it == m_supportedGroups.end()) {
        throw std::runtime_error("Unsupported Tensor Group");
    }

    switch (group) {

        case TensorGroup::HostInput:
            allocateTensors(m_hostInputs, tensorInfos);
            break;
        
        case TensorGroup::PinnedInput:
            allocateTensors(m_pinnedHostInputs, tensorInfos);
            break;
        
        case TensorGroup::DeviceInput:
            allocateTensors(m_deviceInputs, tensorInfos);
            break;

        case TensorGroup::UnifiedInput:
            allocateTensors(m_unifiedMemInputs, tensorInfos);
            break;

        case TensorGroup::HostOutput:
            allocateTensors(m_hostOutputs, tensorInfos);
            break;

        case TensorGroup::PinnedOutput:
            allocateTensors(m_pinnedHostOutputs, tensorInfos);
            break;

        case TensorGroup::DeviceOutput:
            allocateTensors(m_deviceOutputs, tensorInfos);
            break;

        case TensorGroup::UnifiedOutput:
            allocateTensors(m_unifiedMemOutputs, tensorInfos);
            break;

        case TensorGroup::HostPostProcessOutput:
            allocateTensors(m_hostPostProcessOutputs);
        
        case TensorGroup::DevicePostProcessOutput:
            allocateTensors(m_devicePostProcessOutputs);

        default:
            throw std::runtime_error("Invalid Tensor Group");
            break;
    }
}


TensorViewMap MemoryManager::getTensorViewsFromGroup(TensorGroup group) {

    auto it = std::find(m_supportedGroups.begin(), m_supportedGroups.end(), group);

    if (it == m_supportedGroups.end()) {
        throw std::runtime_error("Unsupported Tensor Group");
    }

    switch (group) {

        case TensorGroup::HostInput:
            getTensorViewMap(m_hostInputs);
            break;
        
        case TensorGroup::PinnedInput:
            getTensorViewMap(m_pinnedHostInputs);
            break;
        
        case TensorGroup::DeviceInput:
            getTensorViewMap(m_deviceInputs);
            break;

        case TensorGroup::UnifiedInput:
            getTensorViewMap(m_unifiedMemInputs);
            break;

        case TensorGroup::HostOutput:
            getTensorViewMap(m_hostOutputs);
            break;

        case TensorGroup::PinnedOutput:
            getTensorViewMap(m_pinnedHostOutputs);
            break;

        case TensorGroup::DeviceOutput:
            getTensorViewMap(m_deviceOutputs);
            break;

        case TensorGroup::UnifiedOutput:
            getTensorViewMap(m_unifiedMemOutputs);
            break;

        case TensorGroup::HostPostProcessOutput:
            getTensorViewMap(m_hostPostProcessOutputs);
        
        case TensorGroup::DevicePostProcessOutput:
            getTensorViewMap(m_devicePostProcessOutputs);

        default:
            throw std::runtime_error("Invalid Tensor Group");
            break;
    }
}