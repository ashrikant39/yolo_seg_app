#include "memory_management/MemoryManager.hpp"

#include <algorithm>
#include <functional>
#include <utility>

namespace {

bool isInputGroup(TensorGroup group) {
    return group == TensorGroup::HostInput ||
           group == TensorGroup::PinnedInput ||
           group == TensorGroup::DeviceInput ||
           group == TensorGroup::UnifiedInput;
}

bool isOutputGroup(TensorGroup group) {
    return group == TensorGroup::HostOutput ||
           group == TensorGroup::PinnedOutput ||
           group == TensorGroup::DeviceOutput ||
           group == TensorGroup::UnifiedOutput ||
           group == TensorGroup::HostPostProcessOutput ||
           group == TensorGroup::DevicePostProcessOutput;
}

bool tensorSpecsMatchGroup(const TensorSpecMap& tensorSpecs, TensorGroup group) {
    if (tensorSpecs.empty()) {
        return false;
    }

    return std::all_of(tensorSpecs.begin(), tensorSpecs.end(), [&](const auto& item) {
        const IOMode mode = item.second.mode;
        return (isInputGroup(group) && mode == IOMode::Input) ||
               (isOutputGroup(group) && mode == IOMode::Output);
    });
}

bool hasUnifiedMemoryBuffer(const TensorViewMap& tensorMap) {
    return std::any_of(tensorMap.begin(), tensorMap.end(), [](const auto& item) {
        return item.second.memoryType == MemoryType::Unified;
    });
}

} // namespace


MemoryManager::MemoryManager(TensorGroupConfig groupLists):
    m_groupLists(std::move(groupLists)) {}

bool MemoryManager::supportsGroup(TensorGroup group) const {
    const std::vector<TensorGroup> groups = allConfiguredGroups();
    return std::find(groups.begin(), groups.end(), group) != groups.end();
}

bool MemoryManager::hasAllocatedGroup(TensorGroup group) const {
    return m_TensorViewsByGroup.find(group) != m_TensorViewsByGroup.end();
}

void MemoryManager::allocateTensorsForGroup(const TensorSpecMap& tensorSpecs, TensorGroup group) {
    if (!supportsGroup(group)) {
        throw std::runtime_error("Unsupported tensor group requested");
    }

    if (hasAllocatedGroup(group)) {
        throw std::runtime_error("Duplicate tensor group allocation");
    }

    switch (group) {
        case TensorGroup::HostInput:
            allocateBuffers(m_hostInputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_hostInputs));
            break;
        case TensorGroup::PinnedInput:
            allocateBuffers(m_pinnedInputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_pinnedInputs));
            break;
        case TensorGroup::DeviceInput:
            allocateBuffers(m_deviceInputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_deviceInputs));
            break;
        case TensorGroup::UnifiedInput:
            allocateBuffers(m_unifiedInputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_unifiedInputs));
            break;
        case TensorGroup::HostOutput:
            allocateBuffers(m_hostOutputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_hostOutputs));
            break;
        case TensorGroup::PinnedOutput:
            allocateBuffers(m_pinnedOutputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_pinnedOutputs));
            break;
        case TensorGroup::DeviceOutput:
            allocateBuffers(m_deviceOutputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_deviceOutputs));
            break;
        case TensorGroup::UnifiedOutput:
            allocateBuffers(m_unifiedOutputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_unifiedOutputs));
            break;
        case TensorGroup::HostPostProcessOutput:
            allocateBuffers(m_hostPostProcessOutputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_hostPostProcessOutputs));
            break;
        case TensorGroup::DevicePostProcessOutput:
            allocateBuffers(m_devicePostProcessOutputs, tensorSpecs);
            m_TensorViewsByGroup.emplace(group, getTensorViews(m_devicePostProcessOutputs));
            break;
    }
}

void MemoryManager::allocateAllTensors(const TensorSpecMap& tensorSpecs) {
    for (TensorGroup group : allConfiguredGroups()) {
        if (!hasAllocatedGroup(group) && tensorSpecsMatchGroup(tensorSpecs, group)) {
            allocateTensorsForGroup(tensorSpecs, group);
        }
    }
}

TensorViewMap& MemoryManager::getTensorViewsFromGroup(TensorGroup group) {
    return m_TensorViewsByGroup.at(group);
}

const TensorViewMap& MemoryManager::getTensorViewsFromGroup(TensorGroup group) const {
    return m_TensorViewsByGroup.at(group);
}

std::vector<TensorGroup> MemoryManager::allConfiguredGroups() const {
    std::vector<TensorGroup> groups;
    groups.reserve(
        m_groupLists.preProcessing.size() +
        m_groupLists.inference.size() +
        m_groupLists.postProcessing.size()
    );

    auto appendUnique = [&](const TensorGroupList& source) {
        for (TensorGroup group : source) {
            if (std::find(groups.begin(), groups.end(), group) == groups.end()) {
                groups.push_back(group);
            }
        }
    };

    appendUnique(m_groupLists.preProcessing);
    appendUnique(m_groupLists.inference);
    appendUnique(m_groupLists.postProcessing);
    return groups;
}

TensorGroup MemoryManager::selectFirstAllocatedGroup(
    const TensorGroupList& groups,
    const std::string& label
) const {

    for (TensorGroup group : groups) {
        if (hasAllocatedGroup(group)) {
            return group;
        }
    }
    throw std::runtime_error("No allocated tensor group for " + label);
}

TensorGroup MemoryManager::selectInferenceInputGroup() const {
    for (TensorGroup group : m_groupLists.inference) {
        if (isInputGroup(group) && hasAllocatedGroup(group)) {
            return group;
        }
    }
    throw std::runtime_error("No allocated inference input tensor group");
}

TensorGroup MemoryManager::selectInferenceOutputGroup() const {
    for (TensorGroup group : m_groupLists.inference) {
        if (isOutputGroup(group) && hasAllocatedGroup(group)) {
            return group;
        }
    }
    throw std::runtime_error("No allocated inference output tensor group");
}

DeviceType MemoryManager::getTensorViewDevice(const TensorViewMap& tensorMap) const {
    if (tensorMap.empty()) {
        return DeviceType::UNSET;
    }

    const DeviceType device = tensorMap.begin()->second.device;
    for (const auto& [name, tensor] : tensorMap) {
        if (tensor.device != device) {
            throw std::runtime_error("Tensor map contains mixed device types");
        }
    }
    return device;
}

void MemoryManager::appendUnifiedReadiness(
    std::vector<TensorTransfer>& transfers,
    TensorGroup group,
    TensorViewMap& bufferViews,
    DeviceType targetDevice
) {
    if (!hasUnifiedMemoryBuffer(bufferViews)) {
        return;
    }

    if (targetDevice == DeviceType::CPU) {
        transfers.push_back(TensorTransfer{
            .sourceGroup = group,
            .targetGroup = group,
            .sourceBufferViews = std::ref(bufferViews),
            .targetBufferViews = std::ref(bufferViews),
            .kind = TensorTransferKind::PrefetchToCpu
        });
    } else if (targetDevice == DeviceType::CUDA) {
        transfers.push_back(TensorTransfer{
            .sourceGroup = group,
            .targetGroup = group,
            .sourceBufferViews = std::ref(bufferViews),
            .targetBufferViews = std::ref(bufferViews),
            .kind = TensorTransferKind::PrefetchToCuda
        });
    }
}

std::vector<TensorTransfer> MemoryManager::makeTransfers(
    TensorGroup sourceGroup,
    TensorGroup targetGroup
) {
    auto& sourceBufferViews = getTensorViewsFromGroup(sourceGroup);
    auto& targetBufferViews = getTensorViewsFromGroup(targetGroup);
    const DeviceType sourceDevice = getTensorViewDevice(sourceBufferViews);
    const DeviceType targetDevice = getTensorViewDevice(targetBufferViews);

    std::vector<TensorTransfer> transfers;

    if (sourceGroup == targetGroup || sourceDevice == targetDevice) {
        appendUnifiedReadiness(transfers, sourceGroup, sourceBufferViews, targetDevice);
        return transfers;
    }

    transfers.push_back(TensorTransfer{
        .sourceGroup = sourceGroup,
        .targetGroup = targetGroup,
        .sourceBufferViews = std::ref(sourceBufferViews),
        .targetBufferViews = std::ref(targetBufferViews),
        .kind = TensorTransferKind::Copy
    });
    appendUnifiedReadiness(transfers, targetGroup, targetBufferViews, targetDevice);

    return transfers;
}

PipelineTensorContext MemoryManager::createPipelineTensorContext() {

    const TensorGroup preProcessingGroup = selectFirstAllocatedGroup(
        m_groupLists.preProcessing,
        "preprocessing"
    );
    const TensorGroup inferenceInputGroup = selectInferenceInputGroup();
    const TensorGroup inferenceOutputGroup = selectInferenceOutputGroup();
    const TensorGroup postProcessingGroup = selectFirstAllocatedGroup(
        m_groupLists.postProcessing,
        "postprocessing"
    );

    auto& preProcessingBufferViews = getTensorViewsFromGroup(preProcessingGroup);
    auto& configuredInferenceInputBufferViews = getTensorViewsFromGroup(inferenceInputGroup);
    auto& inferenceOutputBufferViews = getTensorViewsFromGroup(inferenceOutputGroup);
    auto& configuredPostProcessingBufferViews = getTensorViewsFromGroup(postProcessingGroup);

    const bool aliasPreProcessingIntoInference =
        preProcessingGroup == inferenceInputGroup ||
        getTensorViewDevice(preProcessingBufferViews) ==
        getTensorViewDevice(configuredInferenceInputBufferViews);

    const bool aliasInferenceIntoPostProcessing =
        inferenceOutputGroup == postProcessingGroup ||
        getTensorViewDevice(inferenceOutputBufferViews) ==
        getTensorViewDevice(configuredPostProcessingBufferViews);

    std::reference_wrapper<TensorViewMap> effectiveInferenceInputBufferViews =
        aliasPreProcessingIntoInference
            ? std::ref(preProcessingBufferViews)
            : std::ref(configuredInferenceInputBufferViews);
    TensorGroup effectiveInferenceInputGroup =
        aliasPreProcessingIntoInference ? preProcessingGroup : inferenceInputGroup;

    std::reference_wrapper<TensorViewMap> effectivePostProcessingBufferViews =
        aliasInferenceIntoPostProcessing
            ? std::ref(inferenceOutputBufferViews)
            : std::ref(configuredPostProcessingBufferViews);
    TensorGroup effectivePostProcessingGroup =
        aliasInferenceIntoPostProcessing ? inferenceOutputGroup : postProcessingGroup;

    std::vector<std::reference_wrapper<TensorViewMap>> bindableTensorViews;
    auto hasBindableMap = [&](const TensorViewMap& candidate) {
        return std::any_of(
            bindableTensorViews.begin(),
            bindableTensorViews.end(),
            [&](const std::reference_wrapper<TensorViewMap>& existing) {
                return &existing.get() == &candidate;
            }
        );
    };

    for (TensorGroup group : m_groupLists.inference) {
        if (hasAllocatedGroup(group)) {
            std::reference_wrapper<TensorViewMap> bindableMap = getTensorViewsFromGroup(group);
            if (group == inferenceInputGroup) {
                bindableMap = effectiveInferenceInputBufferViews;
            }

            if (!hasBindableMap(bindableMap.get())) {
                bindableTensorViews.push_back(bindableMap);
            }
        }
    }

    return PipelineTensorContext{
        .preProcessing = PreProcessingTensorContext{
            .bufferViews = std::ref(preProcessingBufferViews),
            .group = preProcessingGroup
        },
        .inference = InferenceTensorContext{
            .inputBufferViews = effectiveInferenceInputBufferViews,
            .inputGroup = effectiveInferenceInputGroup,
            .outputBufferViews = std::ref(inferenceOutputBufferViews),
            .outputGroup = inferenceOutputGroup,
            .bindableTensorViews = std::move(bindableTensorViews)
        },
        .postProcessing = PostProcessingTensorContext{
            .bufferViews = effectivePostProcessingBufferViews,
            .group = effectivePostProcessingGroup
        },
        .preProcessingToInference = makeTransfers(preProcessingGroup, inferenceInputGroup),
        .inferenceToPostProcessing = makeTransfers(inferenceOutputGroup, postProcessingGroup)
    };
}

void MemoryManager::transferTensors(
    const std::vector<TensorTransfer>& transfers,
    const std::vector<std::string>& TensorKeys,
    cudaStream_t stream
) {
    for (const TensorTransfer& transfer : transfers) {
        for (const std::string& key : TensorKeys) {
            const auto& source = transfer.sourceBufferViews.get().at(key);
            auto& target = transfer.targetBufferViews.get().at(key);

            switch (transfer.kind) {
                case TensorTransferKind::Copy:
                    if (source.totalBytes != target.totalBytes) {
                        throw std::runtime_error("Cannot transfer tensor buffers with mismatched byte sizes: " + key);
                    }

                    if (source.device == target.device) {
                        continue;
                    }

                    CUDA_THROW(cudaMemcpyAsync(
                        target.data,
                        source.data,
                        source.totalBytes,
                        cudaMemcpyDefault,
                        stream
                    ));
                    break;

                case TensorTransferKind::PrefetchToCpu:
                    if (source.memoryType == MemoryType::Unified) {
                        CUDA_THROW(
                            cudaMemPrefetchAsync(
                                source.data,
                                source.totalBytes,
                                {cudaMemLocationTypeHost, 0},
                                0,
                                stream
                            ));
                    }
                    break;

                case TensorTransferKind::PrefetchToCuda:
                    if (source.memoryType == MemoryType::Unified) {
                        CUDA_THROW(
                            cudaMemPrefetchAsync(
                                source.data,
                                source.totalBytes,
                                {cudaMemLocationTypeDevice, 0},
                                0,
                                stream
                            ));
                    }
                    break;
            }
        }
    }
}
