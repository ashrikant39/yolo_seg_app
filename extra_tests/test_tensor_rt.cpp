#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <memory>
#include <NvOnnxParser.h>

using namespace nvinfer1;

class ExampleLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

int main() {
    ExampleLogger logger;

    // 1. Create builder and network
    // IBuilder* builder = createInferBuilder(logger);
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));

    // INetworkDefinition* network = builder->createNetworkV2(0);
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));

    // 2. Input tensor
    ITensor* input1 = network->addInput("input1", DataType::kFLOAT, Dims3{1, 1, 1});
    ITensor* input2 = network->addInput("input2", DataType::kFLOAT, Dims2{1, 1});

    // 3. Constant layer (value = 1.0f)
    std::unique_ptr<float[]> onePtr(new float[1]{1.0f});
    Weights one{DataType::kFLOAT, onePtr.get(), 1};

    IConstantLayer* const_layer_1 = network->addConstant(Dims3{1, 1, 1}, one);
    IConstantLayer* const_layer_2 = network->addConstant(Dims2{1, 1}, one);

    // 4. Add elementwise addition
    IElementWiseLayer* add_layer1 = network->addElementWise(*input1, *const_layer_1->getOutput(0), ElementWiseOperation::kSUM);
    IElementWiseLayer* add_layer2 = network->addElementWise(*input2, *const_layer_2->getOutput(0), ElementWiseOperation::kSUM);

    ITensor* output1 = add_layer1->getOutput(0);
    ITensor* output2 = add_layer2->getOutput(0);

    output1->setName("output1");
    output2->setName("output2");

    network->markOutput(*output1);
    network->markOutput(*output2);

    // 5. Build engine
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    std::unique_ptr<IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));

    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size()));
    std::cout << "Model Size: " << serializedModel->size() << "\n";

    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());

    int32_t numIOTensors = engine->getNbIOTensors();
    std::cout << "There are " << numIOTensors << " IO Tensors.\n";
    
    for(int i=0; i<numIOTensors; i++){

        const char* name = engine->getIOTensorName(i);
        std::cout << "Idx: " << i << " Name: " << name << " Shape: ";
        Dims dims = engine->getTensorShape(name);

        std::cout << "(";

        for(int64_t dim:dims.d){
            std::cout << dim << ", ";
        }
        std::cout << ")" << "\n";

    }

    // 6. Allocate device memory
    float h_input1[] = {42.0f}, h_input2[] = {23.0f};
    float h_output1[1], h_output2[1];

    void *d_input1, *d_input2, *d_output1, *d_output2;

    cudaMalloc(&d_input1, sizeof(float));
    cudaMalloc(&d_input2, sizeof(float));

    cudaMalloc(&d_output1, sizeof(float));
    cudaMalloc(&d_output2, sizeof(float));

    cudaMemcpy(d_input1, h_input1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, sizeof(float), cudaMemcpyHostToDevice);

    // 7. Set up bindings
    void* bindings[] = {d_input1, d_input2, d_output1, d_output2};

    context->executeV2(bindings);

    // 8. Copy result back
    cudaMemcpy(h_output1, d_output1, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Input1: " << h_input1[0] << ", Output1: " << h_output1[0] << std::endl;

    cudaMemcpy(h_output2, d_output2, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Input2: " << h_input2[0] << ", Output2: " << h_output2[0] << std::endl;

    // 9. Cleanup
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output1);
    cudaFree(d_output2);

    return 0;
}
