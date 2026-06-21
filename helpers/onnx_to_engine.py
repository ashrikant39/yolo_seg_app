import tensorrt as trt
import sys, pdb

onnx_file = sys.argv[1]
engine_file = onnx_file[:-4] + "engine"
batch_size = 1

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)

parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_file, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

# Set builder config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33) # 8 GB

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
else:
    config.set_flag(trt.BuilderFlag.TF32)


# Set optimization profile
profile = builder.create_optimization_profile()
input_tensor = network.get_input(0)
input_shape = (batch_size,) + tuple(input_tensor.shape[1:])
profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
config.add_optimization_profile(profile)

# Build engine
engine = builder.build_serialized_network(network, config)

# Serialize and save
with open(engine_file, "wb") as f:
    f.write(engine)