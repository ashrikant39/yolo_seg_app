# Usage

> Build dependency note: the full app target requires CUDA, TensorRT, OpenCV, `cxxopts`, and `yaml-cpp`.

## CLI

Run the application from a YAML configuration file:

```bash
./bin/yoloSegApp --config /path/to/config.yaml
```

or positionally:

```bash
./bin/yoloSegApp /path/to/config.yaml
```

See `configs/YoloSegSimple.yaml` for the expected configuration shape.

## Configuration Ownership

`main.cpp` only parses the YAML path and calls:

```cpp
Application app(configPath);
app.run();
```

All construction values come from YAML, including:

- logging path and severity
- frame source type/path/shape/batch size
- preprocessing options
- inference backend/model path/device
- memory tensor buffer groups
- input/output tensor specs
- postprocessing thresholds/options
- result sink mode and `resultsDir`

## C++ API

```cpp
#include "application/Application.hpp"

int main() {
    Application app("/path/to/config.yaml");
    app.run();
    return 0;
}
```
