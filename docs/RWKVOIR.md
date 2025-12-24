# rwkvoir.cpp - Reservoir Computing for RWKV

This is a reservoir computing extension for rwkv.cpp, implementing ReservoirPy-style features in C++.

## Overview

**rwkvoir** (RWKV + Reservoir) adds Echo State Network (ESN) and reservoir computing capabilities to rwkv.cpp. It provides a modular node-based API for building and running reservoir computing architectures.

## Features

### Node-Based Architecture

- **Node abstraction**: Computational units with state management
- **Multiple node types**: Reservoir, Ridge regression, Input nodes
- **Flexible activation functions**: tanh, sigmoid, ReLU, identity
- **State management**: Get, set, and reset node states

### Model Composition

- **Graph-based models**: Connect nodes to form computation graphs
- **Topological execution**: Automatic execution order computation
- **Multiple inputs**: Support for branching and merging dataflows

### Reservoir Computing

- **Echo State Networks**: Classical reservoir computing with configurable parameters
- **Spectral radius control**: For stability and echo state property
- **Leak rate**: Control memory and temporal dynamics
- **Ridge regression readout**: Trainable linear readout layer

## API Overview

### Core Types

```c
// Node types
enum rwkvoir_node_type {
    RWKVOIR_NODE_RESERVOIR,  // Echo state network reservoir
    RWKVOIR_NODE_RIDGE,      // Ridge regression readout
    RWKVOIR_NODE_INPUT,      // Input node
    RWKVOIR_NODE_CUSTOM      // Custom user-defined node
};

// Activation functions
enum rwkvoir_activation {
    RWKVOIR_ACTIVATION_TANH,
    RWKVOIR_ACTIVATION_SIGMOID,
    RWKVOIR_ACTIVATION_RELU,
    RWKVOIR_ACTIVATION_IDENTITY
};
```

### Creating Nodes

```c
// Create a reservoir node
struct rwkvoir_reservoir_params params = {
    .units = 100,              // Number of reservoir units
    .spectral_radius = 1.25,   // Spectral radius for stability
    .leak_rate = 0.3,          // Leak rate for memory
    .input_scaling = 1.0,      // Input scaling factor
    .sparsity = 0.1,           // Connection sparsity
    .activation = RWKVOIR_ACTIVATION_TANH,
    .seed = 42
};
struct rwkvoir_node * reservoir = rwkvoir_create_reservoir(&params);

// Create a ridge regression readout
struct rwkvoir_ridge_params ridge_params = {
    .ridge = 1e-5,      // Regularization parameter
    .input_dim = 100,
    .output_dim = 10
};
struct rwkvoir_node * ridge = rwkvoir_create_ridge(&ridge_params);

// Create an input node
struct rwkvoir_node * input = rwkvoir_create_input(5);
```

### Building Models

```c
// Create model
struct rwkvoir_model * model = rwkvoir_model_create();

// Add nodes
int input_idx = rwkvoir_model_add_node(model, input, "input");
int reservoir_idx = rwkvoir_model_add_node(model, reservoir, "reservoir");
int ridge_idx = rwkvoir_model_add_node(model, ridge, "readout");

// Connect nodes (input -> reservoir -> ridge)
rwkvoir_model_connect(model, input_idx, reservoir_idx);
rwkvoir_model_connect(model, reservoir_idx, ridge_idx);
```

### Running Models

```c
// Run model on input
float input_data[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
float * output = NULL;
size_t output_len;

bool success = rwkvoir_model_run(model, input_data, 5, &output, &output_len);

// Reset model state
rwkvoir_model_reset(model);

// Cleanup
free(output);
rwkvoir_model_free(model);
```

## Building

rwkvoir is built as part of rwkv.cpp:

```bash
cmake .
cmake --build . --config Release
```

This will produce:
- `librwkvoir.so` (or `.dll`/`.dylib`) - The rwkvoir library
- `test_rwkvoir` - Test suite
- `rwkvoir_example` - Example Echo State Network

## Examples

See `examples/rwkvoir_example.c` for a complete example of creating and running an Echo State Network for time series prediction.

Run the example:
```bash
./bin/rwkvoir_example
```

## Testing

Run the test suite:
```bash
./bin/test_rwkvoir
```

Or use CMake's test runner:
```bash
ctest -R test_rwkvoir
```

## Use Cases

- **Time series prediction**: Financial data, weather forecasting
- **Signal processing**: Audio, video, sensor data
- **Pattern recognition**: Sequence classification
- **Chaos modeling**: Lorenz attractors, strange attractors
- **Dynamical systems**: Non-linear system identification

## Implementation Details

### Node Implementation

Nodes are implemented with function pointers for polymorphism:

```c
struct rwkvoir_node {
    enum rwkvoir_node_type type;
    size_t output_dim;
    size_t state_dim;
    float * state;
    void * params;
    
    bool (*forward)(struct rwkvoir_node * node, const float * input, size_t input_len, float * output);
    void (*reset)(struct rwkvoir_node * node);
    void (*free_params)(void * params);
};
```

### Model Execution

Models use topological sorting (Kahn's algorithm) to determine execution order, ensuring nodes are computed in the correct dependency order.

### Memory Management

- Nodes own their internal state and parameters
- Models own their nodes (freed on model cleanup)
- User is responsible for freeing outputs from `rwkvoir_model_run`

## Comparison with ReservoirPy

| Feature | ReservoirPy (Python) | rwkvoir (C) |
|---------|---------------------|-------------|
| Node abstraction | ✓ | ✓ |
| Model composition | ✓ | ✓ |
| Echo State Networks | ✓ | ✓ |
| Ridge regression | ✓ | ✓ |
| Online training | ✓ | Partial |
| Deep ESN | ✓ | Roadmap |
| NVAR | ✓ | Roadmap |

## Future Work

- [ ] Online/incremental training for Ridge regression
- [ ] Deep Echo State Networks (DeepESN)
- [ ] NVAR (Nonlinear Vector Auto-Regression)
- [ ] Intrinsic plasticity
- [ ] GPU acceleration via ggml
- [ ] Python bindings
- [ ] More activation functions
- [ ] Sparse matrix optimizations

## License

Same as rwkv.cpp (MIT License)

## References

- [ReservoirPy Documentation](https://reservoirpy.readthedocs.io/)
- [Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network)
- [Reservoir Computing](https://en.wikipedia.org/wiki/Reservoir_computing)

## Contributing

Contributions are welcome! Please follow the code style in `docs/CODE_STYLE.md`.
