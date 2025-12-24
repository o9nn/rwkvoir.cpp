#ifndef RWKVOIR_H
#define RWKVOIR_H

#include "rwkv.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(RWKVOIR_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#        if defined(RWKVOIR_BUILD)
#            define RWKVOIR_API __declspec(dllexport)
#        else
#            define RWKVOIR_API __declspec(dllimport)
#        endif
#    else
#        define RWKVOIR_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define RWKVOIR_API
#endif

#if defined(__cplusplus)
extern "C" {
#endif

    // Forward declarations
    struct rwkvoir_node;
    struct rwkvoir_model;

    // Node types for reservoir computing
    enum rwkvoir_node_type {
        RWKVOIR_NODE_RESERVOIR,    // Echo state network reservoir
        RWKVOIR_NODE_RIDGE,        // Ridge regression readout
        RWKVOIR_NODE_INPUT,        // Input node
        RWKVOIR_NODE_CUSTOM        // Custom user-defined node
    };

    // Activation functions for nodes
    enum rwkvoir_activation {
        RWKVOIR_ACTIVATION_TANH,
        RWKVOIR_ACTIVATION_SIGMOID,
        RWKVOIR_ACTIVATION_RELU,
        RWKVOIR_ACTIVATION_IDENTITY
    };

    // Parameters for creating a reservoir node
    struct rwkvoir_reservoir_params {
        size_t units;              // Number of reservoir units
        float spectral_radius;     // Spectral radius (sr) for stability
        float leak_rate;           // Leak rate (lr) for leaky integration
        float input_scaling;       // Input scaling factor
        float sparsity;            // Sparsity of reservoir connections (0.0-1.0)
        enum rwkvoir_activation activation;
        uint32_t seed;             // Random seed for initialization
    };

    // Parameters for creating a ridge regression node
    struct rwkvoir_ridge_params {
        float ridge;               // Ridge regularization parameter
        size_t input_dim;          // Input dimension
        size_t output_dim;         // Output dimension
    };

    // Node interface - represents a computational unit with state
    // All nodes can be connected to form computational graphs
    
    // Creates a new reservoir node with specified parameters
    // Returns NULL on error
    RWKVOIR_API struct rwkvoir_node * rwkvoir_create_reservoir(
        const struct rwkvoir_reservoir_params * params
    );

    // Creates a new ridge regression readout node
    // Returns NULL on error
    RWKVOIR_API struct rwkvoir_node * rwkvoir_create_ridge(
        const struct rwkvoir_ridge_params * params
    );

    // Creates a new input node
    // Returns NULL on error
    RWKVOIR_API struct rwkvoir_node * rwkvoir_create_input(
        size_t input_dim
    );

    // Runs a forward pass through the node
    // - node: the node to run
    // - input: input data (size depends on node type)
    // - input_len: length of input array
    // - output: output buffer (will be allocated if NULL)
    // - output_len: pointer to store output length
    // Returns false on error
    RWKVOIR_API bool rwkvoir_node_forward(
        struct rwkvoir_node * node,
        const float * input,
        const size_t input_len,
        float ** output,
        size_t * output_len
    );

    // Gets the current state of a node
    // - node: the node
    // - state: output buffer for state (will be allocated if NULL)
    // - state_len: pointer to store state length
    // Returns false on error
    RWKVOIR_API bool rwkvoir_node_get_state(
        const struct rwkvoir_node * node,
        float ** state,
        size_t * state_len
    );

    // Sets the state of a node
    // - node: the node
    // - state: new state data
    // - state_len: length of state array
    // Returns false on error
    RWKVOIR_API bool rwkvoir_node_set_state(
        struct rwkvoir_node * node,
        const float * state,
        const size_t state_len
    );

    // Resets the node state to initial values
    RWKVOIR_API void rwkvoir_node_reset(
        struct rwkvoir_node * node
    );

    // Gets the output dimension of a node
    RWKVOIR_API size_t rwkvoir_node_get_output_dim(
        const struct rwkvoir_node * node
    );

    // Gets the state dimension of a node
    RWKVOIR_API size_t rwkvoir_node_get_state_dim(
        const struct rwkvoir_node * node
    );

    // Frees a node and its resources
    RWKVOIR_API void rwkvoir_node_free(
        struct rwkvoir_node * node
    );

    // Model interface - represents a composition of connected nodes
    
    // Creates a new empty model
    // Returns NULL on error
    RWKVOIR_API struct rwkvoir_model * rwkvoir_model_create(void);

    // Adds a node to the model
    // - model: the model
    // - node: the node to add
    // - name: optional name for the node (can be NULL)
    // Returns node index in model, or -1 on error
    RWKVOIR_API int rwkvoir_model_add_node(
        struct rwkvoir_model * model,
        struct rwkvoir_node * node,
        const char * name
    );

    // Connects two nodes in the model (creates an edge in the computation graph)
    // - model: the model
    // - from_idx: index of source node
    // - to_idx: index of destination node
    // Returns false on error
    RWKVOIR_API bool rwkvoir_model_connect(
        struct rwkvoir_model * model,
        const int from_idx,
        const int to_idx
    );

    // Runs the model on input data
    // - model: the model
    // - input: input data
    // - input_len: length of input array
    // - output: output buffer (will be allocated if NULL)
    // - output_len: pointer to store output length
    // Returns false on error
    RWKVOIR_API bool rwkvoir_model_run(
        struct rwkvoir_model * model,
        const float * input,
        const size_t input_len,
        float ** output,
        size_t * output_len
    );

    // Trains the model (trains trainable nodes like Ridge)
    // - model: the model
    // - X_train: training input data (batch_size * input_dim)
    // - y_train: training target data (batch_size * output_dim)
    // - batch_size: number of training samples
    // - warmup: number of initial samples to skip for warmup
    // Returns false on error
    RWKVOIR_API bool rwkvoir_model_fit(
        struct rwkvoir_model * model,
        const float * X_train,
        const float * y_train,
        const size_t batch_size,
        const size_t warmup
    );

    // Resets all node states in the model
    RWKVOIR_API void rwkvoir_model_reset(
        struct rwkvoir_model * model
    );

    // Gets the number of nodes in the model
    RWKVOIR_API size_t rwkvoir_model_get_node_count(
        const struct rwkvoir_model * model
    );

    // Frees a model and all its nodes
    RWKVOIR_API void rwkvoir_model_free(
        struct rwkvoir_model * model
    );

#if defined(__cplusplus)
}
#endif

#endif // RWKVOIR_H
