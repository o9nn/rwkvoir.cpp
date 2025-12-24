// Test for rwkvoir reservoir computing features

#include "rwkvoir.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "ASSERTION FAILED: %s\n", msg); \
        return 1; \
    } \
} while(0)

#define ASSERT_CLOSE(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        fprintf(stderr, "ASSERTION FAILED: %s (%.6f != %.6f, diff=%.6f)\n", msg, (float)(a), (float)(b), fabsf((a) - (b))); \
        return 1; \
    } \
} while(0)

int test_node_creation() {
    printf("Testing node creation...\n");
    
    // Test reservoir node creation
    struct rwkvoir_reservoir_params res_params = {
        .units = 100,
        .spectral_radius = 1.0f,
        .leak_rate = 0.3f,
        .input_scaling = 1.0f,
        .sparsity = 0.1f,
        .activation = RWKVOIR_ACTIVATION_TANH,
        .seed = 42
    };
    
    struct rwkvoir_node * reservoir = rwkvoir_create_reservoir(&res_params);
    ASSERT(reservoir != NULL, "Failed to create reservoir node");
    ASSERT(rwkvoir_node_get_output_dim(reservoir) == 100, "Reservoir output dim incorrect");
    ASSERT(rwkvoir_node_get_state_dim(reservoir) == 100, "Reservoir state dim incorrect");
    
    // Test ridge node creation
    struct rwkvoir_ridge_params ridge_params = {
        .ridge = 1e-5f,
        .input_dim = 100,
        .output_dim = 10
    };
    
    struct rwkvoir_node * ridge = rwkvoir_create_ridge(&ridge_params);
    ASSERT(ridge != NULL, "Failed to create ridge node");
    ASSERT(rwkvoir_node_get_output_dim(ridge) == 10, "Ridge output dim incorrect");
    ASSERT(rwkvoir_node_get_state_dim(ridge) == 0, "Ridge should have no state");
    
    // Test input node creation
    struct rwkvoir_node * input = rwkvoir_create_input(5);
    ASSERT(input != NULL, "Failed to create input node");
    ASSERT(rwkvoir_node_get_output_dim(input) == 5, "Input output dim incorrect");
    
    // Cleanup
    rwkvoir_node_free(reservoir);
    rwkvoir_node_free(ridge);
    rwkvoir_node_free(input);
    
    printf("Node creation tests passed!\n");
    return 0;
}

int test_node_forward() {
    printf("Testing node forward pass...\n");
    
    // Create reservoir
    struct rwkvoir_reservoir_params res_params = {
        .units = 50,
        .spectral_radius = 0.9f,
        .leak_rate = 0.5f,
        .input_scaling = 1.0f,
        .sparsity = 0.1f,
        .activation = RWKVOIR_ACTIVATION_TANH,
        .seed = 42
    };
    
    struct rwkvoir_node * reservoir = rwkvoir_create_reservoir(&res_params);
    ASSERT(reservoir != NULL, "Failed to create reservoir");
    
    // Create input
    float input[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    float * output = NULL;
    size_t output_len = 0;
    
    // Run forward pass
    bool success = rwkvoir_node_forward(reservoir, input, 5, &output, &output_len);
    ASSERT(success, "Forward pass failed");
    ASSERT(output != NULL, "Output is NULL");
    ASSERT(output_len == 50, "Output length incorrect");
    
    // Check output is bounded (tanh activation)
    for (size_t i = 0; i < output_len; i++) {
        ASSERT(output[i] >= -1.0f && output[i] <= 1.0f, "Output out of tanh range");
    }
    
    // Run another forward pass - state should change
    float * output2 = NULL;
    size_t output2_len = 0;
    success = rwkvoir_node_forward(reservoir, input, 5, &output2, &output2_len);
    ASSERT(success, "Second forward pass failed");
    
    // Outputs should be different due to state
    bool different = false;
    for (size_t i = 0; i < output_len; i++) {
        if (fabsf(output[i] - output2[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    ASSERT(different, "State did not change between forward passes");
    
    // Cleanup
    free(output);
    free(output2);
    rwkvoir_node_free(reservoir);
    
    printf("Node forward pass tests passed!\n");
    return 0;
}

int test_node_state() {
    printf("Testing node state management...\n");
    
    struct rwkvoir_reservoir_params res_params = {
        .units = 20,
        .spectral_radius = 0.9f,
        .leak_rate = 0.5f,
        .input_scaling = 1.0f,
        .sparsity = 0.1f,
        .activation = RWKVOIR_ACTIVATION_TANH,
        .seed = 42
    };
    
    struct rwkvoir_node * reservoir = rwkvoir_create_reservoir(&res_params);
    ASSERT(reservoir != NULL, "Failed to create reservoir");
    
    // Run some forward passes
    float input[3] = {0.1f, 0.2f, 0.3f};
    float * output = NULL;
    size_t output_len = 0;
    
    for (int i = 0; i < 5; i++) {
        rwkvoir_node_forward(reservoir, input, 3, &output, &output_len);
    }
    
    // Get state
    float * state = NULL;
    size_t state_len = 0;
    bool success = rwkvoir_node_get_state(reservoir, &state, &state_len);
    ASSERT(success, "Failed to get state");
    ASSERT(state != NULL, "State is NULL");
    ASSERT(state_len == 20, "State length incorrect");
    
    // Reset node
    rwkvoir_node_reset(reservoir);
    
    // Get state after reset - should be zeros
    float * state_after_reset = NULL;
    size_t state_after_reset_len = 0;
    success = rwkvoir_node_get_state(reservoir, &state_after_reset, &state_after_reset_len);
    ASSERT(success, "Failed to get state after reset");
    
    for (size_t i = 0; i < state_after_reset_len; i++) {
        ASSERT_CLOSE(state_after_reset[i], 0.0f, 1e-6f, "State not reset to zero");
    }
    
    // Cleanup
    free(output);
    free(state);
    free(state_after_reset);
    rwkvoir_node_free(reservoir);
    
    printf("Node state tests passed!\n");
    return 0;
}

int test_model_creation() {
    printf("Testing model creation...\n");
    
    struct rwkvoir_model * model = rwkvoir_model_create();
    ASSERT(model != NULL, "Failed to create model");
    ASSERT(rwkvoir_model_get_node_count(model) == 0, "New model should have 0 nodes");
    
    // Add some nodes
    struct rwkvoir_node * input = rwkvoir_create_input(5);
    struct rwkvoir_reservoir_params res_params = {
        .units = 50,
        .spectral_radius = 0.9f,
        .leak_rate = 0.3f,
        .input_scaling = 1.0f,
        .sparsity = 0.1f,
        .activation = RWKVOIR_ACTIVATION_TANH,
        .seed = 42
    };
    struct rwkvoir_node * reservoir = rwkvoir_create_reservoir(&res_params);
    
    struct rwkvoir_ridge_params ridge_params = {
        .ridge = 1e-5f,
        .input_dim = 50,
        .output_dim = 3
    };
    struct rwkvoir_node * ridge = rwkvoir_create_ridge(&ridge_params);
    
    int input_idx = rwkvoir_model_add_node(model, input, "input");
    int reservoir_idx = rwkvoir_model_add_node(model, reservoir, "reservoir");
    int ridge_idx = rwkvoir_model_add_node(model, ridge, "ridge");
    
    ASSERT(input_idx == 0, "Input index incorrect");
    ASSERT(reservoir_idx == 1, "Reservoir index incorrect");
    ASSERT(ridge_idx == 2, "Ridge index incorrect");
    ASSERT(rwkvoir_model_get_node_count(model) == 3, "Model should have 3 nodes");
    
    // Connect nodes
    bool success = rwkvoir_model_connect(model, input_idx, reservoir_idx);
    ASSERT(success, "Failed to connect input to reservoir");
    
    success = rwkvoir_model_connect(model, reservoir_idx, ridge_idx);
    ASSERT(success, "Failed to connect reservoir to ridge");
    
    // Cleanup
    rwkvoir_model_free(model);
    
    printf("Model creation tests passed!\n");
    return 0;
}

int test_model_run() {
    printf("Testing model execution...\n");
    
    // Create a simple model: input -> reservoir -> ridge
    struct rwkvoir_model * model = rwkvoir_model_create();
    ASSERT(model != NULL, "Failed to create model");
    
    struct rwkvoir_node * input = rwkvoir_create_input(3);
    struct rwkvoir_reservoir_params res_params = {
        .units = 30,
        .spectral_radius = 0.9f,
        .leak_rate = 0.3f,
        .input_scaling = 1.0f,
        .sparsity = 0.1f,
        .activation = RWKVOIR_ACTIVATION_TANH,
        .seed = 42
    };
    struct rwkvoir_node * reservoir = rwkvoir_create_reservoir(&res_params);
    
    struct rwkvoir_ridge_params ridge_params = {
        .ridge = 1e-5f,
        .input_dim = 30,
        .output_dim = 2
    };
    struct rwkvoir_node * ridge = rwkvoir_create_ridge(&ridge_params);
    
    int input_idx = rwkvoir_model_add_node(model, input, "input");
    int reservoir_idx = rwkvoir_model_add_node(model, reservoir, "reservoir");
    int ridge_idx = rwkvoir_model_add_node(model, ridge, "ridge");
    
    rwkvoir_model_connect(model, input_idx, reservoir_idx);
    rwkvoir_model_connect(model, reservoir_idx, ridge_idx);
    
    // Run model
    float input_data[3] = {0.5f, 0.3f, 0.1f};
    float * output = NULL;
    size_t output_len = 0;
    
    bool success = rwkvoir_model_run(model, input_data, 3, &output, &output_len);
    ASSERT(success, "Model run failed");
    ASSERT(output != NULL, "Output is NULL");
    ASSERT(output_len == 2, "Output length incorrect");
    
    // Run again with different input
    float input_data2[3] = {0.1f, 0.7f, 0.2f};
    float * output2 = NULL;
    size_t output2_len = 0;
    
    success = rwkvoir_model_run(model, input_data2, 3, &output2, &output2_len);
    ASSERT(success, "Second model run failed");
    
    // Cleanup
    free(output);
    free(output2);
    rwkvoir_model_free(model);
    
    printf("Model execution tests passed!\n");
    return 0;
}

int main() {
    printf("=== Running rwkvoir tests ===\n\n");
    
    int result = 0;
    
    result |= test_node_creation();
    result |= test_node_forward();
    result |= test_node_state();
    result |= test_model_creation();
    result |= test_model_run();
    
    if (result == 0) {
        printf("\n=== All tests passed! ===\n");
    } else {
        printf("\n=== Some tests failed ===\n");
    }
    
    return result;
}
