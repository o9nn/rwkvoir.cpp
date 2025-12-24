// Example usage of rwkvoir reservoir computing library
// Demonstrates creating an Echo State Network for time series prediction

#include "rwkvoir.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    printf("=== rwkvoir Example: Echo State Network ===\n\n");
    
    // Create a simple ESN model for time series prediction
    // Architecture: input -> reservoir -> ridge readout
    
    printf("Creating Echo State Network model...\n");
    
    struct rwkvoir_model * esn = rwkvoir_model_create();
    if (!esn) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }
    
    // Create input node (1D input for time series)
    struct rwkvoir_node * input = rwkvoir_create_input(1);
    if (!input) {
        fprintf(stderr, "Failed to create input node\n");
        rwkvoir_model_free(esn);
        return 1;
    }
    
    // Create reservoir with 100 units
    struct rwkvoir_reservoir_params reservoir_params = {
        .units = 100,
        .spectral_radius = 1.25f,
        .leak_rate = 0.3f,
        .input_scaling = 1.0f,
        .sparsity = 0.1f,
        .activation = RWKVOIR_ACTIVATION_TANH,
        .seed = 42
    };
    
    struct rwkvoir_node * reservoir = rwkvoir_create_reservoir(&reservoir_params);
    if (!reservoir) {
        fprintf(stderr, "Failed to create reservoir node\n");
        rwkvoir_node_free(input);
        rwkvoir_model_free(esn);
        return 1;
    }
    
    // Create ridge regression readout (1D output)
    struct rwkvoir_ridge_params ridge_params = {
        .ridge = 1e-5f,
        .input_dim = 100,
        .output_dim = 1
    };
    
    struct rwkvoir_node * ridge = rwkvoir_create_ridge(&ridge_params);
    if (!ridge) {
        fprintf(stderr, "Failed to create ridge node\n");
        rwkvoir_node_free(input);
        rwkvoir_node_free(reservoir);
        rwkvoir_model_free(esn);
        return 1;
    }
    
    // Add nodes to model
    int input_idx = rwkvoir_model_add_node(esn, input, "input");
    int reservoir_idx = rwkvoir_model_add_node(esn, reservoir, "reservoir");
    int ridge_idx = rwkvoir_model_add_node(esn, ridge, "readout");
    
    printf("  Input node: index %d\n", input_idx);
    printf("  Reservoir node: index %d\n", reservoir_idx);
    printf("  Readout node: index %d\n", ridge_idx);
    
    // Connect nodes
    rwkvoir_model_connect(esn, input_idx, reservoir_idx);
    rwkvoir_model_connect(esn, reservoir_idx, ridge_idx);
    
    printf("Model created with %zu nodes\n\n", rwkvoir_model_get_node_count(esn));
    
    // Generate synthetic time series data
    printf("Generating synthetic sine wave data...\n");
    const int n_samples = 200;
    float * time_series = (float *)malloc(n_samples * sizeof(float));
    
    for (int i = 0; i < n_samples; i++) {
        time_series[i] = sinf(2.0f * M_PI * (float)i / 20.0f);
    }
    
    printf("Generated %d samples\n\n", n_samples);
    
    // Run model on time series
    printf("Running model on time series...\n");
    
    float * output = NULL;
    size_t output_len = 0;
    
    for (int i = 0; i < 10; i++) {
        float input_value = time_series[i];
        rwkvoir_model_run(esn, &input_value, 1, &output, &output_len);
        printf("  Step %d: input=%.4f, output=%.4f\n", i, input_value, output[0]);
    }
    
    printf("\nModel execution complete!\n");
    
    // Cleanup
    printf("\nCleaning up...\n");
    free(time_series);
    free(output);
    rwkvoir_model_free(esn);
    
    printf("Done!\n");
    
    return 0;
}
