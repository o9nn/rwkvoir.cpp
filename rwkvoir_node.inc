// Node implementation for rwkvoir reservoir computing

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Base node structure
struct rwkvoir_node {
    enum rwkvoir_node_type type;
    size_t output_dim;
    size_t state_dim;
    float * state;
    void * params;  // Type-specific parameters
    
    // Function pointers for node operations
    bool (*forward)(struct rwkvoir_node * node, const float * input, size_t input_len, float * output);
    void (*reset)(struct rwkvoir_node * node);
    void (*free_params)(void * params);
};

// Reservoir node specific data
struct rwkvoir_reservoir_data {
    size_t units;
    float spectral_radius;
    float leak_rate;
    float input_scaling;
    size_t input_dim;
    enum rwkvoir_activation activation;
    
    float * W_in;      // Input weights: units x input_dim
    float * W_res;     // Reservoir weights: units x units
    float * bias;      // Bias: units
};

// Ridge node specific data
struct rwkvoir_ridge_data {
    float ridge;
    size_t input_dim;
    size_t output_dim;
    float * W_out;     // Output weights: output_dim x input_dim
    float * bias;      // Bias: output_dim
    bool trained;
    
    // Accumulated matrices for training
    float * XtX;       // input_dim x input_dim
    float * XtY;       // input_dim x output_dim
    size_t n_samples;
};

// Input node specific data
struct rwkvoir_input_data {
    size_t input_dim;
};

// Activation functions
static inline float rwkvoir_activation_tanh(float x) {
    return tanhf(x);
}

static inline float rwkvoir_activation_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float rwkvoir_activation_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float rwkvoir_activation_identity(float x) {
    return x;
}

static inline float rwkvoir_apply_activation(float x, enum rwkvoir_activation act) {
    switch (act) {
        case RWKVOIR_ACTIVATION_TANH: return rwkvoir_activation_tanh(x);
        case RWKVOIR_ACTIVATION_SIGMOID: return rwkvoir_activation_sigmoid(x);
        case RWKVOIR_ACTIVATION_RELU: return rwkvoir_activation_relu(x);
        case RWKVOIR_ACTIVATION_IDENTITY: return rwkvoir_activation_identity(x);
        default: return rwkvoir_activation_tanh(x);
    }
}

// Random number generation helpers
static inline float rwkvoir_rand_uniform(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

static inline float rwkvoir_rand_normal(float mean, float std) {
    // Box-Muller transform
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + std * z;
}

// Matrix operations
static void rwkvoir_matrix_vector_mult(const float * A, const float * x, float * y, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        y[i] = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            y[i] += A[i * cols + j] * x[j];
        }
    }
}

// Reservoir node forward pass
static bool rwkvoir_reservoir_forward(struct rwkvoir_node * node, const float * input, size_t input_len, float * output) {
    struct rwkvoir_reservoir_data * data = (struct rwkvoir_reservoir_data *)node->params;
    
    if (input_len != data->input_dim) {
        return false;
    }
    
    float * new_state = (float *)malloc(data->units * sizeof(float));
    if (!new_state) {
        return false;
    }
    
    // Compute: new_state = (1 - lr) * state + lr * activation(W_in * input + W_res * state + bias)
    float * temp = (float *)malloc(data->units * sizeof(float));
    if (!temp) {
        free(new_state);
        return false;
    }
    
    // W_in * input
    rwkvoir_matrix_vector_mult(data->W_in, input, temp, data->units, data->input_dim);
    
    // Add W_res * state
    float * res_contribution = (float *)malloc(data->units * sizeof(float));
    if (!res_contribution) {
        free(new_state);
        free(temp);
        return false;
    }
    rwkvoir_matrix_vector_mult(data->W_res, node->state, res_contribution, data->units, data->units);
    
    // Combine and apply activation
    for (size_t i = 0; i < data->units; i++) {
        float pre_activation = temp[i] + res_contribution[i] + data->bias[i];
        float activated = rwkvoir_apply_activation(pre_activation, data->activation);
        new_state[i] = (1.0f - data->leak_rate) * node->state[i] + data->leak_rate * activated;
    }
    
    // Update state
    memcpy(node->state, new_state, data->units * sizeof(float));
    
    // Output is the state
    memcpy(output, new_state, data->units * sizeof(float));
    
    free(new_state);
    free(temp);
    free(res_contribution);
    
    return true;
}

// Reservoir node reset
static void rwkvoir_reservoir_reset(struct rwkvoir_node * node) {
    struct rwkvoir_reservoir_data * data = (struct rwkvoir_reservoir_data *)node->params;
    memset(node->state, 0, data->units * sizeof(float));
}

// Reservoir node cleanup
static void rwkvoir_reservoir_free(void * params) {
    struct rwkvoir_reservoir_data * data = (struct rwkvoir_reservoir_data *)params;
    if (data) {
        free(data->W_in);
        free(data->W_res);
        free(data->bias);
        free(data);
    }
}

// Ridge node forward pass
static bool rwkvoir_ridge_forward(struct rwkvoir_node * node, const float * input, size_t input_len, float * output) {
    struct rwkvoir_ridge_data * data = (struct rwkvoir_ridge_data *)node->params;
    
    if (input_len != data->input_dim) {
        return false;
    }
    
    if (!data->trained) {
        // Not trained yet, output zeros
        memset(output, 0, data->output_dim * sizeof(float));
        return true;
    }
    
    // output = W_out * input + bias
    rwkvoir_matrix_vector_mult(data->W_out, input, output, data->output_dim, data->input_dim);
    
    for (size_t i = 0; i < data->output_dim; i++) {
        output[i] += data->bias[i];
    }
    
    return true;
}

// Ridge node reset (no state for ridge)
static void rwkvoir_ridge_reset(struct rwkvoir_node * node) {
    // Ridge has no state
}

// Ridge node cleanup
static void rwkvoir_ridge_free(void * params) {
    struct rwkvoir_ridge_data * data = (struct rwkvoir_ridge_data *)params;
    if (data) {
        free(data->W_out);
        free(data->bias);
        free(data->XtX);
        free(data->XtY);
        free(data);
    }
}

// Input node forward pass (identity)
static bool rwkvoir_input_forward(struct rwkvoir_node * node, const float * input, size_t input_len, float * output) {
    struct rwkvoir_input_data * data = (struct rwkvoir_input_data *)node->params;
    
    if (input_len != data->input_dim) {
        return false;
    }
    
    memcpy(output, input, input_len * sizeof(float));
    return true;
}

// Input node reset
static void rwkvoir_input_reset(struct rwkvoir_node * node) {
    // Input has no state
}

// Input node cleanup
static void rwkvoir_input_free(void * params) {
    struct rwkvoir_input_data * data = (struct rwkvoir_input_data *)params;
    free(data);
}

// API: Create reservoir node
struct rwkvoir_node * rwkvoir_create_reservoir(const struct rwkvoir_reservoir_params * params) {
    if (!params || params->units == 0) {
        return NULL;
    }
    
    struct rwkvoir_node * node = (struct rwkvoir_node *)calloc(1, sizeof(struct rwkvoir_node));
    if (!node) {
        return NULL;
    }
    
    struct rwkvoir_reservoir_data * data = (struct rwkvoir_reservoir_data *)calloc(1, sizeof(struct rwkvoir_reservoir_data));
    if (!data) {
        free(node);
        return NULL;
    }
    
    // Set parameters
    data->units = params->units;
    data->spectral_radius = params->spectral_radius;
    data->leak_rate = params->leak_rate;
    data->input_scaling = params->input_scaling;
    data->activation = params->activation;
    data->input_dim = 0;  // Will be set on first forward pass or can be passed
    
    // Allocate weights (we'll initialize them when we know input_dim)
    // For now, assume input_dim is inferred from first use
    // This is a simplification - in practice, you'd want to specify it
    
    node->type = RWKVOIR_NODE_RESERVOIR;
    node->output_dim = params->units;
    node->state_dim = params->units;
    node->state = (float *)calloc(params->units, sizeof(float));
    node->params = data;
    node->forward = rwkvoir_reservoir_forward;
    node->reset = rwkvoir_reservoir_reset;
    node->free_params = rwkvoir_reservoir_free;
    
    if (!node->state) {
        rwkvoir_reservoir_free(data);
        free(node);
        return NULL;
    }
    
    // Initialize random weights
    srand(params->seed);
    
    // For now, we'll defer weight initialization until we know input dimensions
    // This is handled in a helper function
    
    return node;
}

// Helper to initialize reservoir weights once input dimension is known
static bool rwkvoir_reservoir_init_weights(struct rwkvoir_node * node, size_t input_dim) {
    struct rwkvoir_reservoir_data * data = (struct rwkvoir_reservoir_data *)node->params;
    
    if (data->W_in != NULL) {
        return true;  // Already initialized
    }
    
    data->input_dim = input_dim;
    
    // Allocate weights
    data->W_in = (float *)malloc(data->units * input_dim * sizeof(float));
    data->W_res = (float *)malloc(data->units * data->units * sizeof(float));
    data->bias = (float *)calloc(data->units, sizeof(float));
    
    if (!data->W_in || !data->W_res || !data->bias) {
        free(data->W_in);
        free(data->W_res);
        free(data->bias);
        return false;
    }
    
    // Initialize input weights with uniform distribution scaled by input_scaling
    for (size_t i = 0; i < data->units * input_dim; i++) {
        data->W_in[i] = rwkvoir_rand_uniform(-1.0f, 1.0f) * data->input_scaling;
    }
    
    // Initialize reservoir weights with sparse random matrix
    // Then scale by spectral radius
    for (size_t i = 0; i < data->units * data->units; i++) {
        data->W_res[i] = rwkvoir_rand_uniform(-0.5f, 0.5f);
    }
    
    // Simplified spectral radius scaling (proper implementation would compute eigenvalues)
    // For now, just scale by spectral_radius / sqrt(units)
    float scale = data->spectral_radius / sqrtf((float)data->units);
    for (size_t i = 0; i < data->units * data->units; i++) {
        data->W_res[i] *= scale;
    }
    
    return true;
}

// API: Create ridge node
struct rwkvoir_node * rwkvoir_create_ridge(const struct rwkvoir_ridge_params * params) {
    if (!params || params->input_dim == 0 || params->output_dim == 0) {
        return NULL;
    }
    
    struct rwkvoir_node * node = (struct rwkvoir_node *)calloc(1, sizeof(struct rwkvoir_node));
    if (!node) {
        return NULL;
    }
    
    struct rwkvoir_ridge_data * data = (struct rwkvoir_ridge_data *)calloc(1, sizeof(struct rwkvoir_ridge_data));
    if (!data) {
        free(node);
        return NULL;
    }
    
    data->ridge = params->ridge;
    data->input_dim = params->input_dim;
    data->output_dim = params->output_dim;
    data->trained = false;
    
    // Allocate weights
    data->W_out = (float *)calloc(params->output_dim * params->input_dim, sizeof(float));
    data->bias = (float *)calloc(params->output_dim, sizeof(float));
    data->XtX = (float *)calloc(params->input_dim * params->input_dim, sizeof(float));
    data->XtY = (float *)calloc(params->input_dim * params->output_dim, sizeof(float));
    
    if (!data->W_out || !data->bias || !data->XtX || !data->XtY) {
        rwkvoir_ridge_free(data);
        free(node);
        return NULL;
    }
    
    node->type = RWKVOIR_NODE_RIDGE;
    node->output_dim = params->output_dim;
    node->state_dim = 0;  // Ridge has no state
    node->state = NULL;
    node->params = data;
    node->forward = rwkvoir_ridge_forward;
    node->reset = rwkvoir_ridge_reset;
    node->free_params = rwkvoir_ridge_free;
    
    return node;
}

// API: Create input node
struct rwkvoir_node * rwkvoir_create_input(size_t input_dim) {
    if (input_dim == 0) {
        return NULL;
    }
    
    struct rwkvoir_node * node = (struct rwkvoir_node *)calloc(1, sizeof(struct rwkvoir_node));
    if (!node) {
        return NULL;
    }
    
    struct rwkvoir_input_data * data = (struct rwkvoir_input_data *)calloc(1, sizeof(struct rwkvoir_input_data));
    if (!data) {
        free(node);
        return NULL;
    }
    
    data->input_dim = input_dim;
    
    node->type = RWKVOIR_NODE_INPUT;
    node->output_dim = input_dim;
    node->state_dim = 0;
    node->state = NULL;
    node->params = data;
    node->forward = rwkvoir_input_forward;
    node->reset = rwkvoir_input_reset;
    node->free_params = rwkvoir_input_free;
    
    return node;
}

// API: Node forward pass
bool rwkvoir_node_forward(struct rwkvoir_node * node, const float * input, const size_t input_len, 
                          float ** output, size_t * output_len) {
    if (!node || !input || !output_len) {
        return false;
    }
    
    // Initialize reservoir weights if needed
    if (node->type == RWKVOIR_NODE_RESERVOIR) {
        struct rwkvoir_reservoir_data * data = (struct rwkvoir_reservoir_data *)node->params;
        if (!data->W_in) {
            if (!rwkvoir_reservoir_init_weights(node, input_len)) {
                return false;
            }
        }
    }
    
    // Allocate output if needed
    if (*output == NULL) {
        *output = (float *)malloc(node->output_dim * sizeof(float));
        if (!*output) {
            return false;
        }
    }
    
    *output_len = node->output_dim;
    
    return node->forward(node, input, input_len, *output);
}

// API: Get node state
bool rwkvoir_node_get_state(const struct rwkvoir_node * node, float ** state, size_t * state_len) {
    if (!node || !state_len) {
        return false;
    }
    
    *state_len = node->state_dim;
    
    if (node->state_dim == 0) {
        *state = NULL;
        return true;
    }
    
    if (*state == NULL) {
        *state = (float *)malloc(node->state_dim * sizeof(float));
        if (!*state) {
            return false;
        }
    }
    
    memcpy(*state, node->state, node->state_dim * sizeof(float));
    return true;
}

// API: Set node state
bool rwkvoir_node_set_state(struct rwkvoir_node * node, const float * state, const size_t state_len) {
    if (!node || !state || state_len != node->state_dim) {
        return false;
    }
    
    if (node->state_dim == 0) {
        return true;
    }
    
    memcpy(node->state, state, node->state_dim * sizeof(float));
    return true;
}

// API: Reset node
void rwkvoir_node_reset(struct rwkvoir_node * node) {
    if (node && node->reset) {
        node->reset(node);
    }
}

// API: Get output dimension
size_t rwkvoir_node_get_output_dim(const struct rwkvoir_node * node) {
    return node ? node->output_dim : 0;
}

// API: Get state dimension
size_t rwkvoir_node_get_state_dim(const struct rwkvoir_node * node) {
    return node ? node->state_dim : 0;
}

// API: Free node
void rwkvoir_node_free(struct rwkvoir_node * node) {
    if (!node) {
        return;
    }
    
    if (node->free_params && node->params) {
        node->free_params(node->params);
    }
    
    free(node->state);
    free(node);
}
