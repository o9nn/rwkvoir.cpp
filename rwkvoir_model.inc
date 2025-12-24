// Model implementation for rwkvoir reservoir computing

#include <stdlib.h>
#include <string.h>

#define RWKVOIR_MAX_NODES 256
#define RWKVOIR_MAX_EDGES 512

// Edge in the computation graph
struct rwkvoir_edge {
    int from_idx;
    int to_idx;
};

// Model structure
struct rwkvoir_model {
    struct rwkvoir_node ** nodes;
    char ** node_names;
    size_t node_count;
    size_t node_capacity;
    
    struct rwkvoir_edge * edges;
    size_t edge_count;
    size_t edge_capacity;
    
    // Execution order (topologically sorted node indices)
    int * exec_order;
    size_t exec_order_len;
    bool exec_order_valid;
    
    // Temporary buffers for intermediate results
    float ** node_outputs;
};

// Helper: Topological sort for execution order
static bool rwkvoir_model_compute_exec_order(struct rwkvoir_model * model) {
    if (!model || model->node_count == 0) {
        return false;
    }
    
    // Simple topological sort using Kahn's algorithm
    int * in_degree = (int *)calloc(model->node_count, sizeof(int));
    if (!in_degree) {
        return false;
    }
    
    // Compute in-degrees
    for (size_t i = 0; i < model->edge_count; i++) {
        in_degree[model->edges[i].to_idx]++;
    }
    
    // Queue for nodes with no incoming edges
    int * queue = (int *)malloc(model->node_count * sizeof(int));
    if (!queue) {
        free(in_degree);
        return false;
    }
    
    int queue_start = 0;
    int queue_end = 0;
    
    // Add all nodes with no incoming edges
    for (size_t i = 0; i < model->node_count; i++) {
        if (in_degree[i] == 0) {
            queue[queue_end++] = i;
        }
    }
    
    // Process queue
    model->exec_order_len = 0;
    while (queue_start < queue_end) {
        int node_idx = queue[queue_start++];
        model->exec_order[model->exec_order_len++] = node_idx;
        
        // Reduce in-degree of neighbors
        for (size_t i = 0; i < model->edge_count; i++) {
            if (model->edges[i].from_idx == node_idx) {
                int neighbor = model->edges[i].to_idx;
                in_degree[neighbor]--;
                if (in_degree[neighbor] == 0) {
                    queue[queue_end++] = neighbor;
                }
            }
        }
    }
    
    free(in_degree);
    free(queue);
    
    // Check if we have a cycle
    if (model->exec_order_len != model->node_count) {
        return false;  // Graph has a cycle
    }
    
    model->exec_order_valid = true;
    return true;
}

// API: Create model
struct rwkvoir_model * rwkvoir_model_create(void) {
    struct rwkvoir_model * model = (struct rwkvoir_model *)calloc(1, sizeof(struct rwkvoir_model));
    if (!model) {
        return NULL;
    }
    
    model->node_capacity = 16;
    model->nodes = (struct rwkvoir_node **)calloc(model->node_capacity, sizeof(struct rwkvoir_node *));
    model->node_names = (char **)calloc(model->node_capacity, sizeof(char *));
    model->node_outputs = (float **)calloc(model->node_capacity, sizeof(float *));
    
    model->edge_capacity = 32;
    model->edges = (struct rwkvoir_edge *)calloc(model->edge_capacity, sizeof(struct rwkvoir_edge));
    
    model->exec_order = (int *)malloc(RWKVOIR_MAX_NODES * sizeof(int));
    
    if (!model->nodes || !model->node_names || !model->edges || !model->exec_order || !model->node_outputs) {
        free(model->nodes);
        free(model->node_names);
        free(model->node_outputs);
        free(model->edges);
        free(model->exec_order);
        free(model);
        return NULL;
    }
    
    model->node_count = 0;
    model->edge_count = 0;
    model->exec_order_len = 0;
    model->exec_order_valid = false;
    
    return model;
}

// API: Add node to model
int rwkvoir_model_add_node(struct rwkvoir_model * model, struct rwkvoir_node * node, const char * name) {
    if (!model || !node) {
        return -1;
    }
    
    if (model->node_count >= RWKVOIR_MAX_NODES) {
        return -1;
    }
    
    // Resize if needed
    if (model->node_count >= model->node_capacity) {
        size_t new_capacity = model->node_capacity * 2;
        struct rwkvoir_node ** new_nodes = (struct rwkvoir_node **)realloc(model->nodes, new_capacity * sizeof(struct rwkvoir_node *));
        char ** new_names = (char **)realloc(model->node_names, new_capacity * sizeof(char *));
        float ** new_outputs = (float **)realloc(model->node_outputs, new_capacity * sizeof(float *));
        
        if (!new_nodes || !new_names || !new_outputs) {
            return -1;
        }
        
        model->nodes = new_nodes;
        model->node_names = new_names;
        model->node_outputs = new_outputs;
        model->node_capacity = new_capacity;
        
        // Initialize new slots
        for (size_t i = model->node_count; i < new_capacity; i++) {
            model->nodes[i] = NULL;
            model->node_names[i] = NULL;
            model->node_outputs[i] = NULL;
        }
    }
    
    int idx = (int)model->node_count;
    model->nodes[idx] = node;
    
    if (name) {
        model->node_names[idx] = strdup(name);
    }
    
    model->node_count++;
    model->exec_order_valid = false;
    
    return idx;
}

// API: Connect nodes
bool rwkvoir_model_connect(struct rwkvoir_model * model, const int from_idx, const int to_idx) {
    if (!model || from_idx < 0 || to_idx < 0 || 
        (size_t)from_idx >= model->node_count || (size_t)to_idx >= model->node_count) {
        return false;
    }
    
    if (model->edge_count >= RWKVOIR_MAX_EDGES) {
        return false;
    }
    
    // Resize if needed
    if (model->edge_count >= model->edge_capacity) {
        size_t new_capacity = model->edge_capacity * 2;
        struct rwkvoir_edge * new_edges = (struct rwkvoir_edge *)realloc(model->edges, new_capacity * sizeof(struct rwkvoir_edge));
        if (!new_edges) {
            return false;
        }
        model->edges = new_edges;
        model->edge_capacity = new_capacity;
    }
    
    model->edges[model->edge_count].from_idx = from_idx;
    model->edges[model->edge_count].to_idx = to_idx;
    model->edge_count++;
    
    model->exec_order_valid = false;
    
    return true;
}

// Helper: Get incoming edges for a node
static int * rwkvoir_model_get_inputs(struct rwkvoir_model * model, int node_idx, size_t * count) {
    *count = 0;
    for (size_t i = 0; i < model->edge_count; i++) {
        if (model->edges[i].to_idx == node_idx) {
            (*count)++;
        }
    }
    
    if (*count == 0) {
        return NULL;
    }
    
    int * inputs = (int *)malloc(*count * sizeof(int));
    if (!inputs) {
        *count = 0;
        return NULL;
    }
    
    size_t idx = 0;
    for (size_t i = 0; i < model->edge_count; i++) {
        if (model->edges[i].to_idx == node_idx) {
            inputs[idx++] = model->edges[i].from_idx;
        }
    }
    
    return inputs;
}

// API: Run model
bool rwkvoir_model_run(struct rwkvoir_model * model, const float * input, const size_t input_len,
                       float ** output, size_t * output_len) {
    if (!model || !input || !output || !output_len) {
        return false;
    }
    
    if (model->node_count == 0) {
        return false;
    }
    
    // Compute execution order if needed
    if (!model->exec_order_valid) {
        if (!rwkvoir_model_compute_exec_order(model)) {
            return false;
        }
    }
    
    // Execute nodes in topological order
    for (size_t i = 0; i < model->exec_order_len; i++) {
        int node_idx = model->exec_order[i];
        struct rwkvoir_node * node = model->nodes[node_idx];
        
        // Get inputs for this node
        size_t n_inputs;
        int * input_indices = rwkvoir_model_get_inputs(model, node_idx, &n_inputs);
        
        float * node_input = NULL;
        size_t node_input_len = 0;
        
        if (n_inputs == 0) {
            // This is an input node - use the model input
            node_input = (float *)input;
            node_input_len = input_len;
        } else if (n_inputs == 1) {
            // Single input - use output from previous node
            node_input = model->node_outputs[input_indices[0]];
            node_input_len = rwkvoir_node_get_output_dim(model->nodes[input_indices[0]]);
        } else {
            // Multiple inputs - concatenate them
            node_input_len = 0;
            for (size_t j = 0; j < n_inputs; j++) {
                node_input_len += rwkvoir_node_get_output_dim(model->nodes[input_indices[j]]);
            }
            
            node_input = (float *)malloc(node_input_len * sizeof(float));
            if (!node_input) {
                free(input_indices);
                return false;
            }
            
            size_t offset = 0;
            for (size_t j = 0; j < n_inputs; j++) {
                int inp_idx = input_indices[j];
                size_t inp_len = rwkvoir_node_get_output_dim(model->nodes[inp_idx]);
                memcpy(node_input + offset, model->node_outputs[inp_idx], inp_len * sizeof(float));
                offset += inp_len;
            }
        }
        
        // Run node forward pass
        float * node_output = model->node_outputs[node_idx];
        size_t node_output_len;
        
        bool success = rwkvoir_node_forward(node, node_input, node_input_len, &node_output, &node_output_len);
        
        // Store output
        model->node_outputs[node_idx] = node_output;
        
        // Cleanup
        if (n_inputs > 1 && node_input != input) {
            free(node_input);
        }
        free(input_indices);
        
        if (!success) {
            return false;
        }
    }
    
    // Output is the output of the last node in execution order
    int last_node_idx = model->exec_order[model->exec_order_len - 1];
    *output_len = rwkvoir_node_get_output_dim(model->nodes[last_node_idx]);
    
    if (*output == NULL) {
        *output = (float *)malloc(*output_len * sizeof(float));
        if (!*output) {
            return false;
        }
    }
    
    memcpy(*output, model->node_outputs[last_node_idx], *output_len * sizeof(float));
    
    return true;
}

// API: Fit model (train)
bool rwkvoir_model_fit(struct rwkvoir_model * model, const float * X_train, const float * y_train,
                       const size_t batch_size, const size_t warmup) {
    if (!model || !X_train || !y_train || batch_size == 0) {
        return false;
    }
    
    // This is a simplified implementation
    // In a full implementation, this would:
    // 1. Run the model on all training samples to collect reservoir states
    // 2. Train the ridge regression readout on collected states
    
    // For now, we'll just iterate through samples and collect states
    // Then train any trainable nodes (like Ridge)
    
    // TODO: Implement batch training for ridge regression
    // This requires collecting all intermediate activations and solving
    // the ridge regression problem
    
    return true;
}

// API: Reset model
void rwkvoir_model_reset(struct rwkvoir_model * model) {
    if (!model) {
        return;
    }
    
    for (size_t i = 0; i < model->node_count; i++) {
        rwkvoir_node_reset(model->nodes[i]);
    }
}

// API: Get node count
size_t rwkvoir_model_get_node_count(const struct rwkvoir_model * model) {
    return model ? model->node_count : 0;
}

// API: Free model
void rwkvoir_model_free(struct rwkvoir_model * model) {
    if (!model) {
        return;
    }
    
    // Free all nodes
    for (size_t i = 0; i < model->node_count; i++) {
        rwkvoir_node_free(model->nodes[i]);
        free(model->node_names[i]);
        free(model->node_outputs[i]);
    }
    
    free(model->nodes);
    free(model->node_names);
    free(model->node_outputs);
    free(model->edges);
    free(model->exec_order);
    free(model);
}
