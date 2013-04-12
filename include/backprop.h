#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include "parse_data.h"

static const float LEARNING_RATE = 0.05;

__global__ void backprop_output_layer(int num_weights, int num_neurons, float *layer_input, float *actual_output, float *desired_output, 
                                    float *weights, float *error_prev);//outputs, weights is an input and output
                                    
__global__ void backprop_layer(int num_weights, int num_neurons, float *layer_input, float *output_error,
                                    float *weights, float *error_prev);//outputs
                                    
__global__ void backprop_network(int num_threads, int num_layers, int num_weights, int num_neurons, float *input, float *outputs, float *desired_output, 
                                    float *weights);//output of the function
