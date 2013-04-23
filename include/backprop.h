#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include "parse_data.h"

static const float LEARNING_RATE = 0.0005;

__global__ void backprop_output_layer(int num_threads, int num_layers, int num_weights, int num_neurons, float *outputs, float *desired_output, //inputs
                      float *weights, float *error_prev); //outputs
                                    
__global__ void backprop_layer(int num_threads, int layer, int num_weights, int num_neurons, float *outputs, float *input, //inputs
                            float *weights, float *error_prev);
                                    
__global__ void backprop_network(int num_threads, int num_layers, int num_weights, int num_neurons, float *input, float *outputs, float *desired_output, 
                                    float *weights);//output of the function
