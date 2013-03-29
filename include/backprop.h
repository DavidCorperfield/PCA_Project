#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include "parse_data.h"

#define MAX_NUM_NEURONS 28*28
#define MAX_NUM_WEIGHTS 28*28
#define OUTPUT_LAYER_NEURONS (int)10

#define LEARNING_RATE (float)0.0005


__global__ void backprop_output_layer(int num_weights, int num_neurons, float *layer_input, float *actual_output, float *desired_output, 
                                    float *weights, float *error_prev);//outputs, weights is an input and output
                                    
__global__ void backprop_layer(int num_weights, int num_neurons, int num_outputs, float *input_error, float *weights, 
                                    float *calculated_input);//output
