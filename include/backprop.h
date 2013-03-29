#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include "parse_data.h"

#define MAX_NUM_NEURONS 28*28
#define MAX_NUM_WEIGHTS 28*28

#define LEARNING_RATE 0.0005

__global__ void backprop_output_layer(float *layer_input, float *actual_output, float *desired_output, float *weights, float *error_prev, int num_weights, int num_neurons){
__global__ void backprop_layer(float *input_error, float *weights, float *calculated_input, int num_weights, int num_neurons);
