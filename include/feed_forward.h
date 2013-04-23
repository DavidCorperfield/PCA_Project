#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include "parse_data.h"

__global__ void eval_layer(int num_threads, int layer, int num_weights, int num_neurons, float *input, float *weights, float *output);
__global__ void eval_last_layer(int num_threads, int layer, int num_weights, int num_neurons, float *weights, float *outputs);


__global__ void eval_network(int num_threads, int num_layers, int num_weights, int num_neurons, float *input, float *weights, float *outputs);
