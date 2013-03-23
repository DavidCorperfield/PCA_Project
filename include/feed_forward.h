#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include "parse_data.h"

#define NUM_NEURONS 28*28
#define NUM_WEIGHTS (28*28)

__global__ void eval_layer(float *input, float *weights, float *output);

