#include "../include/feed_forward.h"


__global__ void 
eval_layer(float *input, float *weights, float *output)
{
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;      
    
    __shared__ float output_reduce[NUM_WEIGHTS];    
    //init all the values to zero so we dont get a warning
    output_reduce[weight_index] = 0;
        
    if(neuron_index < NUM_NEURONS && weight_index < NUM_WEIGHTS){
        output_reduce[weight_index] += weights[neuron_index*NUM_NEURONS + weight_index]*input[weight_index];
        
    }
    //calculated all the weights for each neuron, so now sync
    __syncthreads();
    
    //reduce all the weights for each neuron
    for (int s = NUM_WEIGHTS/2; s > 0; s>>=1){
        if(weight_index < s){
            output_reduce[weight_index] += output_reduce[weight_index + s]; 
        }
        __syncthreads();
    }
    //each block will calculate its output value
    output[neuron_index] = atanhf(output_reduce[0]);
    
    
}

