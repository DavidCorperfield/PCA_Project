#include "../include/feed_forward.h"

__global__ void 
eval_layer(float *input, float *weights, float *output, int num_weights, int num_neurons)
{
    
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;      
    
    __shared__ float output_reduce[MAX_NUM_WEIGHTS];    
    //init all the values to zero so we dont get a warning
    if(neuron_index < num_neurons && weight_index < num_weights){
        output_reduce[weight_index] = 0;        
    }
    __syncthreads();
    if(neuron_index < num_neurons && weight_index < num_weights){
        output_reduce[weight_index] = weights[neuron_index*num_neurons + weight_index]*input[weight_index];
    }
    //calculated all the weights for each neuron, so now sync
    __syncthreads();
    
    //reduce all the weights for each neuron
    for (int s = num_weights/2; s > 0; s>>=1){
        if(weight_index < s){
            output_reduce[weight_index] += output_reduce[weight_index + s]; 
        }
        __syncthreads();
    }
    
    output[neuron_index] = (float)tanh(output_reduce[0]);    
    
}

