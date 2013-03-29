#include "../include/backprop.h"

__global__ void  
backprop_output_layer(float *layer_input, float *actual_output, float *desired_output, float *weights, float *error_prev, int num_weights, int num_neurons){
    
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;      
    
    __shared__ int error[MAX_NUM_WEIGHTS];
    
    if(weight_index < num_weights and num_neurons == OUTPUT_LAYER_NEURONS){
        //calculate error
        error[weight_index] = actual_output[weight_index] - desired_output[weight_index]);
        //calculate partial derivative error
        error[weight_index] = (1 - layer_output[weight_index]^2) * error[weight_index];
        //calcuate final error for finding the weight change amount, will have to perform a reduction
        error[weight_index] = layer_input[weight_index]*error[weight_index];
        //now calculate errors for the previous layer
        error_prev[weight_index] = weights[weight_index]*error[weight_index];
        
        //re-adjust weights for the current layer
        weights[weight_index] = weights[weight_index] - LEARNING_RATE*error[weight_index];
    }
    
    
    
}
