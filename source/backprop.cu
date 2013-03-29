#include "../include/backprop.h"

//this funciton will calculate the new weights and error_prev for the previous layer
__global__ void 
backprop_output_layer(int num_weights, int num_neurons, float *layer_input, float *actual_output, float *desired_output, //inputs
                      float *weights, float *error_prev) //outputs
{    
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;     
     
    //shared variable so each neuron has an error array
    __shared__ int error[MAX_NUM_WEIGHTS];
    __shared__ int weight_error_reduce[MAX_NUM_WEIGHTS];
    
    //index for each weight for a given neuron
    int neuron_weight_index = neuron_index*num_neurons + weight_index;
    
    if(weight_index < num_weights and num_neurons == OUTPUT_LAYER_NEURONS){
        //calculate error
        error[weight_index] = actual_output[weight_index] - desired_output[weight_index];
        //calculate partial derivative error
        error[weight_index] = (1 - actual_output[weight_index]*actual_output[weight_index]) * error[weight_index];
        //calcuate final error for finding the weight change amount, will have to perform a reduction
        error[weight_index] = layer_input[weight_index]*error[weight_index];
        //make a copy of the above so that we can perform a reduction and not corrupt data
        weight_error_reduce[weight_index] = error[weight_index];

        __syncthreads();
        //now calculate errors for the previous layer
        //need to perform reduction of errors first
        for (int s = num_weights/2; s > 0; s>>=1){
            if(weight_index < s){
                weight_error_reduce[weight_index] += weight_error_reduce[weight_index + s]; 
            }
            __syncthreads();
        }
        error_prev[neuron_index] = weight_error_reduce[0];
        
        //re-adjust weights for the current layer
        weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[weight_index];
    }   
    
    
}
