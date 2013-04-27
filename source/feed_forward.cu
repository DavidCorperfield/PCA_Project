#include "../include/feed_forward.h"

//this function will determine the output of a signle layer in the network
//needs to have the number of block and threads equal to neurons and weights respectively
__global__ void eval_layer(int num_threads, int layer, int num_weights, int num_neurons, float *input, float *weights, float *outputs)
{
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;
    //we will need two indexes if there are not enough threads to execute each weight
    int weight_index2 = 0;
    if(num_threads < num_weights){
        weight_index2 = weight_index + num_threads;
    }
    __shared__ float output_reduce[SHARED_ARRAY_SIZE]; 
    __syncthreads();
    for (int i = weight_index; i < SHARED_ARRAY_SIZE; i+=num_threads){
        output_reduce[i]=0;
    }
    __syncthreads();
    //__shared__ float layer_input[MAX_NUM_WEIGHTS];   
    int neuron_weight_index = num_weights*MAX_NUM_NEURONS*layer + num_weights*neuron_index + weight_index;
    int neuron_weight_index2 = num_weights*MAX_NUM_NEURONS*layer + num_weights*neuron_index + weight_index2;
    if(neuron_index < num_neurons && weight_index < num_weights){
        if(layer == 0){//first layer so read from the input data
            output_reduce[weight_index] = weights[neuron_weight_index]*input[weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                output_reduce[weight_index2] = weights[neuron_weight_index2]*input[weight_index2];
            }
        }
        else{
            output_reduce[weight_index] = weights[neuron_weight_index]*outputs[(layer-1)*MAX_NUM_NEURONS+weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                output_reduce[weight_index2] = weights[neuron_weight_index2]*outputs[(layer-1)*MAX_NUM_NEURONS+weight_index2];
            }
        }
        //calculated all the weights for each neuron, so now sync
        __syncthreads();
        //reduce all the weights for each neuron
        for (int s = (int)SHARED_ARRAY_SIZE/2; s > 0; s>>=1) {
            if(weight_index < s){
                output_reduce[weight_index] += output_reduce[weight_index + s]; 
            }
            if(weight_index2 && weight_index2 < s){
                output_reduce[weight_index2] += output_reduce[weight_index2 + s]; 
            }
            __syncthreads();
        }
         __syncthreads();
     if(weight_index == 0){
            //set the next layers input the the current layers output
            //outputs[layer*MAX_NUM_NEURONS + neuron_index] = output_reduce[0];
            outputs[layer*MAX_NUM_NEURONS + neuron_index] = (float)tanh(output_reduce[0]);
            if(outputs[layer*MAX_NUM_NEURONS + neuron_index] <= (float)-0.8){
                outputs[layer*MAX_NUM_NEURONS + neuron_index] = (float)-0.8;
            }
            else if(outputs[layer*MAX_NUM_NEURONS + neuron_index] >= (float) 0.8){
                outputs[layer*MAX_NUM_NEURONS + neuron_index] = (float)0.8;
            }
        }
        __syncthreads();
    }
}



