#include "../include/feed_forward.h"

//this function will determine the output of a signle layer in the network
//needs to have the number of block and threads equal to neurons and weights respectively
__global__ void 
eval_layer(int num_weights, int num_neurons, float *input, float *weights, float *output)
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

__global__ void 
eval_network(int num_threads, int num_layers, int num_weights, int num_neurons, float *input, float *weights, 
             float *outputs){ //output of the network
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;
    //we will need two indexes if there are not enough threads to execute each weight
    int weight_index2 = 0;
    if(num_threads < num_weights){
        weight_index2 = weight_index + num_threads;
    }
    __shared__ float output_reduce[MAX_NUM_WEIGHTS]; 
    __shared__ float layer_input[MAX_NUM_WEIGHTS];   
    //init all the output values to zero and shared input to the input values
    if(neuron_index < num_neurons && weight_index < num_weights){
        output_reduce[weight_index] = 0;
        layer_input[weight_index] = input[weight_index];
        if(weight_index2 && weight_index2 < num_weights){
            output_reduce[weight_index2] = 0;
            layer_input[weight_index2] = input[weight_index2];
        }
    }
    __syncthreads();
    //evaluate all the input and hidden layers
    for(int layer = 0; layer < num_layers-1; layer++){
        if(neuron_index < num_neurons && weight_index < num_weights){
            output_reduce[weight_index] = weights[num_neurons*layer + num_weights*neuron_index + weight_index]*layer_input[weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                output_reduce[weight_index2] = weights[num_neurons*layer + num_weights*neuron_index + weight_index2]*layer_input[weight_index2];
            }
        }
        //calculated all the weights for each neuron, so now sync
        __syncthreads();
        //reduce all the weights for each neuron
        for (int s = num_weights/2; s > 0; s>>=1){
            if(weight_index < s){
                output_reduce[weight_index] += output_reduce[weight_index + s]; 
            }
            if(weight_index2 && weight_index2 < s){
                output_reduce[weight_index2] += output_reduce[weight_index2 + s]; 
            }
            __syncthreads();
        }
        //set the next layers input the the current layers output
        layer_input[neuron_index] = (float)tanh(output_reduce[0]);
        if(layer_input[neuron_index] <= -1)
            layer_input[neuron_index] = (float)-0.8;
        else if(layer_input[neuron_index] >= 1)
            layer_input[neuron_index] = (float)0.8;
        __syncthreads();
        //put a copy of each layers output in the outputs array, used for training
        outputs[layer*MAX_NUM_NEURONS + neuron_index] = layer_input[neuron_index];
       
    }
    
    /****************now evaluate the last layer***************************/
    if(neuron_index < NUM_OUTPUT_NEURONS && weight_index < num_weights){
        output_reduce[weight_index] = weights[num_neurons*num_layers + num_weights*neuron_index + weight_index]*layer_input[weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                output_reduce[weight_index2] = weights[num_neurons*(num_layers) + num_weights*neuron_index + weight_index2]*layer_input[weight_index2];
            }
        }
        //calculated all the weights for each neuron, so now sync
        __syncthreads();
        //reduce all the weights for each neuron
        for (int s = num_weights/2; s > 0; s>>=1){
            if(weight_index < s){
                output_reduce[weight_index] += output_reduce[weight_index + s]; 
            }
            if(weight_index2 && weight_index2 < s){
                output_reduce[weight_index2] += output_reduce[weight_index2 + s]; 
            }
            __syncthreads();
        }
        outputs[(num_layers-1)*MAX_NUM_NEURONS + neuron_index] = (float)tanh(output_reduce[0]);
        if(outputs[(num_layers-1)*MAX_NUM_NEURONS + neuron_index] <= -0.8)
            outputs[(num_layers-1)*MAX_NUM_NEURONS + neuron_index] = (float)-0.8;
        else if(outputs[(num_layers-1)*MAX_NUM_NEURONS + neuron_index] >= 1)
            outputs[(num_layers-1)*MAX_NUM_NEURONS + neuron_index] = (float)0.8;
}


