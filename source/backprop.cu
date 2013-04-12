#include "../include/backprop.h"
/*TODO:
 * 
 * combine these two functions!!*/

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
    __shared__ float error[MAX_NUM_WEIGHTS];
    __shared__ float weight_error_reduce[MAX_NUM_WEIGHTS];
    
    //index for each weight into the cumulative weights array
    int neuron_weight_index = neuron_index*num_neurons + weight_index;
    
    if(weight_index < num_weights and num_neurons == NUM_OUTPUT_NEURONS and neuron_index < NUM_OUTPUT_NEURONS){
        //calculate error
        error[weight_index] = actual_output[weight_index] - desired_output[weight_index];
        //calculate partial derivative error
        error[weight_index] = (1 - actual_output[weight_index]*actual_output[weight_index]) * error[weight_index];
        //make a copy of the error before  it is corrupted so that we can calculate the error_prev
        weight_error_reduce[weight_index] = error[weight_index];
        
        /*calculate the change in the weights for this output layer*/
        //calcuate final error for finding the weight change amount
        error[weight_index] = layer_input[weight_index]*error[weight_index];
        //re-adjust weights for the current layer
        weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[weight_index];
        
        /*calculate the error for the previous layer*/
        weight_error_reduce[weight_index] = weights[neuron_weight_index]*weight_error_reduce[weight_index];
        __syncthreads();
        //now calculate errors for the previous layer
        //need to perform reduction of errors first
        for (int s = num_weights/2; s > 0; s>>=1){
            if(weight_index < s){
                weight_error_reduce[weight_index] += weight_error_reduce[weight_index + s]; 
            }
            __syncthreads();
        }
        error_prev[neuron_index] = error[0];
    }   
}


__global__ void backprop_layer(int num_weights, int num_neurons, float *actual_output, float *layer_input, float *output_error, 
                                    float *weights, float *error_prev)//output
{      
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;     
     
    //shared variable so each neuron has an error array
    __shared__ float error[MAX_NUM_WEIGHTS];
    __shared__ float weight_error_reduce[MAX_NUM_WEIGHTS];
    
    //index for each weight into the cumulative weights array
    int neuron_weight_index = neuron_index*num_neurons + weight_index;
    
    if(weight_index < num_weights and neuron_index < num_neurons){
        //calculate partial derivative error
        error[weight_index] = (1 - actual_output[weight_index]*actual_output[weight_index]) * output_error[weight_index];
        //make a copy of the error for calculating error_prev
        weight_error_reduce[weight_index] = error[weight_index];
        
        /*calcuate change in weights for current layer*/
        //calcuate final error for finding the weight change amount, will have to perform a reduction
        error[weight_index] = layer_input[weight_index]*error[weight_index];
        //re-adjust weights for the current layer
        weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[weight_index];
        
        /*calculate error for previous layer*/
        weight_error_reduce[weight_index] = weights[neuron_weight_index]*weight_error_reduce[weight_index];
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
    }   
}

__global__ void 
backprop_network(int num_threads, int num_layers, int num_weights, int num_neurons, float *input, float *outputs, float *desired_output, 
                                    float *weights)//output of the function
{    
    //make a pointer for the output layer outputs
   // float *actual_output = (float*)(outputs + (num_layers-1)*MAX_NUM_NEURONS);
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;     
    int weight_index2 = 0;
    //shared variable so each neuron has an error array
    __shared__ float error[MAX_NUM_NEURONS];
    __shared__ float weight_error_reduce[MAX_NUM_WEIGHTS];
    //if we dont have enough threads for each weight, must make another index
    if(num_threads < num_weights){
        weight_index2 = weight_index + num_threads;
    }
    int weights_per_layer = num_weights*num_neurons;
    
    
    //index for each weight into the cumulative weights array for the output layer
    int neuron_weight_index = (num_layers-1)*weights_per_layer + neuron_index*num_weights + weight_index;
    int neuron_weight_index2 = (num_layers-1)*weights_per_layer + neuron_index*num_weights + weight_index2;
    int outputs_index = (num_layers-1)*num_neurons + neuron_index;
    //only used for inner layers
    int inputs_index = 0;
    if(weight_index < num_weights and neuron_index < num_neurons){
        /************Backprop the outer layer of the network*******************/
        if(neuron_index < NUM_OUTPUT_NEURONS){
            //calculate the error for each neuron and store a value for each weight
            error[neuron_index] = outputs[outputs_index] - desired_output[neuron_index];
            //calculate partial derivative error
            error[neuron_index] = (1 - outputs[outputs_index]*outputs[outputs_index]) * error[neuron_index];
            //make a copy of the error before  it is corrupted so that we can calculate the error_prev
            weight_error_reduce[weight_index] = error[weight_index];
            //make additional copies if we didnt have enough threads
            if(weight_index2 && weight_index2 < num_weights){          
                weight_error_reduce[weight_index2] = error[weight_index2];
            }
            /*calculate the change in the weights for this output layer*/
            //calcuate final error for finding the weight change amount
            error[neuron_index] = outputs[outputs_index]*error[neuron_index];
            //re-adjust weights for the current layer
            weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[neuron_index];
            if(weight_index2 && weight_index2 < num_weights){
                weights[neuron_weight_index2] = weights[neuron_weight_index2] - LEARNING_RATE*error[neuron_index];
            }
        /*calculate the error for the previous layer*/
        weight_error_reduce[weight_index] = weights[neuron_weight_index]*weight_error_reduce[weight_index];
        if(weight_index2 && weight_index2 < num_weights){
            weight_error_reduce[weight_index2] = weights[neuron_weight_index2]*weight_error_reduce[weight_index2];
        }
        __syncthreads();
        }
        //need to perform reduction of errors
        for (int s = (NUM_OUTPUT_NEURONS)/2; s > 0; s>>=1){
            if(weight_index < s){
                weight_error_reduce[weight_index] += weight_error_reduce[weight_index + s]; 
            }
            if(weight_index2 && weight_index2 < s){
                weight_error_reduce[weight_index2] += weight_error_reduce[weight_index2 + s];
            }
            __syncthreads();
        }
        error[neuron_index] = weight_error_reduce[0];
        /*******************now backprop through the rest of the network**************************/
        //we already did the outer layer, start at layer-2.
        for(int layer = num_layers-2; layer >= 0; layer--){
            //recalculate the indexes for each thread
            neuron_weight_index = (layer)*weights_per_layer + neuron_index*num_weights + weight_index;
            neuron_weight_index2 = (layer)*weights_per_layer + neuron_index*num_weights + weight_index2;  
            outputs_index = (layer)*num_neurons + neuron_index;
            inputs_index = (layer-1)*num_neurons + neuron_index;
            
            //use the output from this layer to calculate the partial derivative of error  
            error[neuron_index] = (1 - outputs[outputs_index]*outputs[outputs_index]) * error[neuron_index];
            //make a copy of the error before  it is corrupted so that we can calculate the error_prev
            weight_error_reduce[weight_index] = error[neuron_index];
            //make additional copies if we didnt have enough threads
            if(weight_index2 && weight_index2 < num_weights){          
                weight_error_reduce[weight_index2] = error[neuron_index];
            }
            /*calculate the change in the weights for this output layer*/
            //calcuate final error for finding the weight change amount
            if(layer == 0){//if we are at the input layer, use the inputs to the network
                error[neuron_index] = input[neuron_index]*error[neuron_index];
            }
            else{//else use the outputs from the previous layer
                error[neuron_index] = outputs[inputs_index]*error[neuron_index];
            }               
            //re-adjust weights for the current layer
            weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[neuron_index];
            if(weight_index2 && weight_index2 < num_weights){
                weights[neuron_weight_index2] = weights[neuron_weight_index2] - LEARNING_RATE*error[neuron_index];
            }

            /*calculate the error for the previous layer*/
            if(layer != 0){
                weight_error_reduce[weight_index] = weights[neuron_weight_index]*weight_error_reduce[weight_index];
                if(weight_index2 && weight_index2 < num_weights){
                    weight_error_reduce[weight_index2] = weights[neuron_weight_index2]*weight_error_reduce[weight_index2];
                }
                __syncthreads();
                //need to perform reduction of errors
                for (int s = num_weights/2; s > 0; s>>=1){
                    if(weight_index < s){
                        weight_error_reduce[weight_index] += weight_error_reduce[weight_index + s]; 
                    }
                    if(weight_index2 && weight_index2 < s){
                        weight_error_reduce[weight_index2] += weight_error_reduce[weight_index2 + s];
                    }
                __syncthreads();
                }
                error[neuron_index] = weight_error_reduce[0];
            }
            __syncthreads();
                
        }
    }   
}
