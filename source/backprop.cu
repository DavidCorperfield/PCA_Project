#include "../include/backprop.h"

//this funciton will calculate the new weights and error_prev for the previous layer
__global__ void 
backprop_output_layer(int num_threads, int num_layers, int num_weights, int num_neurons, float *outputs, float *desired_output, //inputs
                      float *weights, float *error_prev) //outputs
{    
    //make a pointer for the output layer outputs
    //float *actual_output = (float*)(outputs + (num_layers-1)*MAX_NUM_NEURONS);
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;     
    int weight_index2 = 0;
    //shared variable so each neuron has an error array
    __shared__ float error[MAX_NUM_NEURONS];
    __shared__ float weight_error_reduce[SHARED_ARRAY_SIZE];
     __syncthreads();
    for (int i = weight_index; i < SHARED_ARRAY_SIZE; i+=num_threads){
        weight_error_reduce[i]=0;
    }
    __syncthreads();
    //if we dont have enough threads for each weight, must make another index
    if(num_threads < num_weights){
        weight_index2 = weight_index + num_threads;
    }    
    //index for each weight into the cumulative weights array for the output layer
    int neuron_weight_index = (num_layers-1)*num_weights*MAX_NUM_NEURONS + neuron_index*num_weights + weight_index;
    int neuron_weight_index2 = (num_layers-1)*num_weights*MAX_NUM_NEURONS + neuron_index*num_weights + weight_index2;
    int outputs_index = (num_layers-1)*MAX_NUM_NEURONS + neuron_index;
    int outputs_index2 = 0;
    //only used for inner layers
    __syncthreads();
    if(weight_index < num_weights and neuron_index < num_neurons){
        /************Backprop the outer layer of the network*******************/
        if(neuron_index < NUM_OUTPUT_NEURONS and weight_index == 0){
            //calculate the error for each neuron and store a value for each weight
            error[neuron_index] = outputs[outputs_index] - desired_output[neuron_index];
            //calculate partial derivative error
            error[neuron_index] = (1 - outputs[outputs_index]*outputs[outputs_index]) * error[neuron_index];
            if(error[neuron_index] == 0){
                //error[neuron_index] = 0.1;
            }
        }
        
        if(neuron_index < NUM_OUTPUT_NEURONS){
            //make a copy of the error before  it is corrupted so that we can calculate the error_prev
            weight_error_reduce[weight_index] = error[neuron_index];
            //make additional copies if we didnt have enough threads
            if(weight_index2 && weight_index2 < NUM_OUTPUT_NEURONS){          
                weight_error_reduce[weight_index2] = error[neuron_index];
            }
            __syncthreads();
            //need to use outputs from the previous layer to calculate weight change amount
            outputs_index = (num_layers-2)*MAX_NUM_NEURONS + weight_index;
            outputs_index2 = (num_layers-2)*MAX_NUM_NEURONS + weight_index2;
            /*calculate the change in the weights for this output layer*/
            //calcuate final error for finding the weight change amount
            if(weight_index < num_weights){
                error[weight_index] = outputs[outputs_index]*weight_error_reduce[neuron_index];
                if(weight_index2 && weight_index2 < num_weights){
                    error[weight_index2] = outputs[outputs_index2]*weight_error_reduce[neuron_index];
                }
            }
            
        }
        
        __syncthreads();
        /*calculate the error for the previous layer*/
        if(weight_index < NUM_OUTPUT_NEURONS){
            weight_error_reduce[weight_index] *= weights[neuron_weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                weight_error_reduce[weight_index2] *= weights[neuron_weight_index2];
            }
            __syncthreads();
        }
        
        //need to perform reduction of errors
        for (int s = (int)(SHARED_ARRAY_SIZE)/2; s > 0; s>>=1){
            if(weight_index < s){
                weight_error_reduce[weight_index] += weight_error_reduce[weight_index + s]; 
            }
            if(weight_index2 && weight_index2 < s){
                weight_error_reduce[weight_index2] += weight_error_reduce[weight_index2 + s];
            }
            __syncthreads();
        }
        if(weight_index == 0){
             error_prev[neuron_index] = weight_error_reduce[0];
        }
        __syncthreads();
        
        if(neuron_index < NUM_OUTPUT_NEURONS){
            //re-adjust weights for the current layer
            weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                weights[neuron_weight_index2] = weights[neuron_weight_index2] - LEARNING_RATE*error[weight_index2];
            }
        }
    }
}


__global__ void 
backprop_layer(int num_threads, int layer, int num_weights, int num_neurons, float *outputs, float *input, //inputs
                      float *weights, float *error_prev)//output
{      
    //each block will correspond to one neuron
    int neuron_index = blockIdx.x;
    
    //each thread will correspond to a weight of a neuron
    int weight_index = threadIdx.x;     
    int weight_index2 = 0;
    //shared variable so each neuron has an error array
    __shared__ float error[SHARED_ARRAY_SIZE];
    __shared__ float weight_error_reduce[SHARED_ARRAY_SIZE];
     __syncthreads();
    for (int i = weight_index; i < SHARED_ARRAY_SIZE; i+=num_threads){
        weight_error_reduce[i]=0;
        error[i] = 0;
    }
    __syncthreads();
    //if we dont have enough threads for each weight, must make another index
    if(num_threads < num_weights){
        weight_index2 = weight_index + num_threads;
    } 
    //index for each weight into the cumulative weights array
    int neuron_weight_index = layer*num_neurons*num_weights + neuron_index*num_neurons + weight_index;
    int neuron_weight_index2 = layer*num_neurons*num_weights + neuron_index*num_neurons + weight_index2;
    int outputs_index = layer*num_neurons + neuron_index;
    int outputs_index2 = 0;
    __syncthreads();
    
    if(weight_index < num_weights and neuron_index < num_neurons){
        //calculate partial derivative error, only 1 thread needed
        if(neuron_index < num_neurons and weight_index == 0){
            error[neuron_index] = (1 - outputs[outputs_index]*outputs[outputs_index]) * error_prev[neuron_index];
        }
        //make a copy of the error before  it is corrupted so that we can calculate the error_prev
        weight_error_reduce[weight_index] = error[neuron_index];
        if(weight_index2 && weight_index2 < NUM_OUTPUT_NEURONS){          
            weight_error_reduce[weight_index2] = error[neuron_index];
        }
        __syncthreads();

        outputs_index = (layer-1)*num_neurons + weight_index;
        outputs_index = (layer-1)*num_neurons + weight_index2;
        //calcuate final error for finding the weight change amount, will have to perform a reduction
        if(layer > 0){
            error[weight_index2] = outputs[outputs_index]*weight_error_reduce[neuron_index];
            if(weight_index2 && weight_index2 < num_weights){
                error[weight_index2] = outputs[outputs_index2]*weight_error_reduce[neuron_index];
            }
        }
        else{//use the inputs since its the first layer
            error[weight_index] = input[weight_index]*weight_error_reduce[neuron_index];
            if(weight_index2 && weight_index2 < num_weights){
                error[weight_index2] = input[weight_index2]*weight_error_reduce[neuron_index];
            }
        }
        __syncthreads();
                
        /*****calculate error for previous layer******/
        if(weight_index < num_weights){
            weight_error_reduce[weight_index] *= weights[neuron_weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                weight_error_reduce[weight_index2] *= weights[neuron_weight_index2];
            }
        }
        __syncthreads();
        //need to perform reduction of errors
        for (int s = (int)SHARED_ARRAY_SIZE/2; s > 0; s>>=1){
            if(weight_index < s){
                weight_error_reduce[weight_index] += weight_error_reduce[weight_index + s]; 
            }
            if(weight_index2 && weight_index2 < s){
                weight_error_reduce[weight_index2] += weight_error_reduce[weight_index2 + s];
            }
            __syncthreads();
        }
        if(weight_index == 0)
            error_prev[neuron_index] = weight_error_reduce[0];
        
        /*****re-adjust weights for the current layer******/
        weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[weight_index];
        if(weight_index2 && weight_index2 < num_weights){
            weights[neuron_weight_index2] = weights[neuron_weight_index2] - LEARNING_RATE*error[weight_index2];
        }    
    }
}

__global__ void 
backprop_network(int num_threads, int num_layers, int num_weights, int num_neurons, float *input, float *outputs, float *desired_output, 
                                    float *weights)//output of the function
{    
    //make a pointer for the output layer outputs
    //float *actual_output = (float*)(outputs + (num_layers-1)*MAX_NUM_NEURONS);
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
    //index for each weight into the cumulative weights array for the output layer
    int neuron_weight_index = (num_layers-1)*num_weights*num_neurons + neuron_index*num_weights + weight_index;
    int outputs_index = (num_layers-1)*num_neurons + neuron_index;
    //only used for inner layers
    int inputs_index = 0;
    if(weight_index < num_weights and neuron_index < num_neurons){
        /************Backprop the outer layer of the network*******************/
        if(neuron_index < NUM_OUTPUT_NEURONS and weight_index == 0){
            //calculate the error for each neuron and store a value for each weight
            error[neuron_index] = outputs[outputs_index] - desired_output[neuron_index];
            //calculate partial derivative error
            error[neuron_index] = (1 - outputs[outputs_index]*outputs[outputs_index]) * error[neuron_index];
            if(error[neuron_index] == 0){
                //error[neuron_index] = 0.1;
            }
        }
        //previous layer will only have 10 weights for all neurons leading to output
        if(neuron_index < NUM_OUTPUT_NEURONS){
            //make a copy of the error before  it is corrupted so that we can calculate the error_prev
            weight_error_reduce[weight_index] = error[neuron_index];
            //make additional copies if we didnt have enough threads
            if(weight_index2 && weight_index2 < NUM_OUTPUT_NEURONS){          
                weight_error_reduce[weight_index2] = error[neuron_index];
            }
            
            //need to use outputs from the previous layer to calculate weight change amount
            outputs_index = (num_layers-2)*num_neurons + neuron_index;
            /*calculate the change in the weights for this output layer*/
            //calcuate final error for finding the weight change amount
            error[neuron_index] = outputs[outputs_index]*error[neuron_index];
            //re-adjust weights for the current layer
            weights[neuron_weight_index] = weights[neuron_weight_index] - LEARNING_RATE*error[neuron_index];
            if(weight_index2 && weight_index2 < num_weights){
                weights[neuron_weight_index+weight_index2] = weights[neuron_weight_index+weight_index2] - LEARNING_RATE*error[neuron_index];
            }
        }
        /*calculate the error for the previous layer*/
        if(weight_index < NUM_OUTPUT_NEURONS){
            weight_error_reduce[weight_index] = weights[neuron_weight_index]*weight_error_reduce[weight_index];
            if(weight_index2 && weight_index2 < num_weights){
                weight_error_reduce[weight_index2] = weights[neuron_weight_index+weight_index2]*weight_error_reduce[weight_index2];
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
        //we already did the outer layer, start at num_layer-2.
        for(int layer = num_layers-2; layer >= 0; layer--){
            //recalculate the indexes for each thread
            neuron_weight_index = (layer)*num_weights*num_neurons + neuron_index*num_weights + weight_index; 
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
                weights[neuron_weight_index+weight_index2] = weights[neuron_weight_index+weight_index2] - LEARNING_RATE*error[neuron_index];
            }

            /*calculate the error for the previous layer*/
            if(layer != 0){
                weight_error_reduce[weight_index] = weights[neuron_weight_index]*weight_error_reduce[weight_index];
                if(weight_index2 && weight_index2 < num_weights){
                    weight_error_reduce[weight_index2] = weights[neuron_weight_index+weight_index2]*weight_error_reduce[weight_index2];
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
