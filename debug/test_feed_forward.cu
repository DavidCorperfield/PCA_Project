#include "../include/feed_forward.h"
#include <helper_cuda.h>

int main(int argc, char **argv){
    printf("testing the feed forward network\n");
    
    //uint8_t *images = get_data("train-images.idx3-ubyte");
    
    //allocate space on the host
    int size_weights = sizeof(float) * NUM_NEURONS*NUM_WEIGHTS;
    float *h_weights = (float*)malloc(size_weights);    
    
    int size_neurons = sizeof(float) * NUM_NEURONS;
    float *h_input = (float*)malloc(size_neurons);
    float *h_output = (float*)malloc(size_neurons);
  
    for(int i = 0; i < NUM_NEURONS; i++){
        h_input[i] = 1;
    }

    for(int i = 0; i < NUM_NEURONS*NUM_WEIGHTS; i++){
        h_weights[i] = 1;
    }
   
    //allocate vectors on the device
    float *d_weights;
    cudaMalloc(&d_weights, size_weights);
    float *d_input;
    cudaMalloc(&d_input, size_neurons);
    float *d_output;
    cudaMalloc(&d_output, size_neurons);
     
    cudaError_t error;
    //copy from cpu(host) to the gpu(device)
    error = cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_input, h_input, size_neurons, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    //evaluate the test layer
    eval_layer<<<NUM_NEURONS, NUM_WEIGHTS>>>(d_input, d_weights, d_output);
    
    //read back the output values from the layer
    cudaMemcpy(h_output, d_output, size_neurons, cudaMemcpyDeviceToHost);    
    
    printf("test: %f\n" , h_output[0]);
    cudaFree(&d_output);cudaFree(&d_input);cudaFree(d_weights);
    return 0;
    
}
