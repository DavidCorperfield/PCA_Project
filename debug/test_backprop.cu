#include "../include/backprop.h"
#include <helper_cuda.h>

int main(int argc, char **argv){
    printf("testing the outer layer using back propagation\n");
    
    uint8_t *images = get_data("train-images.idx3-ubyte");
    uint8_t *labels = get_data("train-labels.idx1-ubyte");
    
    //allocate space on the host
    unsigned int size_weights = sizeof(float) * OUTPUT_LAYER_NEURONS*MAX_NUM_WEIGHTS;
    float *h_weights = (float*)malloc(size_weights);    
      
    unsigned int size_neurons = sizeof(float) * OUTPUT_LAYER_NEURONS;
    float *h_layer_input = (float*)malloc(size_neurons);
    float *h_actual_output = (float*)malloc(size_neurons);
    float *h_desired_output = (float*)malloc(size_neurons);
    float *h_error_prev = (float*)malloc(size_neurons);
    
    if(!h_error_prev or !h_layer_input or !h_actual_output or !h_weights or !h_desired_output){
        printf("unable to create host pointer\n");
        return 1;
    }
  
    //init the data for the function call
    for(int i = 0; i < OUTPUT_LAYER_NEURONS; i++){
        h_layer_input[i] = 0.5;
        h_desired_output[i] = 0;
        h_actual_output[i] = 0;
    }
    h_actual_output[0] = 0.8;
    h_actual_output[1] = 0.8;
    printf("testing with character %i \n", (int)labels[0]);
    h_desired_output[(int)labels[0]] = 0.8;

    for(int i = 0; i < OUTPUT_LAYER_NEURONS*MAX_NUM_WEIGHTS; i++){
        h_weights[i] = 0.001f;
    }
   
    //allocate space on the device
    float *d_weights;
    cudaMalloc((void**)&d_weights, size_weights);
    float *d_layer_input;
    cudaMalloc((void **)&d_layer_input, size_neurons);
    float *d_desired_output;
    cudaMalloc((void **)&d_desired_output, size_neurons);
    float *d_actual_output;
    cudaMalloc((void **)&d_actual_output, size_neurons);
    float *d_error_prev;
    cudaMalloc((void **)&d_error_prev, size_neurons);
     
    cudaError_t error;
    //copy from cpu(host) to the gpu(device)
    error = cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice); 
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_layer_input, h_layer_input, size_neurons, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    error = cudaMemcpy(d_desired_output, h_desired_output, size_neurons, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_actual_output, h_actual_output, size_neurons, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    for(int i = 0; i < OUTPUT_LAYER_NEURONS; i++){
       h_error_prev[i] = 0;
    }
    error = cudaMemcpy(d_error_prev, h_error_prev, size_neurons, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    cudaDeviceProp deviceProp;
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    }  
      
    if(deviceProp.major == 1){
        backprop_output_layer<<<512, 512>>>(MAX_NUM_WEIGHTS, OUTPUT_LAYER_NEURONS, d_layer_input, d_actual_output, d_desired_output, 
                                            d_weights, d_error_prev);
    }
    else{
        //eval_layer<<<(int)MAX_NUM_NEURONS, (int)MAX_NUM_WEIGHTS>>>(d_input, d_weights, d_output#include "../include/feed_forward.h"    
    }
    
    error = cudaGetLastError();
    printf("running eval_layer returned error code %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
    
    //read back the output values from the layer
    cudaMemcpy(h_error_prev, d_error_prev, size_neurons, cudaMemcpyDeviceToHost);  
    cudaMemcpy(h_weights, d_weights, size_neurons, cudaMemcpyDeviceToHost); 
      
    for(int i = 0; i < OUTPUT_LAYER_NEURONS; i++){
        printf("error_prev: %f\n" , h_error_prev[i]);
    }
    
    for(int i = 0; i < 20; i++){
        printf("weights: %f\n" , h_weights[i]);
    }
    
    cudaFree(d_actual_output);cudaFree(d_layer_input);cudaFree(d_desired_output);cudaFree(d_weights);
}
