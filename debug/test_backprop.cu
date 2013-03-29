#include "../include/feed_forward.h"
#include <helper_cuda.h>

int main(int argc, char **argv){
    printf("testing the feed forward network\n");
    
    uint8_t *images = get_data("train-images.idx3-ubyte");
    
    //allocate space on the host
    unsigned int size_weights = sizeof(float) * MAX_NUM_NEURONS*MAX_NUM_WEIGHTS;
    float *h_weights = (float*)malloc(size_weights);    
    
    unsigned int size_neurons = sizeof(float) * MAX_NUM_NEURONS;
    float *h_input = (float*)malloc(size_neurons);
    float *h_output = (float*)malloc(size_neurons);
    
    if(h_output == NULL){
        printf("unable to create host output pointer");
    }
  
    for(int i = 0; i < MAX_NUM_NEURONS; i++){
        h_input[i] = 1.0f;
    }

    for(int i = 0; i < MAX_NUM_NEURONS*MAX_NUM_WEIGHTS; i++){
        h_weights[i] = 0.001953125f;
    }
   
    //allocate vectors on the device
    float *d_weights;
    cudaMalloc((void**)&d_weights, size_weights);
    float *d_input;
    cudaMalloc((void **)&d_input, size_neurons);
    float *d_output;
    cudaMalloc((void **)&d_output, size_neurons);
     
    cudaError_t error;
    //copy from cpu(host) to the gpu(device)
    error = cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice); 
     if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_input, h_input, size_neurons, cudaMemcpyHostToDevice);
     if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
 
    error = cudaMemcpy(d_output, h_output, size_neurons, cudaMemcpyHostToDevice);
    printf("The number of neurons is : %i\n", (int)MAX_NUM_NEURONS);
    printf("The number of weights is : %i\n", (int)MAX_NUM_WEIGHTS);
    //evaluate the test layer   
    
   for(int i = 0; i < 20; i++){
       h_output[i] = 12;
    }
    error = cudaMemcpy(d_output, h_output, size_neurons, cudaMemcpyHostToDevice);
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    cudaDeviceProp deviceProp;
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    }    
    if(deviceProp.major == 1){
        eval_layer<<<512, 512>>>(d_input, d_weights, d_output, 512, 512);
    }
    else{
        //eval_layer<<<(int)MAX_NUM_NEURONS, (int)MAX_NUM_WEIGHTS>>>(d_input, d_weights, d_output#include "../include/feed_forward.h"
#include <helper_cuda.h>

int main(int argc, char **argv){
    printf("testing the feed forward network\n");
    
    //uint8_t *images = get_data("train-images.idx3-ubyte");
    
    //allocate space on the host
    unsigned int size_weights = sizeof(float) * MAX_NUM_NEURONS*MAX_NUM_WEIGHTS;
    float *h_weights = (float*)malloc(size_weights);    
    
    unsigned int size_neurons = sizeof(float) * MAX_NUM_NEURONS;
    float *h_input = (float*)malloc(size_neurons);
    float *h_output = (float*)malloc(size_neurons);
    
    if(h_output == NULL){
        printf("unable to create host output pointer");
    }
  
    for(int i = 0; i < MAX_NUM_NEURONS; i++){
        h_input[i] = 1.0f;
    }

    for(int i = 0; i < MAX_NUM_NEURONS*MAX_NUM_WEIGHTS; i++){
        h_weights[i] = 0.001953125f;
    }
   
    //allocate vectors on the device
    float *d_weights;
    cudaMalloc((void**)&d_weights, size_weights);
    float *d_input;
    cudaMalloc((void **)&d_input, size_neurons);
    float *d_output;
    cudaMalloc((void **)&d_output, size_neurons);
     
    cudaError_t error;
    //copy from cpu(host) to the gpu(device)
    error = cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice); 
     if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_input, h_input, size_neurons, cudaMemcpyHostToDevice);
     if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
 
    error = cudaMemcpy(d_output, h_output, size_neurons, cudaMemcpyHostToDevice);
    printf("The number of neurons is : %i\n", (int)MAX_NUM_NEURONS);
    printf("The number of weights is : %i\n", (int)MAX_NUM_WEIGHTS);
    //evaluate the test layer   
    
   for(int i = 0; i < 20; i++){
       h_output[i] = 12;
    }
    error = cudaMemcpy(d_output, h_output, size_neurons, cudaMemcpyHostToDevice);
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    cudaDeviceProp deviceProp;
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    }    
    if(deviceProp.major == 1){
        eval_layer<<<512, 512>>>(d_input, d_weights, d_output, 512, 512);
    }
    else{
        //eval_layer<<<(int)MAX_NUM_NEURONS, (int)MAX_NUM_WEIGHTS>>>(d_input, d_weights, d_output);
    }
    error = cudaGetLastError();
    printf("running eval_layer returned error code %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
    
    //read back the output values from the layer
    cudaMemcpy(h_output, d_output, size_neurons, cudaMemcpyDeviceToHost);  
      
    for(int i = 0; i < 20; i++){
        printf("test: %f\n" , h_output[i]);
    }
    cudaFree(d_output);cudaFree(d_input);cudaFree(d_weights);
    return 0;
    
}
);
    }
    error = cudaGetLastError();
    printf("running eval_layer returned error code %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
    
    //read back the output values from the layer
    cudaMemcpy(h_output, d_output, size_neurons, cudaMemcpyDeviceToHost);  
      
    for(int i = 0; i < 20; i++){
        printf("test: %f\n" , h_output[i]);
    }
    cudaFree(d_output);cudaFree(d_input);cudaFree(d_weights);
    return 0;
    
}
