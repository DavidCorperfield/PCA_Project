#include "../include/backprop.h"
#include "../include/feed_forward.h"
#include "../include/parse_data.h"
#include <string.h>
#include <helper_cuda.h>
#include <cuda.h>

int main(int argc, char **argv){
    printf("testing the outer layer using back propagation\n");
    
    uint8_t *images = get_data("train-images.idx3-ubyte");
    uint8_t *test_images = get_data("t10k-images.idx3-ubyte");
    uint8_t *labels = get_data("train-labels.idx1-ubyte");
    uint8_t *test_labels = get_data("t10k-labels.idx1-ubyte");
    
    //uint8_t *labels = images;    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    int num_layers = 3;
    int size_weights = (int)sizeof(float)*MAX_NUM_NEURONS*MAX_NUM_WEIGHTS*num_layers;
    //array to hold EVERY weight value of the network
    float *h_weights = (float*)malloc((size_t)size_weights);    
    
    int size_outputs = (int)sizeof(float)*MAX_NUM_NEURONS*num_layers;
    float *h_outputs = (float*)malloc((size_t)size_outputs);
    
    int size_input = (int)sizeof(float)*MAX_NUM_NEURONS;
    float *h_input = (float*)malloc((size_t)size_input);
    float *h_input2 = (float*)malloc((size_t)size_input);
    float *h_input3 = (float*)malloc((size_t)size_input);
    
    if(h_outputs == NULL){
        printf("unable to create host output pointer");
    }
    /*for(int i = 0; i < MAX_NUM_NEURONS; i++){
        h_input[i] = (((float)images[i]*1.6)/255.0)-0.8;//scale the input data to -0.8 to 0.8
        if(h_input[i] > 0)
            h_input[i] = 0.8;
        else
            h_input[i] = -0.8;
        //printf("%f ", h_input[i]);
    }*/
    get_norm_image(h_input, images, 0);
    srand(time(NULL));
    //print_example(0, images, labels);
    for(int i = 0; i < MAX_NUM_NEURONS*MAX_NUM_WEIGHTS*num_layers; i++){
        h_weights[i] = (float)rand()/(float)RAND_MAX*(2*(1/sqrt(MAX_NUM_NEURONS))) - (float)1/sqrt(MAX_NUM_NEURONS);
    }
   
    //allocate vectors on the device
    cudaError_t error;
    float *d_weights;
    error = cudaMalloc((void**)&d_weights, size_weights);
     if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }
    float *d_input;
    error = cudaMalloc((void **)&d_input, size_input);
     if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }
    float *d_outputs;
    error = cudaMalloc((void **)&d_outputs, size_outputs);
     if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }
    //copy from cpu(host) to the gpu(device)
    error = cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice); 
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }
    for(int j = 0; j < size_outputs/sizeof(float);j++){
        h_outputs[j] = (float)0.1;
    }
    error = cudaMemcpy(d_outputs, h_outputs, size_outputs, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }    
    
    ///***********Backprop allocations*************/
    int size_network_output = (int)sizeof(float)*(int)NUM_OUTPUT_NEURONS;
    float *h_desired_output = (float*)malloc((size_t)size_network_output);
    
    //set all character outputs to false, or -0.8
    //for(int i = 0; i < NUM_OUTPUT_NEURONS; i++){
    //    h_desired_output[i] = -0.8;
    //}
    //set the desired output for the first hand written character to 0.8
   // h_desired_output[(int)labels[0]] = 0.8;
   // printf("testing with character %i \n", (int)labels[0]);
   // for(int i = 0; i < NUM_OUTPUT_NEURONS; i++){
           // printf("%f", h_desired_output[i]);
   // }
    
    float *d_desired_output;
    error = cudaMalloc((void**)&d_desired_output, (size_t)size_network_output);
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }
    //error = cudaMemcpy(d_desired_output, h_desired_output, size_network_output, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }
    
    float *d_error_prev;
    float *h_error_prev = (float*)malloc((size_t)size_input);
    error = cudaMalloc((void**)&d_error_prev, (size_t)size_input);
    if (error != cudaSuccess){
        printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

/*************************************************************************************************************************************/
    if(deviceProp.major == 1){
        int epoch; int j;
        for(epoch = 0; epoch < 2; epoch++){
            printf("epoch num: %i\n", epoch);
            for(j = 0; j < 60000; j++) {
               // printf("img is %i\n", labels[j]);
                /******prepare data for next loop*******/
               get_norm_image(h_input, images, j);
               error = cudaMemcpy(d_input, h_input, (size_t)size_input, cudaMemcpyHostToDevice);
                if (error != cudaSuccess){
                    printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
                    exit(EXIT_FAILURE);
                }
                //set all character outputs to false, or -0.8
                for(int i = 0; i < NUM_OUTPUT_NEURONS; i++){
                    h_desired_output[i] = -0.8;
                }
                //set the desired output for the first hand written character to 0.8
                h_desired_output[(int)labels[j]] = 0.8;
                //printf("desired output is %i\n", (int)labels[1]);
                error = cudaMemcpy(d_desired_output, h_desired_output, size_network_output, cudaMemcpyHostToDevice);
                if (error != cudaSuccess){
                    printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
                    exit(EXIT_FAILURE);
                }
                
                for(int l = 0; l < num_layers-1; l++){
                    eval_layer<<<784, 512>>>(512, l, 784, 784, d_input, d_weights, d_outputs); 
                    cudaDeviceSynchronize();
                }
                eval_layer<<<784, 512>>>(512, num_layers-1, 784, 10, d_input, d_weights, d_outputs); 
                cudaDeviceSynchronize();
                
                backprop_output_layer<<<784, 512>>>(512, num_layers, 784, 784, d_outputs, d_desired_output, d_weights, d_error_prev); 
                cudaDeviceSynchronize();
                
                for(int l = num_layers-2; l >= 0; l--){
                    backprop_layer<<<784, 512>>>(512, l, 784, 784, d_outputs, d_input, d_weights, d_error_prev); 
                    cudaDeviceSynchronize();

                }
                
                //read back the output values from the layer
                cudaMemcpy(h_weights, d_weights, size_weights, cudaMemcpyDeviceToHost);
                //cudaMemcpy(h_desired_output, d_desired_output, size_network_output, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_outputs, d_outputs, size_outputs, cudaMemcpyDeviceToHost);  
                cudaMemcpy(h_error_prev, d_error_prev, size_input, cudaMemcpyDeviceToHost);  
                error = cudaGetLastError();
                if(strcmp(cudaGetErrorString(error),"no error"))
                  //  printf("running eval_network returned error code %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
                // for(int i = 0; i < 20; i++){
                for(int i = (num_layers-1)*MAX_NUM_NEURONS*MAX_NUM_WEIGHTS; i <(num_layers-1)*MAX_NUM_NEURONS*MAX_NUM_WEIGHTS+20; i++){
                    //printf("weight%i: %f\n" , i, h_weights[i]);
                    
                }
                //for(int i = 0; i < 10; i++){
                for(int i = (num_layers-1)*MAX_NUM_NEURONS; i < (num_layers-1)*MAX_NUM_NEURONS+NUM_OUTPUT_NEURONS; i++){
                //    printf("output%i: %f\n" , i-(num_layers-1)*MAX_NUM_NEURONS, h_outputs[i]);
                //     printf("output%i: %f\n" , i, h_desired_output[i]);
                    }
                //printf("\n");
                for(int i = 0; i < 10; i++){
                    //printf("error%i: %f\n" , i, h_error_prev[i]);
                //     printf("output%i: %f\n" , i, h_desired_output[i]);
                    }
                
                }
        }
        //test the network
        int final_output; int errors = 0;
        float current_max;
        int test_loops = 100;
        for(int j=1; j < test_loops; j++){
            get_norm_image(h_input, test_images, j);
            //get_norm_image(h_input2, images, j-1);
            error = cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
            if (error != cudaSuccess){
                printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
                exit(EXIT_FAILURE);
            }
            //set all character outputs to false, or -0.8
            for(int i = 0; i < NUM_OUTPUT_NEURONS; i++){
                h_desired_output[i] = -0.8;
            }
            //set the desired output for the first hand written character to 0.8
            h_desired_output[(int)test_labels[j]] = 0.8;
            //printf("the desired output is %i\n", test_labels[j]);
            error = cudaMemcpy(d_desired_output, h_desired_output, size_network_output, cudaMemcpyHostToDevice);
            if (error != cudaSuccess){
                printf("cudaMalloc d_A returned error %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
                exit(EXIT_FAILURE);
            }

           for(int l = 0; l < num_layers-1; l++){
                    eval_layer<<<784, 512>>>(512, l, 784, 784, d_input, d_weights, d_outputs); 
                    cudaDeviceSynchronize();
                }
            eval_layer<<<784, 512>>>(512, num_layers-1, 784, 10, d_input, d_weights, d_outputs); 
            cudaDeviceSynchronize();
           
            cudaMemcpy(h_outputs, d_outputs, size_outputs, cudaMemcpyDeviceToHost);  
            error = cudaGetLastError();
            //printf("running eval_network returned error code %s, line(%d)\n", cudaGetErrorString(error), __LINE__);
            current_max = -10;
            for(int k = (int)(num_layers-1)*MAX_NUM_NEURONS; k < (int)(num_layers-1)*MAX_NUM_NEURONS+NUM_OUTPUT_NEURONS; k++){
                //printf("k = %i and output = %f \n", k-MAX_NUM_NEURONS, h_outputs[k]);
                if((float)h_outputs[k] > current_max){
                    current_max = h_outputs[k];
                    final_output = (int)k-(num_layers-1)*MAX_NUM_NEURONS;
                }
            }
            if (final_output != test_labels[j]){
                errors += 1;
                printf("output should be: %i      out was: %i\n", test_labels[j], final_output);
            }        
        }
        printf("there were %i errors which makes a percent correct of %f %%\n", errors, 100*(float)(test_loops-errors)/test_loops);                                           
/*************************************************************************************************************************************/
    }
        return 0;
        //cudaFree(d_actual_output);cudaFree(d_layer_input);cudaFree(d_desired_output);cudaFree(d_weights);
}
