#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <cuda.h>


#define MAX_NUM_WEIGHTS (int)784
#define MAX_NUM_NEURONS (int)784
#define NUM_OUTPUT_NEURONS (int)10
#define SHARED_ARRAY_SIZE int(1024)

uint8_t *get_data(char *filename);
float *get_data_f(char *filename);
void print_example(int start_index, uint8_t* images, uint8_t *labels);
void get_norm_image(float *norm_img, uint8_t *images, int img_num);
void get_input_image(float *images, float *input, int img_num);
float scale_data(uint8_t val);
int get_final_output(float *outputs, int size);
