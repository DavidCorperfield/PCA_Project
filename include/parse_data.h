#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

//data directory with respect to the binary
#define data_dir "./data/"

#define MAX_NUM_WEIGHTS (int)784
#define MAX_NUM_NEURONS (int)784
#define NUM_OUTPUT_NEURONS (int)10

uint8_t *get_data(char *filename);
void print_example(int start_index, uint8_t* images, uint8_t *labels);
float scale_data(uint8_t val);

