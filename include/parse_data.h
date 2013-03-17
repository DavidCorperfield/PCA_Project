#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
 
//data directory with respect to the binary
#define data_dir "./data/"

int num_items = 0;
int num_images = 0;

uint8_t *get_data(char *filename);
void print_example(int start_index, uint8_t* images, uint8_t *labels);
