#include "../include/parse_data.h"
#include <byteswap.h>

//num from 0-255 for thresholding black and white in the print_example
//0 is white 255 is black
#define blackwhite_threshold 10

int main(void)
{
	int i;
    char file[] = "train-labels.idx1-ubyte";
    uint8_t *labels = get_data(file);
    char file2[] = "train-labels.idx1-ubyte";
	uint8_t *images = get_data(file2);
	print_example(10000, images, labels);
	return 0;
}


