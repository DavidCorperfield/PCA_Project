#include "../include/parse_data.h"
#include <byteswap.h>

//num from 0-255 for thresholding black and white in the print_example
//0 is white 255 is black
#define blackwhite_threshold 10

uint8_t * get_data(char *filename){
	//make sure int is of size 4, otherwise this parser will not work
	assert(!((int)sizeof(int) - 4));
	
	FILE * fp;
	char * file_dir = malloc(snprintf(NULL, 0, "%s%s", data_dir, filename) + 1);
	sprintf(file_dir, "%s%s", data_dir, filename);
	printf("file to be opened: %s\n",file_dir);
	fp = fopen(file_dir,"rb");
	if(!fp)
		printf("bad filename\n");

	//read first number out of the file
	uint32_t magic_number = 0;
	fread(&magic_number, sizeof(int), 1, fp);	
	magic_number = __bswap_32(magic_number); 		
	
	//must be a labels file
	if(magic_number == 2049){
		fread(&num_items, sizeof(int), 1, fp);
		num_items = __bswap_32(num_items);
		printf("the number of items in this file is: %i\n", num_items);
		
		/*now read the actual data*/
		uint8_t *items = malloc(num_items*sizeof(uint8_t));
		fread(items, sizeof(uint8_t), num_items, fp);
		fclose(fp);
		return items;
	}
	//must be a images file
	else if(magic_number == 2051){
		/*read the header data of the file*/
		fread(&num_images, sizeof(int), 1, fp);
		//we already know the next two integers will be 28(dimesions of images)
		uint32_t temp;
		fread(&temp, sizeof(int), 2, fp);
		num_images = __bswap_32(num_images);
		printf("the number of images in this file is: %i\n", num_images);
		
		/*now read the actual data*/
		uint8_t *images = malloc(28*28*num_images*sizeof(uint8_t));
		fread(images, sizeof(uint8_t), num_images*28*28, fp);	
		fclose(fp);
		return images;
	}
	else{
		printf("read an invalid magic number\n");
		fclose(fp);
		return 0;
	}
}

void print_example(int img_num, uint8_t * images, uint8_t * labels){
	int i,j;
	printf("\nthe character is: %u", labels[img_num]);
	int start_index = img_num*28*28;
	printf("\nprinting image\n");
	for(i = 0; i < 28; i++){
		for(j = 0; j < 28; j++){
			if(images[28*i + j + start_index] < blackwhite_threshold)
				printf(" ");
			else
				printf("X");
		}
		printf("\n");
	}
}
