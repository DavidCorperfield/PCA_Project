#include "../include/parse_data.h"
#include <byteswap.h>
#include <time.h>

//num from 0-255 for thresholding black and white in the print_example
//0 is white 255 is black
#define blackwhite_threshold 20
//data directory with respect to the binary
#define data_dir "/scratch/hpc/colinw/Project/PCA_Project/data/"
//#define data_dir "./data/"


uint8_t * get_data(char *filename){
	//make sure int is of size 4, otherwise this parser will not work
	assert(!((int)sizeof(int) - 4));
	
	FILE * fp;
	char * file_dir = (char*)malloc(snprintf(NULL, 0, "%s%s", data_dir, filename) + 1);
	sprintf(file_dir, "%s%s", data_dir, filename);
	printf("file to be opened: %s .\n",file_dir);
	fp = fopen(file_dir,"rb");
	if(!fp)
		printf("could not open file");

	//read first number out of the file
	uint32_t magic_number = 0;
	fread(&magic_number, sizeof(int), 1, fp);	
	magic_number = __bswap_32(magic_number); 		
	//must be a labels file
	if(magic_number == 2049){
		int num_items;
		fread(&num_items, sizeof(int), 1, fp);
		num_items = __bswap_32(num_items);
		printf("the number of items in this file is: %i\n", num_items);
		
		/*now read the actual data*/
		uint8_t *items = (uint8_t*)malloc(num_items*sizeof(uint8_t));
		fread(items, sizeof(uint8_t), num_items, fp);
		fclose(fp);
		return items;
	}
	//must be a images file
	else if(magic_number == 2051){
		int num_images;
		/*read the header data of the file*/
		fread(&num_images, (size_t)sizeof(int), (size_t)1, fp);
		//we already know the next two integers will be 28(dimesions of images)
		uint32_t temp;
        fread(&temp, (size_t)sizeof(int), (size_t)2, fp);
        temp = __bswap_32(num_images);
		num_images = __bswap_32(num_images);
		printf("the number of images in this file is: %i and dimensions are %i and %i\n", num_images,temp, temp);
		
		/*now read the actual data*/
		uint8_t *images = (uint8_t*)malloc(28*28*num_images*sizeof(uint8_t));
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

float *get_data_f(char *filename){
	//make sure int is of size 4, otherwise this parser will not work
	assert(!((int)sizeof(int) - 4));
	
	FILE * fp;
	char * file_dir = (char*)malloc(snprintf(NULL, 0, "%s%s", data_dir, filename) + 1);
	sprintf(file_dir, "%s%s", data_dir, filename);
	printf("file to be opened: %s .\n",file_dir);
	fp = fopen(file_dir,"rb");
	if(!fp)
		printf("could not open file");

	//read first number out of the file
	uint32_t magic_number = 0;
	fread(&magic_number, (size_t)sizeof(int), (size_t)1, fp);	
	magic_number = __bswap_32(magic_number); 		
	
	//must be a labels file
	if(magic_number == 2049){
		int num_items;
		fread(&num_items, sizeof(int), 1, fp);
		num_items = __bswap_32(num_items);
		printf("the number of items in this file is: %i\n", num_items);
		
		/*now read the actual data*/
		float *items = (float*)malloc(num_items*sizeof(float));
		fread(items, sizeof(uint8_t), num_items, fp);
		fclose(fp);
		return items;
	}
	//must be a images file
	else if(magic_number == 2051){
		int num_images;
		/*read the header data of the file*/
		fread(&num_images, (size_t)sizeof(int), (size_t)1, fp);
		//we already know the next two integers will be 28(dimesions of images)
		uint32_t temp;
        uint32_t temp2;

        fread(&temp, (size_t)sizeof(int), (size_t)2, fp);
        temp = __bswap_32(temp);
		num_images = __bswap_32(num_images);
		printf("the number of images in this file is: %i and dimensions are %i and %i\n", num_images,temp, temp);
		/*now read the actual data*/
        uint8_t *images = (uint8_t*)malloc(28*28*num_images*sizeof(uint8_t));
		fread(images, sizeof(uint8_t), num_images*28*28, fp);	
		float *images_f = (float*)malloc(28*28*num_images*sizeof(float));
        for (int i = 0; i < 28*28*num_images; i++){
            images_f[i] = images[i];
        }
		fclose(fp);
		return images_f;
	}
	else{
		printf("read an invalid magic number\n");
		fclose(fp);
		return 0;
	}
}
 
 
void get_norm_image(float *norm_img, uint8_t *images, int img_num){
    int i,j;
    int start_index = img_num*MAX_NUM_NEURONS;
        for (i = 0; i < 28; i++){
            for (j = 0; j < 28; j++){
                norm_img[i*28 + j] = (((float)images[start_index + i*28 + j])*1.6)/255.0 - 0.8;//scale the input data to -0.8 to 0.8
            }
        }
}


void get_input_image(float *images, float *input, int img_num){
    int i,j;
    int start_index = img_num*MAX_NUM_NEURONS;
        #pragma omp for
        for (i = 0; i < 28; i++){
            for (j = 0; j < 28; j++){
                input[i*28 + j] = (float)images[start_index + i*28 + j];
            }
        }
}
            
void print_example(int img_num, uint8_t *images, uint8_t *labels){
	int i,j;
	printf("\nthe character is: %u", labels[img_num]);
	int start_index = img_num*28*28;
	printf("\nprinting image !\n");
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

