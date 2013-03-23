

SOURCE = ./source/*.cpp ./source/*.cu
INCLUDE = ./include/parse_data.h ./include/feed_forward.h
CUDA_INCLUDE = ~/NVIDIA_CUDA-5.0_Samples/common/inc



all:
	nvcc $(SOURCE) -o executable

test_parse:
	gcc ./include/parse_data.h 
	gcc ./source/parse_data.cpp ./debug/test_parse_data.cpp -o parse 	
	
test_feedforward:
	nvcc $(SOURCE) ./debug/test_feed_forward.cu -o feed_forward -I $(CUDA_INCLUDE)

headers:
	gcc $(INCLUDE)
	
program: cudacode.o
	nvcc 

cudacode.o:
	nvcc -c ./include/feed_forward.h ./source/feedforward.cu 

clean:
	rm -r *.o
