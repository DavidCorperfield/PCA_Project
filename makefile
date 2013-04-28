

SOURCE = ./source/*.cpp ./source/*.cu
INCLUDE = ./include/*.h
CUDA_INCLUDE = ~/NVIDIA_CUDA-5.0_Samples/common/inc


all:
	nvcc $(SOURCE) -o executable

test_parse:
	gcc ./include/parse_data.h 
	gcc ./source/parse_data.cpp ./debug/test_parse_data.cpp -o parse.o 	
	
test_feedforward:
	nvcc $(SOURCE) ./debug/test_feed_forward.cu -o feed_forward.o -I $(CUDA_INCLUDE)
	
test_backprop:
	nvcc $(SOURCE) ./debug/test_backprop.cu -o backprop.o -I $(CUDA_INCLUDE)
	
test_network:
	nvcc -g ./debug/test_network.cu $(SOURCE) -o network.o -Xcompiler -fopenmp

headers:
	gcc $(INCLUDE) -I $(CUDA_INCLUDE)
	
program: cudacode.o
	nvcc 

cudacode.o:
	nvcc -c ./include/feed_forward.h ./source/feedforward.cu 

clean:
	rm -r *.o
