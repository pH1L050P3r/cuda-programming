CC=nvcc
CFLAGS=-arch=sm_61

# gtx1050 architecture is sm_61 for more info : https://gt3rs.medium.com/compile-with-nvcc-3566fbdfdbf

gpu: main_cuda.cu
	$(CC) main_cuda.cu -o main_cuda $(CFLAGS)

cpu: main_cpu.cpp
	$(CC) main_cpu.cpp -o main_cpu $(CFLAGS)
