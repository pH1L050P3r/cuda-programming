CC=nvcc
CFLAGS=-arch=sm_61

# gtx1050 architecture is sm_61 for more info : https://gt3rs.medium.com/compile-with-nvcc-3566fbdfdbf

gpu: main.cu
	$(CC) main.cu ./CPU/vector_sum.cpp  -o main $(CFLAGS)
	# $(CC) main_cuda_um.cu -o main_cuda_um $(CFLAGS)

cpu: main.cpp
	$(CC) main.cpp ./CPU/vector_sum.cpp -o main $(CFLAGS)

clean:
	rm main