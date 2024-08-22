CC=nvcc
CFLAGS=-arch=sm_61

# gtx1050 architecture is sm_61 for more info : https://gt3rs.medium.com/compile-with-nvcc-3566fbdfdbf

gpu: main.cu
	$(CC) main.cu -o main $(CFLAGS)
	# $(CC) main_cuda_um.cu -o main_cuda_um $(CFLAGS)

# cpu: main.cpp
# 	$(CC) main_cpu.cpp -o main_cpu $(CFLAGS)

clean:
	rm main