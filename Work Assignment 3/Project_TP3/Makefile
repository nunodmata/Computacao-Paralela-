CC = nvcc
SRC = src/
CFLAGS = -O3 -Xcompiler "-msse4 -mtune=native -march=native -funroll-loops -fomit-frame-pointer" -lm -lstdc++ -g -std=c++14 -arch=sm_35 -Wno-deprecated-gpu-targets
RFLAGS = srun  --partition cpar --constraint=k20

.DEFAULT_GOAL = cuda.exe

cuda.exe: $(SRC)/kernel.cu
	module load gcc/7.2.0;\
	module load cuda/11.3.1;\
	$(CC) $(CFLAGS) $(SRC)kernel.cu -lm -o cuda.exe

clean:
	rm ./*.exe

run:
	$(RFLAGS) ./cuda.exe <inputdata.txt

