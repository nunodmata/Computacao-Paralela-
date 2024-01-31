CC = gcc
SRC = src/
CFLAGS = -Ofast -lm -lstdc++ -march=native 

.DEFAULT_GOAL = MD.exe

MD.exe: $(SRC)/MD.cpp
	$(CC) $(CFLAGS) $(SRC)MD.cpp -lm -o MD.exe

clean:
	rm ./MD.exe

run:
	srun --partition=cpar perf stat -r 1 -M cpi ./MD.exe < inputdata.txt
