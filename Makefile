.phony: seq_build seq_test seq par_build par_test par clean

par_build:
	./build compress.cu parallel.out

par_test:
	sbatch run_tests parallel.out

par: par_build par_test

seq_build:
	g++ -std=gnu++0x -Wall -Werror -I sequential sequential/main.cpp sequential/toojpeg.cpp -o sequential.out

seq_test:
	./sequential.out test/0/input.ppm test/0/output.jpg

seq: seq_build seq_test

clean:
	rm *.out