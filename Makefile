.phony: build

build:
	g++ -std=gnu++0x -Wall -Werror -I . main.cpp toojpeg.cpp -o program.out

run:
	./program.out test/0/input.ppm test/0/output.jpg

all: build run

clean:
	rm *.out