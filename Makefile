CXXFLAGS=-Wall -Wno-unused-function -std=c++11 -g

all: model dump

model: src/model.o
	g++ $^ -o $@

dump: src/dump.o
	g++ $^ -o $@

%.o:%.cpp src/fpga_format.h
	g++ $(CXXFLAGS) $< -c -o $@

clean:
	rm -f model dump src/*.o
