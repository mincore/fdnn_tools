CXXFLAGS=-Wall -Wno-unused-function -std=c++11 -g

all: model dump

model: src/model.o
	g++ $^ -o $@

dump: src/dump.o
	g++ $^ -o $@

%.o:%.c fpga_format.h
	g++ $(CXXFLAGS) $< -o $@

clean:
	rm -f model dump 
