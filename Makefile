CXXFLAGS=-Wall -Wno-unused-function -std=c++11 -g

all: model dump

model: fpga_format.h

clean:
	rm -f model dump 
