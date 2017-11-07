LIBS = `pkg-config --libs opencv`
GPP = g++ -ggdb

convolution: convolution.cpp
	mkdir -p bin
	$(GPP) -o bin/convolution convolution.cpp $(LIBS)
