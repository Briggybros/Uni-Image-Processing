LIBS = `pkg-config --libs opencv`
GPP = g++ -ggdb

colourthr: colourthr.cpp
	mkdir -p bin
	$(GPP) -o bin/colourthr colourthr.cpp $(LIBS)

convolution: convolution.cpp
	mkdir -p bin
	$(GPP) -o bin/convolution convolution.cpp $(LIBS)

revert0: revert0.cpp
	mkdir bin
	$(GPP) -o bin/revert0 revert0.cpp $(LIBS)

revert1: revert1.cpp
	mkdir -p bin
	$(GPP) -o bin/revert1 revert1.cpp $(LIBS)

revert2: revert2.cpp
	mkdir -p bin
	$(GPP) -o bin/revert2 revert2.cpp $(LIBS)

revert3: revert3.cpp
	mkdir -p bin
	$(GPP) -o bin/revert3 revert3.cpp $(LIBS)

RGBtoHSV: RGBtoHSV.cpp
	mkdir -p bin
	$(GPP) -o bin/RGBtoHSV RGBtoHSV.cpp $(LIBS)
