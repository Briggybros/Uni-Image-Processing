LIBS = `pkg-config --libs opencv`
GPP = g++ -ggdb

colourt: colourt.cpp
	mkdir -p bin
	$(GPP) -o bin/colourt colourt.cpp $(LIBS)

display: display.cpp
	mkdir -p bin
	$(GPP) -o bin/display display.cpp $(LIBS)

draw: draw.cpp
	mkdir -p bin
	$(GPP) -o bin/draw draw.cpp $(LIBS)

hello: hello.cpp
	mkdir -p bin
	$(GPP) -o bin/hello hello.cpp $(LIBS)

pixels: pixels.cpp
	mkdir -p bin
	$(GPP) -o bin/pixels pixels.cpp $(LIBS)

red: red.cpp
	mkdir -p bin
	$(GPP) -o bin/red red.cpp $(LIBS)

thresholding: thresholding.cpp
	mkdir -p bin
	$(GPP) -o bin/thresholding thresholding.cpp $(LIBS)
