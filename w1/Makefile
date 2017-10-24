LIBS = `pkg-config --libs opencv`
GPP = g++ -ggdb

colourt: colourt.cpp
	$(GPP) -o colourt.exe colourt.cpp $(LIBS)

display: display.cpp
	$(GPP) -o display.exe display.cpp $(LIBS)

draw: draw.cpp
	$(GPP) -o draw.exe draw.cpp $(LIBS)

hello: hello.cpp
	$(GPP) -o hello.exe hello.cpp $(LIBS)

pixels: pixels.cpp
	$(GPP) -o pixels.exe pixels.cpp $(LIBS)

red: red.cpp
	$(GPP) -o red.exe red.cpp $(LIBS)

thresholding: thresholding.cpp
	$(GPP) -o thresholding.exe thresholding.cpp $(LIBS)
