LIBS = `pkg-config --libs opencv`
GPP = g++ -Wall

coins: coins.cpp
	mkdir bin
	$(GPP) -o bin/coins coins.cpp $(LIBS)
