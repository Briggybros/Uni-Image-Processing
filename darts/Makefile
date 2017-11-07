CC = g++
CFLAGS = -g
SRCS = face.cpp
PROG = face

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG): $(SRCS)
	mkdir -p bin
	$(CC) $(CFLAGS) -o bin/$(PROG) $(SRCS) $(LIBS)
