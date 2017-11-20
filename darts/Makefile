CC = g++
CFLAGS = -g -O3 -std=c++11
SRCS = face.cpp
PROG = face

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG): $(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
