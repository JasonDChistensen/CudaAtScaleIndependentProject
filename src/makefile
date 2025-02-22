CXX = g++
INCLUDE = ../include
CXXFLAGS = -Wall -std=c++11
TARGET = newProjectTest.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) $(OBJECTS)