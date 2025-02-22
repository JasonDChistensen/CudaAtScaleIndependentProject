CXX = g++
CXXFLAGS = -Wall -g -std=c++17
GTEST_DIR = ../extern/googletest/googletest
GTEST_INCLUDE = $(GTEST_DIR)/include
GTEST_LIB = $(GTEST_DIR)/libgtest.a
TARGET = newProjectTests.exe
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TEST_SOURCES = $(wildcard *_test.cpp)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

all: $(TARGET)

$(GTEST_LIB):
	@echo "Building Google Test..."
	$(CXX) $(CXXFLAGS) -I$(GTEST_INCLUDE)  -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest-all.cc -o $(GTEST_DIR)/gtest-all.o
	ar -rv $(GTEST_LIB) $(GTEST_DIR)/gtest-all.o
	@echo "Google Test built successfully."

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(GTEST_INCLUDE) -c $< -o $@

$(TARGET): $(OBJECTS) $(TEST_OBJECTS) $(GTEST_LIB)
	$(CXX) $(CXXFLAGS) -I$(GTEST_INCLUDE) $^ -lpthread -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TARGET)