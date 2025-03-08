COMPILER = nvcc
COMPILER_FLAGS = -std=c++17
GTEST_DIR = ../extern/googletest/googletest
GTEST_INCLUDE = $(GTEST_DIR)/include
GTEST_LIB = $(GTEST_DIR)/libgtest.a
TARGET = newProjectTests.exe
SOURCES = $(wildcard *.cpp)
SOURCES = \
  ../src/fileManagement/readFile.cpp 

SOURCES_CU = \
  ../src/signalProcessing/upsample.cu
 $(info    SOURCES: $(SOURCES))

OBJECTS = $(SOURCES:.cpp=.o) $(SOURCES_CU:.cu=.o)
$(info    OBJECTS: $(OBJECTS))
TEST_SOURCES = $(wildcard *_test.cpp)
TEST_SOURCES = \
  main.cpp \
  fileManagement/tests.cpp \
  signalProcessing/tests.cpp \
#  ../src/fileManagement/readFile.cpp \
#  ../src/signalProcessing/upsample.cpp



INCLUDE = \
  -I../include/ \
  -I../include/fileManagement/ \
  -I../include/signalProcessing/
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

all: $(TARGET)

$(GTEST_LIB):
	@echo "Building Google Test..."
	$(COMPILER) $(COMPILER_FLAGS) -I$(GTEST_INCLUDE)  -I$(GTEST_DIR) -c $(GTEST_DIR)/src/gtest-all.cc -o $(GTEST_DIR)/gtest-all.o
	ar -rv $(GTEST_LIB) $(GTEST_DIR)/gtest-all.o
	@echo "Google Test built successfully."

%.o: %.cpp
	@echo "Building C++ objects..."
	$(COMPILER) $(COMPILER_FLAGS) -I$(GTEST_INCLUDE) $(INCLUDE) -c $< -o $@

%.o: %.cu
	@echo "Building CUDA objects..."
	$(COMPILER) $(COMPILER_FLAGS) -I$(GTEST_INCLUDE) $(INCLUDE) -c $< -o $@

$(TARGET): $(OBJECTS) $(TEST_OBJECTS) $(GTEST_LIB)
	@echo "Building target..."
	$(COMPILER) $(COMPILER_FLAGS) -I$(GTEST_INCLUDE) $(INCLUDE) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(TARGET)