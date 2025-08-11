COMPILER=nvcc

# Libraries stored here: /usr/lib/x86_64-linux-gnu
# NPPC, NPP core library which MUST be included when linking any application, functions are listed in nppCore.h,
#   lnppc
# NPPIST, statistics and linear transform in nppi_statistics_functions.h and nppi_linear_transforms.h,
#   lnppist
# NPPISU, memory support functions in nppi_support_functions.h,
#   -lnppisu
COMPILER_FLAGS= -lcuda --std c++17 -lnpps -lnppisu -lnppist -lnppc -lcublas -Wno-deprecated-gpu-targets
TARGET = cudaAtScaleIndependentProject.exe
SOURCES = $(wildcard ./src/*.cpp)
SOURCES += $(wildcard ./src/fileManagement/*.cpp)
SOURCES_CU = $(wildcard ./src/signalProcessing/*.cu)
OBJECTS = $(SOURCES:.cpp=.o) $(SOURCES_CU:.cu=.o)

$(info    SOURCES:    $(SOURCES))
$(info    SOURCES_CU: $(SOURCES_CU))
$(info    OBJECTS:    $(OBJECTS))

INCLUDE = \
  -I./include/ \
  -I./include/fileManagement/ \
  -I./include/signalProcessing/

# Debug build flags
ifeq ($(dbg),1)
    $(info Adding debugging to build...)
    COMPILER_FLAGS += -g -G
endif

all: $(TARGET)

%.o: %.cpp
	@echo "Building C++ objects..."
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDE) -c $< -o $@

%.o: %.cu
	@echo "Building CUDA objects..."
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDE) -c $< -o $@

build: $(SOURCE)
	$(COMPILER) $(COMPILER_FLAGS) -I$(INCLUDE) -o $(TARGET) $(SOURCES)


$(TARGET): $(OBJECTS)
	@echo "Building target..."
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDE) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)

