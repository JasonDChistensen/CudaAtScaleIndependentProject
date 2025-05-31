COMPILER=nvcc
INCLUDE = \
  -I/usr/local/cuda/include \
  -I./include \
  -I./include/fileManagement/ \
  -I./include/signalProcessing

# Libraries stored here: /usr/lib/x86_64-linux-gnu
# NPPC, NPP core library which MUST be included when linking any application, functions are listed in nppCore.h,
#   lnppc
# NPPIST, statistics and linear transform in nppi_statistics_functions.h and nppi_linear_transforms.h,
#   lnppist
# NPPISU, memory support functions in nppi_support_functions.h,
#   -lnppisu

COMPILER_FLAGS= -lcuda --std c++17 -lnpps -lnppisu -lnppist -lnppc -Wno-deprecated-gpu-targets
TARGET = cudaAtScaleIndependentProject.exe
SOURCES = $(wildcard ./src/*.cpp)
SOURCES += $(wildcard ./src/fileManagement/*.cpp)
SOURCES += $(wildcard ./src/signalProcessing/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(SOURCES VAR is $(SOURCES ))

# Debug build flags
ifeq ($(dbg),1)
    $(info Adding debugging to build...)
    COMPILER_FLAGS += -g -G
endif

.PHONY: clean build run

build: $(SOURCE)
	$(COMPILER) $(COMPILER_FLAGS) -I$(INCLUDE) -o $(TARGET) $(SOURCES)


clean:
	rm -f $(TARGET) $(OBJECTS)

run:
	./$(TARGET)

all: clean build run