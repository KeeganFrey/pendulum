# Compiler
CXX = g++

ARMA_FLAGS = -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_DONT_USE_WRAPPER

# Common Compiler flags
CXXFLAGS = -std=c++14 -I./armadillo/include -Wall -O2 $(ARMA_FLAGS)

# --- OS-Specific Configuration ---
# Default to Linux/macOS settings
OPENBLAS_LIB_DIR = ./OpenBLAS/lib
OPENBLAS_INCLUDE_DIR = ./OpenBLAS/include
# Statically link OpenBLAS by providing the full path to the static library
LDFLAGS_OS_BLAS = $(OPENBLAS_LIB_DIR)/libopenblas.a
LDFLAGS_OS_ARMA = -Wl,-rpath,./armadillo-14.6.0-macos -L./armadillo-14.6.0-macos -larmadillo

# Detect Windows (specifically MinGW)
ifeq ($(OS),Windows_NT)
    # Adjust for typical OpenBLAS naming on Windows with MinGW
    OPENBLAS_LIB_DIR = ./OpenBLAS/lib
    OPENBLAS_INCLUDE_DIR = ./OpenBLAS/include
    # Statically link OpenBLAS by providing the full path to the static library
    LDFLAGS_OS_BLAS = $(OPENBLAS_LIB_DIR)/libopenblas.a
	LDFLAGS_OS_ARMA = LDFLAGS_OS_ARMA = -Wl,-rpath,./armadillo-14.6.0-windows -L./armadillo-14.6.0-windows -larmadillo
endif

# Linker flags
LDFLAGS = $(LDFLAGS_OS_BLAS) $(LDFLAGS_OS_ARMA)

# Source files
SRCS = main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = main_program.x

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(TARGET)

# Rule to compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(OPENBLAS_INCLUDE_DIR) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean