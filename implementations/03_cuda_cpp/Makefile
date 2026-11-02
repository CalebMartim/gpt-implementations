NVCC ?= $(shell if [ -x /opt/cuda/bin/nvcc ]; then echo /opt/cuda/bin/nvcc; else echo nvcc; fi)
NVCCFLAGS ?= -O3
LDFLAGS ?= -lcublas

TARGET = gpt_cuda
SRC = gpt_cuda.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
