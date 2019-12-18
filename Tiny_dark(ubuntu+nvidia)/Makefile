DEBUG=0
OCL=1

VPATH=./src:./src/net/:./src/utils/:./src/layers/common/:./src/layers/
EXEC=darknet
OBJDIR=./obj/

CC=gcc
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -I./src/net/ -I./src/utils/ -I./src/layers/common/ -I./src/layers/

ifeq ($(OCL), 1)
COMMON+= -DOCL
endif

CFLAGS=-Wfatal-errors -std=c99
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
CUDA_LINK=-lOpenCL

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ=gemm.o utils.o convolutional_layer.o list.o image.o tree.o activations.o im2col.o col2im.o blas.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o parser.o option_list.o darknet.o box.o avgpool_layer.o layer.o classifier.o batchnorm_layer.o oclutils.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(CUDA_LINK)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@
	
obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

