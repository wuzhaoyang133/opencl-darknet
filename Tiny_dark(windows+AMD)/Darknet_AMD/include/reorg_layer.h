#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

layer make_reorg_layer(int batch, int h, int w, int c, int stride, int reverse);
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(const layer l, network_state state);
void backward_reorg_layer(const layer l, network_state state);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network_state state);
void backward_reorg_layer_gpu(layer l, network_state state);
#endif

#ifdef OCL
extern void forward_reorg_layer_ocl(layer l, network_state state);
extern void backward_reorg_layer_ocl(layer l, network_state state);
extern cl_int reorg_on_ocl(cl_command_queue command, cl_mem x, int w, int h, int c, int batch, int stride, int forward, cl_mem out);
#endif

#endif

