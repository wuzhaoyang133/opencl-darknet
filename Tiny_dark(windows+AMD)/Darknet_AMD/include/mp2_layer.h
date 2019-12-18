#ifndef MP2_LAYER_H
#define MP2_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer mp2_layer;

image get_mp2_image(mp2_layer l);
mp2_layer make_mp2_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_mp2_layer(mp2_layer *l, int w, int h);
void forward_mp2_layer(const mp2_layer l, network_state state);
void backward_mp2_layer(const mp2_layer l, network_state state);

#ifdef GPU
void forward_mp2_layer_gpu(mp2_layer l, network_state state);
void backward_mp2_layer_gpu(mp2_layer l, network_state state);
#endif

#ifdef OCL
extern void forward_mp2_layer_ocl(mp2_layer layer, network_state state);
extern void backward_mp2_layer_ocl(mp2_layer layer, network_state state);
#endif
#endif

