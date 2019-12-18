#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_batchnorm_layer(int batch, int w, int h, int c);
void forward_batchnorm_layer(layer l, network_state state);
void backward_batchnorm_layer(layer l, network_state state);
#ifdef OCL
extern void forward_batchnorm_layer_ocl(layer l, network_state state);
extern void backward_batchnorm_layer_ocl(const layer l, network_state state);
extern void push_batchnorm_layer_ocl(layer l);
#endif

#endif
