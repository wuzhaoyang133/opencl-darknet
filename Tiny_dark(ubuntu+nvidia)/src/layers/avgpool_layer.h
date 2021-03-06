#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer;

image get_avgpool_image(avgpool_layer l);
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void resize_avgpool_layer(avgpool_layer *l, int w, int h);
void forward_avgpool_layer(const avgpool_layer l, network_state state);
void backward_avgpool_layer(const avgpool_layer l, network_state state);
#ifdef OCL
extern void forward_avgpool_layer_ocl(avgpool_layer l, network_state state);
#endif
#endif

