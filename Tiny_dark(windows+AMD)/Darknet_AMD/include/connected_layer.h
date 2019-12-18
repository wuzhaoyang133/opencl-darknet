#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer connected_layer;

connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize);

#ifdef OCL
extern void forward_connected_layer_ocl(connected_layer l, network_state state);
extern void push_connected_layer_ocl(connected_layer layer);
extern void backward_connected_layer_ocl(connected_layer l, network_state state);
extern void update_connected_layer_ocl(connected_layer layer, int batch, float learning_rate, float momentum, float decay);
#endif
void forward_connected_layer(connected_layer layer, network_state state);
void backward_connected_layer(connected_layer layer, network_state state);
void update_connected_layer(connected_layer layer, int batch, float learning_rate, float momentum, float decay);
void denormalize_connected_layer(layer l);
void statistics_connected_layer(layer l);

#endif

