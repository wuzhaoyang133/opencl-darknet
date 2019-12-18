// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"
#include "oclutils.h"
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;
//typedef struct detection {
//	box bbox;
//	int classes;
//	float *prob;
//	float *mask;
//	float objectness;
//	int sort_class;
//} detection;
typedef struct network{
    float *workspace;
    int n;
    int cut_off;
    int batch;
    int *seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    int outputs;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
	float *input;
	float *truth;
	float *delta;
    int adam;
    float B1;
    float B2;
    float eps;
    int inputs;
    int h, w, c;
    int max_crop;
    int min_crop;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
	int train;
	int index;

    int gpu_index;
    tree *hierarchy;
	float *cost;

    #ifdef GPU
    float **input_gpu;
    float **truth_gpu;
    #endif

#ifdef OCL
    cl_mem* input_ocl;
    cl_mem* truth_ocl;
    cl_mem workspace_ocl;
#endif
} network;

typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    network net;
#ifdef OCL
    cl_mem input_ocl;
    cl_mem truth_ocl;
    cl_mem delta_ocl;
    cl_mem workspace_ocl;
#endif

} network_state;

#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);
void forward_network_gpu(network net, network_state state);
void backward_network_gpu(network net, network_state state);
void update_network_gpu(network net);
#endif

#ifdef OCL
extern float*  get_network_output_ocl(network net);
extern float* network_predict_ocl(network net, float *input);
extern void forward_network_ocl(network net, network_state state);
extern float train_network_datum_ocl(network net, float *x, float *y);
#endif
float get_current_rate(network net);
int get_current_batch(network net);
void free_network(network net);
void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net, network_state state);
void backward_network(network net, network_state state);
void update_network(network net);

float train_network(network net, data d);
float train_network_batch(network net, data d, int n);
float train_network_sgd(network net, data d, int n);
float train_network_datum(network net, float *x, float *y);

matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);
void calc_network_cost(network *netp);

int get_network_nuisance(network net);
int get_network_background(network net);

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);
detection *make_network_boxes(network *net, float thresh, int *num);
void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets);
int num_detections(network *net, float thresh);
int yolo_num_detections(layer l, float thresh);
static int entry_index(layer l, int batch, int location, int entry);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative);
void avg_flipped_yolo(layer l);

#endif

