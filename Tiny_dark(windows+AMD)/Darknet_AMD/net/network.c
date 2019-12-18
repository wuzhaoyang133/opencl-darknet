#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "connected_layer.h"

#include "convolutional_layer.h"



#include "batchnorm_layer.h"
#include "maxpool_layer.h"

#include "avgpool_layer.h"

#include "softmax_layer.h"


int get_current_batch(network net)
{
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

void reset_momentum(network net)
{
    if (net.momentum == 0) return;
    net.learning_rate = 0;
    net.momentum = 0;
    net.decay = 0;
    #ifdef GPU
        //if(net.gpu_index >= 0) update_network_gpu(net);
    #endif
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                //if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case MP2:
            return "mp2";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.cut_off = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(int));
	net.cost = calloc(1, sizeof(float));
    #ifdef GPU
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));
    #endif

#ifdef OCL
    net.input_ocl = calloc(1, sizeof(cl_mem));
    net.truth_ocl = calloc(1, sizeof(cl_mem));
#endif
    return net;
}

void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta){
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, state);
        state.input = l.output;
    }
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    //int start = cut_off == net.n? 0 : cut_off;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

float *get_network_output(network net)
{
#ifdef GPU
    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif 

#ifdef OCL
    if (gpu_index >= 0) return get_network_output_ocl(net);
#endif
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float get_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
	return sum/count;
}

void calc_network_cost(network *netp)
{
	network net = *netp;
	int i;
	float sum = 0;
	int count = 0;
	for (i = 0; i < net.n; ++i) {
		if (net.layers[i].cost) {
			sum += net.layers[i].cost[0];
			++count;
		}
	}
	*net.cost = sum / count;
	//printf("net.cost:%f\n", sum / count);
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

void backward_network(network net, network_state state)
{
    int i;
    float *original_input = state.input;
    float *original_delta = state.delta;
    state.workspace = net.workspace;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output;
            state.delta = prev.delta;
        }
        layer l = net.layers[i];
        l.backward(l, state);
    }
}

float train_network_datum(network net, float *x, float *y)
{
#ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net, x, y);
#endif

#ifdef OCL
    if(gpu_index >= 0) return train_network_datum_ocl(net, x, y);
#endif
    network_state state;
    *net.seen += net.batch;
    state.index = 0;
    state.net = net;
    state.input = x;
    state.delta = 0;
    state.truth = y;
    state.train = 1;
    forward_network(net, state);
    backward_network(net, state);
    float error = get_network_cost(net);
    if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network(network net, data d)
{
    assert(d.X.rows % net.batch == 0);
    int batch = net.batch;
    int n = d.X.rows / batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i)
    {
        get_next_batch(d, batch, i*batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}


float train_network_batch(network net, data d, int n)
{
    int i,j;
    network_state state;
    state.index = 0;
    state.net = net;
    state.train = 1;
    state.delta = 0;
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = rand()%d.X.rows;
            state.input = d.X.vals[index];
            state.truth = d.y.vals[index];
            forward_network(net, state);
            backward_network(net, state);
            sum += get_network_cost(net);
        }
        update_network(net);
    }
    return (float)sum/(n*batch);
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    if(gpu_index >= 0){
        cuda_free(net->workspace);
    }
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
#ifdef GPU
    if(gpu_index >= 0){
        if(net->input_gpu) {
            cuda_free(*net->input_gpu);
            *net->input_gpu = 0;
            cuda_free(*net->truth_gpu);
            *net->truth_gpu = 0;
        }
        net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}


image get_network_image_layer(network net, int i)
{
    layer l = net.layers[i];
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network net, int k, int *index)
{
    int size = get_network_output_size(net);
    float *out = get_network_output(net);
    top_k(out, size, k, index);
}


float *network_predict(network net, float *input)    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif
#ifdef OCL
	if (gpu_index >= 0) return network_predict_ocl(net, input);
	
#endif
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);;
    return out;
}
void avg_flipped_yolo(layer l)
{
	int i, j, n, z;
	float *flip = l.output + l.outputs;
	for (j = 0; j < l.h; ++j) {
		for (i = 0; i < l.w / 2; ++i) {
			for (n = 0; n < l.n; ++n) {
				for (z = 0; z < l.classes + 4 + 1; ++z) {
					int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
					int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
					float swap = flip[i1];
					flip[i1] = flip[i2];
					flip[i2] = swap;
					if (z == 0) {
						flip[i1] = -flip[i1];
						flip[i2] = -flip[i2];
					}
				}
			}
		}
	}
	for (i = 0; i < l.outputs; ++i) {
		l.output[i] = (l.output[i] + flip[i]) / 2.;
	}
}


void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((float)netw / w) < ((float)neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	}
	else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		box b = dets[i].bbox;
		b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
		b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
		b.w *= (float)netw / new_w;
		b.h *= (float)neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((float)netw / w) < ((float)neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	}
	else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		//if (i< 40)
		//{
		//	int index = i;
		//	printf("index: %d, %f, %f, %f, %f, %f\n", index, dets[index].objectness, dets[index].bbox.x, dets[index].bbox.y, dets[index].bbox.w, dets[index].bbox.h);
		//}
		box b = dets[i].bbox;
		b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
		b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
		b.w *= (float)netw / new_w;
		b.h *= (float)neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
	int i, j, n;
	float *predictions = l.output;
	//int per_cell = 5*num+classes;
	for (i = 0; i < l.side*l.side; ++i) {
		int row = i / l.side;
		int col = i % l.side;
		for (n = 0; n < l.n; ++n) {
			int index = i*l.n + n;
			int p_index = l.side*l.side*l.classes + i*l.n + n;
			float scale = predictions[p_index];
			int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n) * 4;
			box b;
			b.x = (predictions[box_index + 0] + col) / l.side * w;
			b.y = (predictions[box_index + 1] + row) / l.side * h;
			b.w = pow(predictions[box_index + 2], (l.sqrt ? 2 : 1)) * w;
			b.h = pow(predictions[box_index + 3], (l.sqrt ? 2 : 1)) * h;
			dets[index].bbox = b;
			dets[index].objectness = scale;
			for (j = 0; j < l.classes; ++j) {
				int class_index = i*l.classes;
				float prob = scale*predictions[class_index + j];
				dets[index].prob[j] = (prob > thresh) ? prob : 0;
			}
		}
	}
}



static int entry_index(layer l, int batch, int location, int entry)
{
	int n = location / (l.w*l.h);
	int loc = location % (l.w*l.h);
	return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1) + entry*l.w*l.h + loc;
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
	int i, j, n;
	float *predictions = l.output;

	if (l.batch == 2) avg_flipped_yolo(l);
	int count = 0;
	for (i = 0; i < l.w*l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		for (n = 0; n < l.n; ++n) {
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			float objectness = predictions[obj_index];
			if (objectness <= thresh) continue;
			int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
			dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
			dets[count].objectness = objectness;
			dets[count].classes = l.classes;
			for (j = 0; j < l.classes; ++j) {
				int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
				float prob = objectness*predictions[class_index];
				dets[count].prob[j] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}
	correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
	return count;
}


box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	box b;
	b.x = (i + x[index + 0 * stride]) / lw;
	b.y = (j + x[index + 1 * stride]) / lh;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}


int num_detections(network *net, float thresh)
{
	int i;
	int s = 0;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO) {
			s += yolo_num_detections(l, thresh);
		}
		if (l.type == DETECTION || l.type == REGION) {
			s += l.w*l.h*l.n;
		}
	}
	return s;
}

int yolo_num_detections(layer l, float thresh)
{
	int i, n;
	int count = 0;
	for (i = 0; i < l.w*l.h; ++i) {
		for (n = 0; n < l.n; ++n) {
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			if (l.output[obj_index] > thresh) {
				++count;
			}
		}
	}
	return count;
}



void free_detections(detection *dets, int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		free(dets[i].prob);
		if (dets[i].mask) free(dets[i].mask);
	}
	free(dets);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net.batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }

		/*for (int ii = 0; ii < 40; ii++)
			printf("%f, ", X[ii]);
		printf("\n");*/

        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network net)
{
    int i,j;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        free_layer(net.layers[i]);
    }
    free(net.layers);
	free(net.scales);
	free(net.seen);
	free(net.steps);
#if defined(GPU)
    if(*net.input_gpu) cuda_free(*net.input_gpu);
    if(*net.truth_gpu) cuda_free(*net.truth_gpu);
    if(net.input_gpu) free(net.input_gpu);
    if(net.truth_gpu) free(net.truth_gpu);

#elif defined(OCL) 
	if(net.input_ocl) free(net.input_ocl);
	if(net.truth_ocl) free(net.truth_ocl);
	if(net.workspace_ocl) clReleaseMemObject(net.workspace_ocl);
#else 
	if (net.workspace) free(net.workspace);
#endif
}
#ifdef OCL

void forward_network_ocl(network net, network_state state)
{
	cl_int errNum;
	static double time = 0.0;
    state.workspace_ocl = net.workspace_ocl;
    int i;
    for(i = 0; i < net.n; ++i)   //net.n = 107
    {
        state.index = i;
        layer l = net.layers[i];
		printf("**************************layer: %d*******************************\n", i);
        l.forward_ocl(l, state);
        state.input_ocl = l.output_ocl;
    }
	calc_network_cost(&net);
}

void backward_network_ocl(network net, network_state state)
{
    state.workspace_ocl = net.workspace_ocl;
    int i;
    cl_mem original_input = state.input_ocl;
    cl_mem original_delta = state.delta_ocl;
    for(i = net.n-1; i >= 0; --i)
    {
        state.index = i;
        layer l = net.layers[i];
        if(i == 0)
        {
            state.input_ocl = original_input;
            state.delta_ocl = original_delta;
        }
        else
        {
            layer prev = net.layers[i-1];
            state.input_ocl = prev.output_ocl;
            state.delta_ocl = prev.delta_ocl;
        }
        l.backward_ocl(l, state);
    }
}

void forward_backward_network_ocl(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths * net.batch;
    if(!*net.input_ocl)
    {
        *net.input_ocl = cl_make_array(context, x, x_size * sizeof(float));
        *net.truth_ocl = cl_make_array(context, y, y_size * sizeof(float));
    }
    else
    {
        cl_push_array(commandQueue, x, *net.input_ocl, x_size * sizeof(float));
        cl_push_array(commandQueue, y, *net.truth_ocl,  y_size * sizeof(float));
    }
    state.input_ocl = *net.input_ocl;
    state.delta_ocl = NULL;
    state.truth_ocl = *net.truth_ocl;
    state.train = 1;
    forward_network_ocl(net, state);
    backward_network_ocl(net, state);
}

void update_network_ocl(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i)
    {
        layer l = net.layers[i];
        l.t = get_current_batch(net);
        if(l.update_ocl)
            l.update_ocl(l, update_batch, rate, net.momentum, net.decay);
    }
}
float train_network_datum_ocl(network net, float *x, float *y)
{
    *net.seen += net.batch;
    forward_backward_network_ocl(net, x, y);
    float error = get_network_cost(net);
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_ocl(net);
    return error;
}
float*  get_network_output_ocl(network net)
{
    cl_int errNum; int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    layer l = net.layers[i];
    if(l.type != REGION)
    {
    	errNum = clEnqueueReadBuffer(commandQueue, l.output_ocl, CL_TRUE, 0, l.outputs * l.batch * sizeof(float), l.output, 0, NULL, NULL);
    	if(!check_errNum(errNum))
    	{
    		fprintf(stderr, "GPU -> CPU, 读取网络输出失败, 退出程序\n");
    		exit(1);
    	}
    }
    return l.output;
}
float *network_predict_ocl(network net, float *input)
{
    int size = get_network_input_size(net) * net.batch;
	int i;
    network_state state;
	state.truth = NULL;
    state.index = 0;
    state.net = net;
    state.input_ocl = cl_make_array(context,input, size * sizeof(float));
    state.truth_ocl = NULL;
    state.train = 0;
    state.delta_ocl = NULL;
    forward_network_ocl(net, state);
	for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
	float *out = net.layers[i].output;
    clReleaseMemObject(state.input_ocl);
    return out;
}

#endif
