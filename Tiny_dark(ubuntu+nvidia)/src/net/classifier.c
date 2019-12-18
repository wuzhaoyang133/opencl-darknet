#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#include "string.h"

float *get_regression_values(char **labels, int n)
{
    float *v = calloc(n, sizeof(float));
    int i;
    for(i = 0; i < n; ++i){
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p+1);
    }
    return v;
}

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    network *nets = calloc(ngpus, sizeof(network));
    srand(time(0));
    for(i = 0; i < ngpus; ++i){

        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    network net = nets[0];
    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    list *options = read_data_cfg(datacfg);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    char **labels = get_labels(label_list);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    clock_t time;
    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.threads = 32;
    args.hierarchy = net.hierarchy;
    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;
    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;
    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);
    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0)
    {
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = 0;
#ifdef OCL
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
        }
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        free_data(train);
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}
void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;
    data val, buffer;
    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;
    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();
        pthread_join(load_thread, 0);
        val = buffer;
        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));
        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int w = net.w;
        int h = net.h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = calloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            if(net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));
    int size = net.w;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(&net, resized.w, resized.h);
        float *pred = network_predict(net, resized.data);
        if(net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);
        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}


void validate_classifier_multi(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    int scales[] = {224, 288, 320, 352, 384};
    int nscales = sizeof(scales)/sizeof(scales[0]);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_min(im, scales[j]);
            resize_network(&net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);
    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net.layers[layer_num];
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }
        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }
        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}

void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
	int i, j;
	network net = parse_network_cfg(filename);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	list *options = read_data_cfg(datacfg);
	char *label_list = option_find_str(options, "labels", "data/labels.list");
	char *leaf_list = option_find_str(options, "leaves", 0);
	if (leaf_list) change_leaves(net.hierarchy, leaf_list);
	char *valid_list = option_find_str(options, "valid", "data/train.list");
	int classes = option_find_int(options, "classes", 2);
	int topk = option_find_int(options, "top", 1);
	char **labels = get_labels(label_list);
	list *plist = get_paths(valid_list);
	char **paths = (char **)list_to_array(plist);
	int m = plist->size;
	free_list(plist);
	float avg_acc = 0;
	float avg_topk = 0;
	int *indexes = calloc(topk, sizeof(int));
	time_start();
	for (i = 0; i < m; ++i) {
		int class = -1;
		char *path = paths[i];
		for (j = 0; j < classes; ++j) {
			if (strstr(path, labels[j])) {
				class = j;
				break;
			}
		}
		image im = load_image_color(paths[i], 0, 0);
		image crop = resize_image(im, net.w, net.h);
		float *pred = network_predict(net, crop.data);
		if (net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);
		free_image(im);
		free_image(crop);
		top_k(pred, classes, topk, indexes);
		if (indexes[0] == class) avg_acc += 1;
		for (j = 0; j < topk; ++j) {
			if (indexes[j] == class) avg_topk += 1;
		}
		printf("%d: top 1: %f, top %d: %f\n", i, avg_acc / (i + 1), topk, avg_topk / (i + 1));
	}
	double elapsed = time_elapse();
	printf("valid elapse  time: %.4fms\n", elapsed);
}
void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(NULL));
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);
    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = resize_image(im, net.w, net.h);
        printf("image size: w = %d, h = %d\n", r.w, r.h);
        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
        top_k(predictions, net.outputs, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
            else printf("%s: %f\n",names[index], predictions[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}


void label_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);

        printf("%s\n", labels[ind]);
    }
}

void test_classifier(char *datacfg, char *cfgfile, char *weightfile0, char *weightfile1, char *weightfile2, int target_layer)
{
	int yanse[5][3] = { { 255, 0, 0 },{ 0,255,0 },{ 0, 0, 255 },{ 255,255,255 }, { 0, 255, 255 }};  
// { 1,208,0 },{ 123,255,0 },{ 149,156,96 },{ 0,135,0 },{ 1,78,0 },{ 160,80,43 },{ 0,213,213 },{ 255,255,255 },{ 216,191,216 },{ 248,0,0 },{ 195,195,195 },{ 117,149,142 },{ 176,0,0 },{ 78,0,0 },{ 146,73,47 },{ 251,251,0 },{ 230,202,19 },{ 222,0,206 },{ 0,0,227 },{ 179,198,223 } 
	int curr = 0;   
	int w, h, c, channels = 3;
	char buff0[256] = "/root/GUI_pointcloud/result/result_214_0";   
	char buff01[256] = "/root/GUI_pointcloud/result/result_214_0.png";
	image img0 = load_image_color("img.png", 0, 0);   			
	fflush(stdout); 
	network net = parse_network_cfg(cfgfile); 
	if (weightfile0) {
		load_weights(&net, weightfile0);   
	}
	srand(time(0));  
	list *options = read_data_cfg(datacfg);  
	char *test_list0 = option_find_str(options, "test", "/root/Darkent-FPGA-DET/x64/Release/data/cifar/test/JAX_214_0_sample.txt");   
	int classes = option_find_int(options, "classes", 5);   //classes = 20
	list *plist = get_paths(test_list0);
	char **paths = (char **)list_to_array(plist);   
	int m = plist->size;   
	free_list(plist);
	clock_t time;
	data val0, buffer0;
	load_args args = { 0 };
	args.w = net.w;   //32
	args.h = net.h;   //32
	args.paths = paths;
	args.classes = classes;   //20
	args.n = net.batch;   //32
	args.m = 0;
	args.labels = 0;
	args.d = &buffer0;   
	args.type = OLD_CLASSIFICATION_DATA;
	pthread_t load_thread = load_data_in_thread(args);  
	clock_t c_start, c_end, cc;
	c_start = clock();
	//FILE *fpWrite = fopen("result_gpu.txt", "w");
	for (curr = net.batch; curr < m; curr += net.batch) {
		//time=clock();
		pthread_join(load_thread, 0);   
		val0 = buffer0;
		if (curr < m) {
			args.paths = paths + curr;
			if (curr + net.batch > m) args.n = m - curr;
			load_thread = load_data_in_thread(args);   //load_thread = 210536192   sizeof(load_thread) = 8
		}
		matrix pred = network_predict_data(net, val0);

		int i, j;
		if (target_layer >= 0) {
		}
		for (i = 0; i < pred.rows; ++i) {
			//printf("%s\n", paths[curr-net.batch+i]);
			float zj = 0;
			int xuhao;
			int row;
			int col;
			for (j = 0; j < pred.cols; ++j) {
				//printf("\t%g", pred.vals[i][j]);
				if (pred.vals[i][j]>zj) {
					zj = pred.vals[i][j];
					xuhao = j;
				}
			}
                        //printf("class:%d\n",xuhao);
			char sub[100] = "";
			char str[100] = "";
			char * p, *q, *r, *t;
			p = strchr(paths[curr - net.batch + i], '_');
			q = strrchr(paths[curr - net.batch + i], '_');
			memcpy(sub, p + 1, q - p - 1);
			row = atoi(sub);
			r = strrchr(paths[curr - net.batch + i], '_');
			t = strchr(paths[curr - net.batch + i], '.');
			memcpy(str, r + 1, t - r - 1);
			col = atoi(str);
                        //printf("row:%d    col:%d\n",row,col);
			img0.data[0 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][0]/255.;
			img0.data[1 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][1]/255.;
			img0.data[2 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][2]/255.;
		}
		free_matrix(pred);
		fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock() - time), val0.X.rows, curr);
		//free_data(val);
	}
	//fclose(fpWrite);
	save_image(img0, buff0);
	image img1 = load_image_color(buff01, 0, 0); 
	FILE *fpWrite = fopen("/root/GUI_pointcloud/result/result_214_0.txt", "w");  
        if(fpWrite) 
	{
	    for(int i = 0; i < 1026; i++){
		for(int j = 0; j < 1026; j++)
            {  
		int r = (int)img1.data[0 * 1026 * 1026 + i * 1026 + j]*255;
		int g = (int)img1.data[1 * 1026 * 1026 + i * 1026 + j]*255;
		int b = (int)img1.data[2 * 1026 * 1026 + i * 1026 + j]*255;
        	fprintf(fpWrite,"%d,%d,%d\n",r,g,b); 
        	} 
		}
        fclose(fpWrite); 
        }
	//===============================
	img0 = load_image_color("img.png", 0, 0);
	char buff1[256] = "/root/GUI_pointcloud/result/result_214_1";   
	char buff11[256] = "/root/GUI_pointcloud/result/result_214_1.png";
	if (weightfile1) {
		load_weights(&net, weightfile1);   
	}
	//srand(time(0));  
	list *options1 = read_data_cfg(datacfg);  
	char *test_list1 = option_find_str(options1, "test", "/root/Darkent-FPGA-DET/x64/Release/data/cifar/test/JAX_214_1_sample.txt");   
	list *plist1 = get_paths(test_list1);
	char **paths1 = (char **)list_to_array(plist1);   
	m = plist1->size;   
	free_list(plist1);
	data val1,buffer1;	
	args.w = net.w;   //32
	args.h = net.h;   //32
	args.paths = paths1;
	args.classes = classes;   //20
	args.n = net.batch;   //32
	args.m = 0;
	args.labels = 0;
	args.d = &buffer1;   
	args.type = OLD_CLASSIFICATION_DATA;  
	//FILE *fpWrite = fopen("result_gpu.txt", "w");
	for (curr = net.batch; curr < m; curr += net.batch) {
		//time=clock();
		pthread_join(load_thread, 0);   
		val1 = buffer1;
		if (curr < m) {
			args.paths = paths1 + curr;
			if (curr + net.batch > m) args.n = m - curr;
			load_thread = load_data_in_thread(args);   //load_thread = 210536192   sizeof(load_thread) = 8
		}
		matrix pred1 = network_predict_data(net, val1);

		if (target_layer >= 0) {
		}
		for (int i = 0; i < pred1.rows; ++i) {
			float zj = 0;
			int xuhao;
			int row;
			int col;
			for (int j = 0; j < pred1.cols; ++j) {
				//printf("\t%g", pred.vals[i][j]);
				if (pred1.vals[i][j]>zj) {
					zj = pred1.vals[i][j];
					xuhao = j;
				}
			}
                        //printf("class:%d\n",xuhao);
			char sub1[100] = "";
			char str1[100] = "";
			char * p, *q, *r, *t;
			p = strchr(paths1[curr - net.batch + i], '_');
			q = strrchr(paths1[curr - net.batch + i], '_');
			memcpy(sub1, p + 1, q - p - 1);
			row = atoi(sub1);
			r = strrchr(paths1[curr - net.batch + i], '_');
			t = strchr(paths1[curr - net.batch + i], '.');
			memcpy(str1, r + 1, t - r - 1);
			col = atoi(str1);
                        //printf("row:%d    col:%d\n",row,col);
			img0.data[0 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][0]/255.;
			img0.data[1 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][1]/255.;
			img0.data[2 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][2]/255.;
		}
		free_matrix(pred1);
		fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock() - time), val1.X.rows, curr);
		//free_data(val);
	}
	//fclose(fpWrite);
	save_image(img0, buff1);
	image img2 = load_image_color(buff11, 0, 0); 
	FILE *fpWrite1 = fopen("/root/GUI_pointcloud/result/result_214_1.txt", "w");  
        if(fpWrite1) 
	{
	    for(int i = 0; i < 1026; i++){
		for(int j = 0; j < 1026; j++)
            {  
		int r = (int)img2.data[0 * 1026 * 1026 + i * 1026 + j]*255;
		int g = (int)img2.data[1 * 1026 * 1026 + i * 1026 + j]*255;
		int b = (int)img2.data[2 * 1026 * 1026 + i * 1026 + j]*255;
        	fprintf(fpWrite1,"%d,%d,%d\n",r,g,b); 
        	} 
		}
        fclose(fpWrite1);
	} 
	//===============================
	img0 = load_image_color("img.png", 0, 0);
	char buff2[256] = "/root/GUI_pointcloud/result/result_214_2";   
	char buff21[256] = "/root/GUI_pointcloud/result/result_214_2.png";
	if (weightfile2) {
		load_weights(&net, weightfile2);   
	}
	//srand(time(0));  
	list *options2 = read_data_cfg(datacfg);  
	char *test_list2 = option_find_str(options2, "test", "/root/Darkent-FPGA-DET/x64/Release/data/cifar/test/JAX_214_2_sample.txt");   
	list *plist2 = get_paths(test_list2);
	char **paths2 = (char **)list_to_array(plist2);   
	m = plist2->size;   
	free_list(plist2);
	data val2,buffer2;	
	args.w = net.w;   //32
	args.h = net.h;   //32
	args.paths = paths2;
	args.classes = classes;   //20
	args.n = net.batch;   //32
	args.m = 0;
	args.labels = 0;
	args.d = &buffer2;   
	args.type = OLD_CLASSIFICATION_DATA;  
	//FILE *fpWrite = fopen("result_gpu.txt", "w");
	for (curr = net.batch; curr < m; curr += net.batch) {
		//time=clock();
		pthread_join(load_thread, 0);   
		val2 = buffer2;
		if (curr < m) {
			args.paths = paths2 + curr;
			if (curr + net.batch > m) args.n = m - curr;
			load_thread = load_data_in_thread(args);   //load_thread = 210536192   sizeof(load_thread) = 8
		}
		matrix pred2 = network_predict_data(net, val2);

		if (target_layer >= 0) {
		}
		for (int i = 0; i < pred2.rows; ++i) {
			float zj = 0;
			int xuhao;
			int row;
			int col;
			for (int j = 0; j < pred2.cols; ++j) {
				//printf("\t%g", pred.vals[i][j]);
				if (pred2.vals[i][j]>zj) {
					zj = pred2.vals[i][j];
					xuhao = j;
				}
			}
                        //printf("class:%d\n",xuhao);
			char sub2[100] = "";
			char str2[100] = "";
			char * p, *q, *r, *t;
			p = strchr(paths2[curr - net.batch + i], '_');
			q = strrchr(paths2[curr - net.batch + i], '_');
			memcpy(sub2, p + 1, q - p - 1);
			row = atoi(sub2);
			r = strrchr(paths2[curr - net.batch + i], '_');
			t = strchr(paths2[curr - net.batch + i], '.');
			memcpy(str2, r + 1, t - r - 1);
			col = atoi(str2);
                        //printf("row:%d    col:%d\n",row,col);
			img0.data[0 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][0]/255.;
			img0.data[1 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][1]/255.;
			img0.data[2 * 1026 * 1026 + (row - 1) * 1026 + col - 1] = yanse[xuhao][2]/255.;
		}
		free_matrix(pred2);
		fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock() - time), val2.X.rows, curr);
		//free_data(val);
	}
	//fclose(fpWrite);

	save_image(img0, buff2);
	image img3 = load_image_color(buff21, 0, 0); 

	FILE *fpWrite2 = fopen("/root/GUI_pointcloud/result/result_214_2.txt", "w");  
        if(fpWrite2) 
	{
	    for(int i = 0; i < 1026; i++){
		for(int j = 0; j < 1026; j++)
            {  
		int r = (int)img3.data[0 * 1026 * 1026 + i * 1026 + j]*255;
		int g = (int)img3.data[1 * 1026 * 1026 + i * 1026 + j]*255;
		int b = (int)img3.data[2 * 1026 * 1026 + i * 1026 + j]*255;
        	fprintf(fpWrite2,"%d,%d,%d\n",r,g,b); 
        	} 
		}
        fclose(fpWrite2);
	} 

	//=====================

	//c_end = clock();
	//cc = c_end - c_start;
	//printf("CLOCKS_PER_SEC:%d\n", CLOCKS_PER_SEC);
	//printf("time:%f s\n", cc / CLOCKS_PER_SEC*1.0);
	char buff[256] = "/root/GUI_pointcloud/result/result_214_fusion"; 
	char bufff[256] = "/root/GUI_pointcloud/result/result_214_fusion.png";  	
 	fflush(stdout);
	for(int i = 0; i < 1025; i++){
		for(int j = 0; j < 1025; j++)
            {  
		if (((img3.data[0 * 1026 * 1026 + i * 1026 + j]) != 0) || ((img3.data[1 * 1026 * 1026 + i * 1026 + j]) !=0) || ((img3.data[2 * 1026 * 1026 + i * 1026 + j]) !=0))
		{
		img0.data[0 * 1026 * 1026 + (i+1) * 1026 + j+1] =  img3.data[0 * 1026 * 1026 + i * 1026 + j];
		img0.data[1 * 1026 * 1026 + (i+1)* 1026 + j+1] =  img3.data[1 * 1026 * 1026 + i * 1026 + j];
		img0.data[2 * 1026 * 1026 + (i+1) * 1026 + j+1] =  img3.data[2 * 1026 * 1026 + i * 1026 + j];
		}
		else if(((img2.data[0 * 1026 * 1026 + i * 1026 + j]) != 0) || ((img2.data[1 * 1026 * 1026 + i * 1026 + j]) !=0) || ((img2.data[2 * 1026 * 1026 + i * 1026 + j]) !=0))
		{
		img0.data[0 * 1026 * 1026 + (i+1) * 1026 + j+1] =  img2.data[0 * 1026 * 1026 + i * 1026 + j];
		img0.data[1 * 1026 * 1026 + (i+1)* 1026 + j+1] =  img2.data[1 * 1026 * 1026 + i * 1026 + j];
		img0.data[2 * 1026 * 1026 + (i+1) * 1026 + j+1] =  img2.data[2 * 1026 * 1026 + i * 1026 + j];
		}
 		else
		{
		img0.data[0 * 1026 * 1026 + (i+1) * 1026 + j+1] =  img1.data[0 * 1026 * 1026 + i * 1026 + j];
		img0.data[1 * 1026 * 1026 + (i+1)* 1026 + j+1] =  img1.data[1 * 1026 * 1026 + i * 1026 + j];
		img0.data[2 * 1026 * 1026 + (i+1) * 1026 + j+1] =  img1.data[2 * 1026 * 1026 + i * 1026 + j];
		}
             } 
	}
	save_image(img0, buff);
	image img4 = load_image_color(bufff, 0, 0); 
	FILE *fpWrite4 = fopen("/root/GUI_pointcloud/result/result_214_finnal.txt", "w");  
        if(fpWrite4) 
	{
	    for(int i = 0; i < 1026; i++){
		for(int j = 0; j < 1026; j++)
            {  //printf("img4: %f\n",img4.data[0 * 1026 * 1026 + i * 1026 + j]);
		int r = img4.data[0 * 1026 * 1026 + i * 1026 + j]*255;
		int g = img4.data[1 * 1026 * 1026 + i * 1026 + j]*255;
		int b = img4.data[2 * 1026 * 1026 + i * 1026 + j]*255;
        	fprintf(fpWrite4,"%d,%d,%d\n",r,g,b); 
        	} 
	    }
        fclose(fpWrite4); 
        } 	
}
	
void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{

}


void gun_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{

}

void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{

}
void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *ocl_list = find_char_arg(argc, argv, "-ocls", 0);
    int *ocls = 0;
    int ocl = 0;
    int nocls = 0;
    if(ocl_list){
        printf("%s\n", ocl_list);
        int len = strlen(ocl_list);
        nocls = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (ocl_list[i] == ',') ++nocls;
        }
        ocls = calloc(nocls, sizeof(int));
        for(i = 0; i < nocls; ++i){
            ocls[i] = atoi(ocl_list);
            ocl_list = strchr(ocl_list, ',')+1;
        }
    } else {
        ocl = ocl_index;
        ocls = &ocl;
        nocls = 1;
    }
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    int cut_off = find_int_arg(argc, argv, "-cut_off", -1);
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *weight1 = (argc > 6) ? argv[6] : 0;
    char *weight2 = (argc > 7) ? argv[7] : 0;
    char *filename = (argc > 8) ? argv[8]: 0;
    char *layer_s = (argc > 9) ? argv[9]: 0;
    int layer = layer_s ? atoi(layer_s) : -1;
    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, ocls, nocls, clear);
    else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, weight1, weight2, layer);
    //else if(0==strcmp(argv[2], "fusionimg")) fusionimg(data, cfg);
    else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights);
    else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);
}


