__kernel void fill_kernel(int N, float ALPHA, __global float *restrict X, int INCX)
{
	size_t i = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < N) X[i*INCX] = ALPHA;
}
__kernel void copy_kernel(int N, __global float *restrict X, int OFFX, int INCX, __global float *restrict Y, int OFFY, int INCY)
{
	int i = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

__kernel void upsample_kernel(int N, __global float *restrict x, int w, int h, int c, int batch, int stride, int forward, float scale,  __global float *restrict out)
{
	size_t i = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i >= N) return;
	int out_index = i;
	int out_w = i % (w*stride);
	i = i / (w*stride);
	int out_h = i % (h*stride);
	i = i / (h*stride);
	int out_c = i%c;
	i = i / c;
	int b = i%batch;

	int in_w = out_w / stride;
	int in_h = out_h / stride;
	int in_c = out_c;
	int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;
	if (forward) out[out_index] += scale * x[in_index];
	//else atomicAdd(x + in_index, scale * out[out_index]);
	else x[in_index] += scale*out[out_index];
}

__kernel void scale_bias_kernel(__global float *restrict output, __global float *restrict biases, int batch, int c, int size)
{
    size_t offset = get_global_id(0);
    size_t filter_id = get_global_id(1);
    size_t batch_id = get_global_id(2);

    if(offset < size && batch_id < batch && filter_id < c)
        output[batch_id * size * c + filter_id * size  + offset] *= biases[filter_id];
}

__kernel void add_bias_kernel(__global float *restrict output, __global float *restrict biases, int batch, int c, int size)
{
    size_t offset = get_global_id(0);
    size_t filter_id = get_global_id(1);
    size_t batch_id = get_global_id(2);

    if(offset < size && batch_id < batch && filter_id < c)
        output[batch_id * size * c + filter_id * size  + offset] += biases[filter_id];
} 

__kernel void normalize_kernel(long N, __global float *restrict x, __global float *restrict mean, __global float *restrict variance, int batch, int filters, int spatial)
{
    int index = get_global_id(0) + get_global_id(1) * get_global_size(0);
    if (index < N)
    { 
        int f = (index/spatial)%filters;   
        x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
    }  
}

__kernel void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, __global float*restrict add, int w2, int h2, int c2 , __global float *restrict out)
{
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (id >= size) return;
	int i = id % minw;
	id /= minw;
	int j = id % minh;
	id /= minh;
	int k = id % minc;
	id /= minc;
	int b = id % batch;

	int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
	int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
	out[out_index] += add[add_index];
	//printf("*****************************************************");
	//if(out_index < 20)
	//	printf("s1:%f, s2:%f, add[%d]: %f, out[%d]: %f\n", s1, s2, add_index, add[add_index], out_index, out[out_index]);
	//out[out_index] += add[add_index];


}
