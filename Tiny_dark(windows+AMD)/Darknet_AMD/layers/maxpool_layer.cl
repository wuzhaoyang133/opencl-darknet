
#define INF ~(1<<31)

__kernel void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, __global float *restrict input, __global float *restrict output, __global int *restrict indexes)
{
	/*int h = (in_h + 2 * pad) / stride;
	int w = (in_w + 2 * pad ) / stride;
	int c = in_c;
	int out_idx = get_global_id(1) * get_global_size(0)  + get_global_id(0);
	if(out_idx < n)
	{	 
		int i = out_idx % w;
    	int j = out_idx / w % h;
    	int k = out_idx /w /h % c;
    	int b = out_idx /w /h /c;
    	
    	int w_offset = -pad;
    	int h_offset = -pad; 
    	
    	float max = -INF;
    	int max_idx = -1;
    	
    	for(int y = 0; y < size; ++y)
    	{
    	    for(int x = 0; x < size; ++x)
    	 	{
    	 		int in_i = w_offset + i * stride + x;
    	 		int in_j = h_offset + j * stride + y;
    	 		int in_idx = in_i + in_j * in_w + k * in_w * in_h + b * in_w * in_h * in_c;
    	 		float val = (in_i <  in_w && in_i >= 0 && in_j < in_h && in_j >=0) ? input[in_idx] : -INF;
    	 		max_idx = (val > max) ? in_idx : max_idx;
    	 		max = (val > max) ? val : max;
    	 	}
    	 }
    	output[out_idx] = max;
    	indexes[out_idx] = max_idx;
	}*/

	int h = (in_h + 2 * pad) / stride;
	int w = (in_w + 2 * pad) / stride;
	int c = in_c;

	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad;
	int h_offset = -pad;

	int out_index = j + w*(i + h*(k + c*b));
	float max = -INF;
	int max_i = -1;
	int l, m;
	for (l = 0; l < size; ++l) {
		for (m = 0; m < size; ++m) {
			int cur_h = h_offset + i*stride + l;
			int cur_w = w_offset + j*stride + m;
			int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
				cur_w >= 0 && cur_w < in_w);
			float val = (valid != 0) ? input[index] : -INF;  //INFINITY
			max_i = (val > max) ? index : max_i;
			max = (val > max) ? val : max;
		}
	}
	output[out_index] = max;
	indexes[out_index] = max_i;

}