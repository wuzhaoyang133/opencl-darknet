__kernel void im2col_kernel(const int n, int offset, __global const float *restrict data_im,   
        const int in_height, 
        const int in_width, 
        const int ksize,
        const int pad,
        const int stride,
        __global float *restrict data_col)
{
    int out_width = (in_width + 2 * pad - ksize) / stride + 1;
    int out_height = (in_height + 2 * pad - ksize) / stride + 1;
    int thread_id = get_global_id(0) + get_global_size(0) * get_global_id(1);
    //printf("thread_id is %d\n", thread_id);
    if(thread_id < n)
    {
        int ow = thread_id % out_width;
        int oh = thread_id / out_width % out_height;
        int ic = thread_id / out_width / out_height;
        int w_offset = -pad;
        int h_offset = -pad;
        float in_val;
        
        for(int j = 0; j < ksize; ++j)
            for(int i = 0; i < ksize; ++i)
            {
                int iw = ow * stride - pad + i;
                int ih = oh * stride - pad + j;
                int in_idx = iw + ih * in_width + ic * in_width * in_height + offset;
                in_val = (iw >= 0 && iw < in_width && ih >= 0 && ih < in_height) ? data_im[in_idx] : 0.0;
                int oc_offset = ksize * ksize * ic * out_width * out_height;
                int out_idx = oc_offset + ow + oh * out_width + (i + ksize * j) * out_width * out_height;
                data_col[out_idx] = in_val;
            }
    }
}
