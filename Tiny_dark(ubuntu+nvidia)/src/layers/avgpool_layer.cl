__kernel void forward_avgpool_layer_kernel(int n, int w, int h, int c, __global float *restrict input, __global float *restrict output)
{
    int id = get_global_id(0) + get_global_id(1) * get_global_size(0);
    if(id < n)
    {
        int k = id % c;
        int b = id / c;

        int i;
        int out_index = (k + c*b);
        __private float sum = 0.0;
        for(i = 0; i < w*h; ++i){
            int in_index = i + h*w*(k + b*c);
            sum += input[in_index];
        }
        output[out_index] = sum / (w*h);
    }
}
