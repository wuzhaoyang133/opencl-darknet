float leaky_activate_kernel(float x){return (x>0) ? x : .1*x;}
float relu_activate_kernel(float x){return x*(x>0);}
float linear_activate_kernel(float x) {return x;}
float logistic_activate_kernel(float x) { return 1. / (1. + exp(-x)); }
float activate_kernel(float x, int a)
{
    switch(a){
		
        case 0: //LINEAR
            return linear_activate_kernel(x);

        case 1: //LOGISTIC
            return logistic_activate_kernel(x);
        //case 2: //LOGGY
        //    return loggy_activate_kernel(x);
		
        case 3: //RELU
            return relu_activate_kernel(x);
		
        //case 4: //ELU
        //    return elu_activate_kernel(x);
        //case 5: //RELIE
        //    return relie_activate_kernel(x);
        //case 6: //RAMP
        //    return ramp_activate_kernel(x);
		
        case 7: //LEAKY
            return leaky_activate_kernel(x);
		
        //case 8: //TANH
        //    return tanh_activate_kernel(x);
        //case 9: //PLSE
        //    return plse_activate_kernel(x);
        //case 10: //STAIR
        //    return stair_activate_kernel(x);
        //case 11: //HARDTAN
        //    return hardtan_activate_kernel(x);
        //case 12: //LHTAN
        //    return lhtan_activate_kernel(x);
		
    }
    return 0;
}

__kernel void activate_array_kernel(__global float *x, int n, int a)
{
    int i = get_global_id(0) + get_global_id(1) * get_global_size(0);
    if(i < n) 
        x[i] = activate_kernel(x[i], a);
}