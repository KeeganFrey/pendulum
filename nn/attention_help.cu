/**
 * 
 */
__global__ void create_head(float *in, float *out, const int i_height, const int o_height, const int width, const int block, const int total){
    if(threadIdx.x + blockIdx.x * blockDim.x < width && threadIdx.y + blockIdx.y * blockDim.y < o_height){
        out[threadIdx.y + blockIdx.y * blockDim.y + (threadIdx.x + blockIdx.x * blockDim.x) * o_height] = in[threadIdx.y + blockIdx.y * blockDim.y + block * o_height + (threadIdx.x + blockIdx.x * blockDim.x) * i_height];
    }
}

__global__ void combine(float *combine, flot *fragment, const int c_h, const int f_h, const int width, const int block, const int total){
    if(threadIdx.x + blockIdx.x * blockDim.x < width && threadIdx.y + blockIdx.y * blockDim.y < o_height){
        combine[threadIdx.y + blockIdx.y * blockDim.y + block * o_height + (threadIdx.x + blockIdx.x * blockDim.x) * i_height] = fragment[threadIdx.y + blockIdx.y * blockDim.y + (threadIdx.x + blockIdx.x * blockDim.x) * o_height];
    }
}