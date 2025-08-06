#include "transformer.h"
#include "matrix.h"
#include "activations.h"
#include "attention_help.h"

//Hidden layer
//input is a column major array of the input to the hidden layer
//output will be a column major array of the output of the hidden layer
//There should probably be some better way to do the weights
void h_layer(const float *input, float *output, const float *w_ud, const float *w_ld, const int ud, const int i_m, const int i_n){

    float *intermediate;
    cudaMalloc(&intermediate, ud * i_n * sizeof(float));

    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 blocks((i_n + threads.x - 1) / threads.x, (ud + threads.y - 1) / threads.y);
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);

    mmul_c<<<blocks, threads, shared_mem_size>>>(w_ud, input, intermediate, block_size, ud, i_n, i_m);
    cudaCheckErrors("kernel execution failure in mm");

    dim3 relu_threads(32, 32);
    dim3 relu_blocks((i_n + relu_threads.x - 1) / relu_threads.x, (ud + relu_threads.y - 1) / relu_threads.y);
    relu_all<<<relu_blocks, relu_threads>>>(d_embedding, ud, i_n);
    cudaCheckErrors("kernel execution failure in ReLU");

    dim3 blocks2((i_n + threads.x - 1) / threads.x, (i_m + threads.y - 1) / threads.y);
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);

    mmul_c<<<blocks2, threads, shared_mem_size>>>(w_ld, intermediate, output, block_size, ud, i_n, i_m);
    cudaCheckErrors("kernel execution failure in mm");

    dim3 relu_blocks2((i_n + relu_threads.x - 1) / relu_threads.x, (i_m + relu_threads.y - 1) / relu_threads.y);
    relu_all<<<relu_blocks2, relu_threads>>>(d_embedding, i_m, i_n);
    cudaCheckErrors("kernel execution failure in ReLU");

    cudaFree(intermediate);
    cudaCheckErrors("cudaFree failure");
}

/**
 * Decoder transformer layer
 * Uses the matrix multiply, ReLU, sum, normalize and mask kernels
 * 
 * 
 */
void transform(const float *input, float *output, const float *K, const float *Q, const float *V, const int i_m, const int i_n){
    // Allocate GPU Memory
    float *kim, *qim, *vim;
    cudaMalloc(&kim, i_m * i_n * sizeof(float));
    cudaMalloc(&qim, i_m * i_n * sizeof(float));
    cudaMalloc(&vim, i_m * i_n * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    
    //K
    const int block_size = 16; // As an example, can be tuned
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);
    dim3 threads(block_size, block_size);
    dim3 blocks((i_n + threads.x - 1) / threads.x, (i_m + threads.y - 1) / threads.y);
    

    mmul_c<<<blocks, threads, shared_mem_size>>>(input, K, kim, block_size, i_m, i_n, i_n);
    cudaCheckErrors("kernel execution failure in mm k");
        
    //Q
    mmul_c<<<blocks, threads, shared_mem_size>>>(input, Q, qim, block_size, i_m, i_n, i_n);
    cudaCheckErrors("kernel execution failure in mm q");
        
    //V
    mmul_c<<<blocks, threads, shared_mem_size>>>(input, V, vim, block_size, i_m, i_n, i_n);
    cudaCheckErrors("kernel execution failure in mm v");

    //create heads
    //this step could probably be skipped with some clever pointer indexing
    const int H = 16;
    int dh = i_m/H;
    int T = i_n;

    float *k_p, *k_pt, *q_p, *v_p, *im_out;
    cudaMalloc(&k_p, dh * i_n * sizeof(float));
    cudaMalloc(&k_pt, dh * i_n * sizeof(float));
    cudaMalloc(&q_p, dh * i_n * sizeof(float));
    cudaMalloc(&v_p, dh * i_n * sizeof(float));
    cudaMalloc(&im_out, dh * i_n * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    dim3 block_head((i_n + threads.x - 1) / threads.x, (dh + threads.y - 1) / threads.y);
    dim3 block_attn((T + threads.x - 1) / threads.x, (T + threads.y - 1) / threads.y);
    dim3 block_v((T + threads.x - 1) / threads.x, (dh + threads.y - 1) / threads.y);

    dim3 threads_mds(1024, 0);
    dim3 blocks_mds((T + threads_mds.x - 1) / threads_mds.x, 0);
    size_t shared_mem_size_mds = 2 * threads_mds.x * sizeof(float);

    const int TILE_DIM = 32;
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((i_n + TILE_DIM - 1) / TILE_DIM, (dh + TILE_DIM - 1) / TILE_DIM);

    for(int i = 0; i < H; i++){
        //create head
        create_head<<<block_head, threads>>>(kim, k_p, i_m, dh, i_n, i, H);
        cudaCheckErrors("k head creation failure");
        create_head<<<block_head, threads>>>(qim, q_p, i_m, dh, i_n, i, H);
        cudaCheckErrors("q head creation failure");
        create_head<<<block_head, threads>>>(vim, v_p, i_m, dh, i_n, i, H);
        cudaCheckErrors("v head creation failure");

        //k*q head
        transpose_col_major<<<dimGrid, dimBlock>>>(k_pt, k_p, dh, i_n);
        cudaCheckErrors("k transpose failure");
        mmul_c<<<block_attn, threads, shared_mem_size>>>(q_p, k_pt, attn, T, T, dh);
        cudaCheckErrors("q * kt failure");

        //attn matrix
        mask_srd_softmax<<<blocks_mds,threads_mds,shared_mem_size_mds>>>(attn, i_m, T);
        cudaCheckErrors("mds failure");
        mmul_c<<<block_v, threads, shared_mem_size>>>(v_p, attn, im_out, block_size, dh, T, T);
        cudaCheckErrors("v * attn failure");

        //add head back to the matrix
        combine<<<block_head, threads>>>(output, im_out, i_m, dh, i_n, i, H);
        cudaCheckErrors("combine failure");
    }
    cudaFree(kim);
    cudaFree(qim);
    cudaFree(vim);
    cudaFree(k_p);
    cudaFree(k_pt);
    cudaFree(q_p);
    cudaFree(v_p);
    cudaFree(im_out);
    cudaCheckErrors("cudaFree failure");
}
