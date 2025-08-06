#include "patch.h"
#include "matrix.h"
#include "activations.h"
#include "attention_help.h"
#include "encoder_pipeline.h"
#include "transformer.h"
#include "tblock.h"

#include <vector>

__global__ void matrix_add(float *c, const float *a, const float *b, int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements)
    {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * Column major layer norm
 * Hardcoded for embedding dimentions of 1024
 */
__global__ void layer_norm_c(float *in, float *out, const int d){
    int vector = 1024 * blockIdx.x;
    int t = threadIdx.x;
    __shared__ temp[1024] = {0};
    temp[t] = in[vector + t] * in[vector + t];
    __syncthreads();
    for(int i = 1; i < 1024; i*=2){
        if((t + 1) * 2 * i <= 1024){
            temp[(t+1) * 2 * i - 1] = temp[(t+1) * 2 * i - 1 - i] + temp[(t+1) * 2 * i - 1]; 
        }
    }
    __syncthreads();
    out[vector + t] = in[vector + t] / ((1/d) * temp[1023]);
}

void t_block(float *m, const int W, const int H, const float *K, const float *Q, const float *V, const float *w_ud, const float *w_ld, const int ud){
    float *im, *ln;
    cudaMalloc(&im, W * H * sizeof(float));
    cudaMalloc(&ln, W * H * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    
    //H should be 1024
    dim3 threads_ln(H, 0);
    dim3 blocks_ln(W, 0);

    layer_norm_c<<<blocks_ln,threads_ln>>>(m, ln, H);
    transform(ln, im, K, Q, V, W, H);
    matrix_add<<<blocks_ln,threads_ln>>>(m, im, m, W * H);

    layer_norm_c<<<blocks_ln,threads_ln>>>(m, ln, H);
    h_layer(ln, im, w_ud, w_ld, ud, W, H);
    matrix_add<<<blocks_ln,threads_ln>>>(m, im, m, W * H);
}

std::vector<float> transformer(std::vector<float> *previous_embeddings, float *enc_patch_weights, int *next_image, float *enc_pos_weights, float *decoder_weights,
                               float *K_1,
                               float *K_2,
                               float *K_3,
                               float *K_4,
                               float *K_5,
                               float *Q_1
                               float *Q_2
                               float *Q_3
                               float *Q_4
                               float *Q_5
                               float *V_1
                               float *V_2
                               float *V_3
                               float *V_4
                               float *V_5
                               float *w_ud_1
                               float *w_ud_2
                               float *w_ud_3
                               float *w_ud_4
                               float *w_ud_5
                               float *w_ld_1
                               float *w_ld_2
                               float *w_ld_3
                               float *w_ld_4
                               float *w_ld_5
                               int num_output_options, int embedding_dim, int max_tokens, int time, int up_dimension, int image_height, int image_width){
    float *d_matrix;
    cudaMalloc(&d_matrix, max_tokens * embedding_dim * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    //These smaller collections of weights are the vector form on CPU
    std::vector<float[1024]> embeddings = encode_image(next_image, image_height, image_width, enc_patch_weights, enc_pos_weights, time);

    std::vector<float[1024]> tokens;

    for(int i = 0; i < embeddings.size(); i++){
        (*previous_embeddings).push_back(embeddings[i]);
    }

    for(int i = 0; i < previous_embeddings.size(); i++){
        (*tokens).push_back(previous_embeddings[i]);
    }

    for(int i = 0; i < max_tokens - tokens.size(); i++){
        float temp[1024] = {0};
        if(i == 0){
            for(int j = 0; j < 1024; j++){
                temp[j] = 0
            }
        }
        tokens.push_back(temp);
    }

    //token values need to be loaded into d_matrix

    //Everything passed already points to GPU memory if a vector
    t_block(d_matrix, max_tokens, embedding_dim, K_1, Q_1, V_1, w_ud_1, w_ld_1, up_dimension);
    t_block(d_matrix, max_tokens, embedding_dim, K_2, Q_2, V_2, w_ud_2, w_ld_2, up_dimension);
    t_block(d_matrix, max_tokens, embedding_dim, K_3, Q_3, V_3, w_ud_3, w_ld_3, up_dimension);
    t_block(d_matrix, max_tokens, embedding_dim, K_4, Q_4, V_4, w_ud_4, w_ld_4, up_dimension);
    t_block(d_matrix, max_tokens, embedding_dim, K_5, Q_5, V_5, w_ud_5, w_ld_5, up_dimension);
    
    //Decoder logits to the GPU
    float *logits, *d_decoder_weights;
    cudaMalloc(&logits, num_output_options * sizeof(float));
    cudaMalloc(&d_decoder_weights, num_output_options * embedding_dim * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_decoder_weights, decoder_weights, num_output_options * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy patch failure");

    //decoder calculation
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 blocks((1 + threads.x - 1) / threads.x, (num_output_options + threads.y - 1) / threads.y);
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);
    //This isn't correct, needs to be the embedding previous_embedings.size() + 1 for the matrix multiply in
    //might need to write a new kernel to get that embedding vector
    mmul_c<<<blocks, threads, shared_mem_size>>>(decoder_weights, d_matrix, logits, block_size, num_output_options, 1, embedding_dim);
    cudaCheckErrors("kernel execution failure in mm");

    float l_back[] = new float[num_output_options];
    cudaMemcpy(l_back, logits, num_output_options * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    std::vector<float> rval;
    for(int i = 0; i < num_output_options; i++){
        rval.push_back(l_back[i]);
    }
    delete[] l_back;
    return rval;
}
