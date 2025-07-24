#include "encoder.h"
#include "matrix_ops.h" // Assuming your mmul_c is declared in here
#include "activations.h" // Assuming your relu_all is declared in here
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Helper function to check for CUDA errors
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void add_positional_encoding(float *patch_embeddings, const float *positional_embeddings, int num_patches, int embedding_dim) {
   int patch_idx = blockIdx.x;
   int dim_idx = threadIdx.x;

   if (patch_idx < num_patches && dim_idx < embedding_dim) {
       int index = patch_idx * embedding_dim + dim_idx;
       patch_embeddings[index] += positional_embeddings[index];
   }
}
/*
Lauched with the following:
// After patch_encode has produced d_embedding
int num_patches = (image_height / patch_size) * (image_width / patch_size);
dim3 threads(embedding_dim);
dim3 blocks(num_patches);

add_positional_encoding<<<blocks, threads>>>(d_embedding, d_positional_embeddings, num_patches, embedding_dim);
*/


void patch_encode(const std::vector<float>& image_patch, std::vector<float>& patch_embedding, const std::vector<float>& weights) {
    // 1. Define Matrix Dimensions
    const int patch_size = 16;
    const int channels = 3;
    const int flattened_dim = patch_size * patch_size * channels; // 16*16*3 = 768
    const int embedding_dim = 1024;

    // A (weights) is embedding_dim x flattened_dim (1024 x 768)
    // B (image_patch) is flattened_dim x 1 (768 x 1)
    // C (patch_embedding) is embedding_dim x 1 (1024 x 1)
    const int M = embedding_dim;
    const int N = 1;
    const int K = flattened_dim;

    // 2. Allocate GPU Memory
    float *d_weights, *d_patch, *d_embedding;
    HANDLE_ERROR(cudaMalloc(&d_weights, M * K * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_patch, K * N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_embedding, M * N * sizeof(float)));

    // 3. Copy Data from Host to GPU
    HANDLE_ERROR(cudaMemcpy(d_weights, weights.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_patch, image_patch.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Launch the Matrix Multiplication Kernel
    const int block_size = 16; // As an example, can be tuned
    dim3 threads(block_size, block_size);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);

    mmul_c<<<blocks, threads, shared_mem_size>>>(d_weights, d_patch, d_embedding, block_size, M, N, K);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // 5. Launch the ReLU Kernel
    dim3 relu_threads(32, 32);
    dim3 relu_blocks((N + relu_threads.x - 1) / relu_threads.x, (M + relu_threads.y - 1) / relu_threads.y);
    relu_all<<<relu_blocks, relu_threads>>>(d_embedding, M, N);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // 6. Copy Result from GPU to Host
    patch_embedding.resize(M * N);
    HANDLE_ERROR(cudaMemcpy(patch_embedding.data(), d_embedding, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Free GPU Memory
    HANDLE_ERROR(cudaFree(d_weights));
    HANDLE_ERROR(cudaFree(d_patch));
    HANDLE_ERROR(cudaFree(d_embedding));
}