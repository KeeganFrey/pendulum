#include "encoder_pipeline.h"
#include "matrix.h"
#include "activations.h" 
#include "patch.h"

#include <vector>

__global__ void vec_add(float *c, const float *a, const float *b, int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements)
    {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * Input: 1D image as a array of ints
 * height of image as an int
 * width of the image as an int
 */
std::vector<float[1024]> encode_image(const int *image, const int height, const int width, float *patch_weights, float *pos_weights, const int time)
{
    //1. setup the variables needed by the following functions
    const int patch_size = 16;
    const int channels = 3;
    const int flattened_dim = patch_size * patch_size * channels; // 16*16*3 = 768
    const int embedding_dim = 1024;

    const int M = embedding_dim;
    const int K = flattened_dim;
    const int N = 1;

    const int pos_vector_size = 3;

    std::vector<float> rval;

    // 2. Allocate GPU Memory
    //For the patches
    float *d_weights, *d_patch, *d_embedding;
    cudaMalloc(&d_weights, M * K * sizeof(float));
    cudaMalloc(&d_patch, K * N * sizeof(float));
    cudaMalloc(&d_embedding, M * N * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    //Send weights to GPU
    cudaMemcpy(d_weights, patch_weights, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    //for the positions
    float *d_weights_pos, *d_pos, *d_embedding_pos;
    cudaMalloc(&d_weights_pos, M * pos_vector_size * sizeof(float));
    cudaMalloc(&d_pos, pos_vector_size * N * sizeof(float));
    cudaMalloc(&d_embedding_pos, M * N * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    //Send weights to GPU
    cudaMemcpy(d_weights_pos, pos_weights, M * pos_vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    //3. Setup the patches
    //Patches will be in squares of 16x16
    //calculates how many patches in the x dimention and y dimention will be made
    int numx = width / patch_size;
    numx = (width % patch_size > 0) ? numx + 1 : numx;
    int numy = height / patch_size;
    numy = (height % patch_size > 0) ? numy + 1 : numy;

    //4. Run the pipeline
    for(int i = 0; i < numx * numy; i++){
        float p_back[M*N] = {0};
        
        int y_pos = (i/numx) + 1;
        int x_pos = (i%numx) + 1;
        int vector[3] = {x_pos,y_pos,time};

        //5. patch the image
        int * patch = patcherlinear(image, height, width, patch_size, patch_size, i);

        // 6. Copy Data from Host to GPU
        cudaMemcpy(d_patch, patch, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy patch failure");

        cudaMemcpy(d_pos, vector, pos_vector_size * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy patch failure");

        // 7. Launch the Matrix Multiplication Kernel
        const int block_size = 16; // As an example, can be tuned
        dim3 threads(block_size, block_size);
        dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
        size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);

        mmul<<<blocks, threads, shared_mem_size>>>(d_weights, d_patch, d_embedding, block_size, M, N, K);
        cudaCheckErrors("kernel execution failure in mm");

        dim3 blocks_pos((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

        mmul<<<blocks_pos, threads, shared_mem_size>>>(d_weights_pos, d_pos, d_embedding_pos, block_size, M, N, K);
        cudaCheckErrors("kernel execution failure in mm");

        // 8. Launch the ReLU Kernel
        dim3 relu_threads(32, 32);
        dim3 relu_blocks((N + relu_threads.x - 1) / relu_threads.x, (M + relu_threads.y - 1) / relu_threads.y);
        relu_all<<<relu_blocks, relu_threads>>>(d_embedding, M, N);
        cudaCheckErrors("kernel execution failure in ReLU");

        relu_all<<<relu_blocks, relu_threads>>>(d_embedding_pos, M, N);
        cudaCheckErrors("kernel execution failure in ReLU");
       
        // 9. Sum the 2 embeddings
        dim3 threads_vec(1024, 0);
        dim3 blocks_vec((embedding_dim + threads_mds.x - 1) / threads_mds.x, 0);
        <<<blocks_vec, threads_vec>>>vec_add(d_embedding, d_embedding, d_embedding_pos, embedding_dim)

        // 10. Copy Result from GPU to Host
        cudaMemcpy(p_back, d_embedding, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy D2H failure");

        // 11. build the matrix
        rval.push_back(p_back);
        delete[] patch;
    }
    
    // 11. Free GPU Memory
    cudaFree(d_weights);
    cudaFree(d_patch);
    cudaFree(d_embedding);
    cudaCheckErrors("cudaFree failure");

    return rval;
}
