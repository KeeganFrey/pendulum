using namespace nvcuda;

#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 8192;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, const int block_size, const int M, const int N, const int K)
{
    extern __shared__ float s[];
    float dot_product = 0.0f;
    for (int tile_l = 0; tile_l < (K + block_size -1) / block_size; tile_l++) {
        s[threadIdx.y*block_size + threadIdx.x] = (blockIdx.y * blockDim.y + threadIdx.y < M && tile_l * blockDim.x + threadIdx.x < K) ? A[(blockIdx.y * blockDim.y + threadIdx.y)*K + tile_l * blockDim.x + threadIdx.x] : 0.0f; //Loads the A matrix section to the first part of s
        s[block_size*block_size + threadIdx.y * block_size + threadIdx.x] = (tile_l * blockDim.y + threadIdx.y < K && blockIdx.x * blockDim.x + threadIdx.x < N) ? B[(tile_l * blockDim.y + threadIdx.y)*N + blockDim.x * blockIdx.x + threadIdx.x] : 0.0f; //Loads the B matrix section to the second part of s
        __syncthreads();
        for (int k = 0; k < block_size; ++k) {
            dot_product += s[threadIdx.y*block_size + k] * s[block_size*block_size + k * block_size + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x + blockIdx.x * blockDim.x < N && threadIdx.y + blockIdx.y * blockDim.y < M) {
        C[(blockIdx.y * blockDim.y + threadIdx.y) * N + blockDim.x * blockIdx.x + threadIdx.x] = dot_product;
    }
}

// matrix multiply (naive) kernel: C = A * B
//for column major
__global__ void mmul_c(const float *A, const float *B, float *C, const int block_size, const int M, const int N, const int K)
{
    extern __shared__ float s[];
    float dot_product = 0.0f;
    for (int tile_l = 0; tile_l < (K + block_size -1) / block_size; tile_l++) {
        s[threadIdx.y + block_size * threadIdx.x] = (blockIdx.y * blockDim.y + threadIdx.y < M && tile_l * blockDim.x + threadIdx.x < K) ? A[blockIdx.y * blockDim.y + threadIdx.y + (tile_l * blockDim.x + threadIdx.x) * M] : 0.0f; //Loads the A matrix section to the first part of s
        s[block_size*block_size + threadIdx.y * block_size + threadIdx.x] = (tile_l * blockDim.y + threadIdx.y < K && blockIdx.x * blockDim.x + threadIdx.x < N) ? B[tile_l * blockDim.y + threadIdx.y + (blockDim.x * blockIdx.x + threadIdx.x) * K] : 0.0f; //Loads the B matrix section to the second part of s
        __syncthreads();
        for (int k = 0; k < block_size; ++k) {
            dot_product += s[threadIdx.y + block_size * k] * s[block_size*block_size + k * block_size + threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x + blockIdx.x * blockDim.x < N && threadIdx.y + blockIdx.y * blockDim.y < M) {
        C[blockIdx.y * blockDim.y + threadIdx.y + (blockDim.x * blockIdx.x + threadIdx.x) * M] = dot_product;
    }
}
// --- Verification and Test Code ---

// CPU implementation for row-major matrix multiplication
void cpu_mmul_row_major(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// CPU implementation for column-major matrix multiplication
void cpu_mmul_col_major(const float *A, const *B, float *C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                // A[row, i] * B[i, col]
                sum += A[row + i * M] * B[i + col * K];
            }
            // C[row, col]
            C[row + col * M] = sum;
        }
    }
}
// Function to verify results against CPU reference
void verify_results(const float *gpu_result, const float *cpu_result, int M, int N) {
    int error_count = 0;
    const float epsilon = 1e-4f;

    for (int i = 0; i < M * N; ++i) {
        if (std::abs(gpu_result[i] - cpu_result[i]) > epsilon) {
            if (error_count < 10) { // Print first 10 mismatches
                fprintf(stderr, "Mismatch at index %d: GPU result = %f, CPU result = %f\n", i, gpu_result[i], cpu_result[i]);
            }
            error_count++;
        }
    }

    if (error_count == 0) {
        printf("Success! The results are correct.\n");
    } else {
        printf("FAILED! There were %d mismatches.\n", error_count);
    }
}



int main(){

  float *h_Ar, *h_Br, *h_Cr, *d_Ar, *d_Br, *d_Cr, *h_Ac, *h_Bc, *h_Cc, *d_Ac, *d_Bc, *d_Cc;

  int R = DSIZE, K = DSIZE, C = DSIZE;

  h_Ar = new float[R*K];
  h_Br = new float[K*C];
  h_Cr = new float[R*C];
  h_Ac = new float[R*K];
  h_Bc = new float[K*C];
  h_Cc = new float[R*C];

  // For row-major
  for (int i = 0; i < M; ++i) for (int j = 0; j < K; ++j) h_Ar[i * K + j] = static_cast<float>(i + 1);
  for (int i = 0; i < K; ++i) for (int j = 0; j < N; ++j) h_Br[i * N + j] = static_cast<float>(j + 1);
  // For column-major
  for (int i = 0; i < M; ++i) for (int j = 0; j < K; ++j) h_Ac[i + j * M] = static_cast<float>(i + 1);
  for (int i = 0; i < K; ++i) for (int j = 0; j < N; ++j) h_Bc[i + j * K] = static_cast<float>(j + 1);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_Ar, R*K*sizeof(float));
  cudaMalloc(&d_Br, K*C*sizeof(float));
  cudaMalloc(&d_Cr, R*C*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_Ar, h_Ar, R*K*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Br, h_Br, K*C*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_Ac, R*K*sizeof(float));
  cudaMalloc(&d_Bc, K*C*sizeof(float));
  cudaMalloc(&d_Cc, R*C*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_Ac, h_Ac, R*K*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bc, h_Bc, K*C*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((C+block.x-1)/block.x, (R+block.y-1)/block.y);
  mmul<<<grid, block, 2 * sizeof(float) * block_size * block_size>>>(d_Ar, d_Br, d_Cr, block_size, R, C, K);

  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((C+block.x-1)/block.x, (R+block.y-1)/block.y);
  mmul_c<<<grid, block, 2 * sizeof(float) * block_size * block_size>>>(d_Ac, d_Bc, d_Cc, block_size, R, C, K);

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_Cr, d_Cr, R*C*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Cc, d_Cc, R*C*sizeof(float), cudaMemcpyDeviceToHost);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  // --- Verification ---
  printf("\nVerifying row-major result...\n");
  cpu_mmul_row_major(h_Ar, h_Br, h_Cr_cpu, M, N, K);
  verify_results(h_Cr_gpu, h_Cr_cpu, M, N);  

  printf("\nVerifying column-major result...\n");
  cpu_mmul_col_major(h_Ac, h_Bc, h_Cc_cpu, M, N, K);
  verify_results(h_Cc_gpu, h_Cc_cpu, M, N);
  printf("Success!\n");
  

  cudaFree(d_Ar);
  cudaFree(d_Br);
  cudaFree(d_Cr);
  cudaFree(d_Ac);
  cudaFree(d_Bc);
  cudaFree(d_Cc);
  delete[] h_Ar;
  delete[] h_Br;
  delete[] h_Cr;
  delete[] h_Ac;
  delete[] h_Bc;
  delete[] h_Cc;
  return 0;
}


/*

#include <mma.h>
#include <cuda_bf16.h>

__global__ void matrixMultiplyKernel(half *A, half *B, half *C, int M, int N, int K) {
    // Tile sizes (example)
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;

    // Shared memory for tiles
    __shared__ half sharedA[TILE_M * TILE_K];
    __shared__ half sharedB[TILE_K * TILE_N];
    __shared__ half sharedC[TILE_M * TILE_N]; // Accumulate here first

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // WMMA fragment types
    using MatrixAFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_M, TILE_K, 16, nvcuda::wmma::row_major>;
    using MatrixBFragment = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_K, TILE_N, 16, nvcuda::wmma::col_major>;
    using AccumulatorFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_M, TILE_N, 16>;

    // Initialize accumulator fragment
    AccumulatorFragment accumulatorFragment;
    nvcuda::wmma::fill_fragment(accumulatorFragment, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += TILE_K) {
        // Load tiles from global memory into shared memory
        int aRow = blockRow * TILE_M + threadRow;
        int aCol = k + threadCol;
        int bRow = k + threadRow;
        int bCol = blockCol * TILE_N + threadCol;

        if (aRow < M && aCol < K) {
            sharedA[threadRow * TILE_K + threadCol] = A[aRow * K + aCol];
        } else {
            sharedA[threadRow * TILE_K + threadCol] = 0.0f; // Pad if out of bounds
        }
        if (bRow < K && bCol < N) {
            sharedB[threadRow * TILE_N + threadCol] = B[bRow * N + bCol];
        } else {
            sharedB[threadRow * TILE_N + threadCol] = 0.0f; // Pad if out of bounds
        }
        __syncthreads(); // Ensure all threads load before proceeding

        // Load matrix fragments
        MatrixAFragment matrixAFragment;
        MatrixBFragment matrixBFragment;
        nvcuda::wmma::load_matrix_sync(matrixAFragment, sharedA + threadRow * TILE_K, TILE_K);
        nvcuda::wmma::load_matrix_sync(matrixBFragment, sharedB + threadRow * TILE_N, TILE_N);

        // Perform WMMA operation
        nvcuda::wmma::mma_sync(accumulatorFragment, matrixAFragment, matrixBFragment, accumulatorFragment);

        __syncthreads(); // Ensure all WMMA operations finish
    }

    // Store results back to shared memory
    nvcuda::wmma::store_matrix_sync(sharedC + threadRow * TILE_N + threadCol, accumulatorFragment, TILE_N, nvcuda::wmma::mem_row_major);

    __syncthreads();

    // Store shared memory tile to global memory
    int cRow = blockRow * TILE_M + threadRow;
    int cCol = blockCol * TILE_N + threadCol;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sharedC[threadRow * TILE_N + threadCol];
    }
}

// Define the matrix dimensions for a warp-level operation
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int B = 1;
//Two ways of loading memory:
//Move the 16xk and kx16 chunks to shared memory before the muliply
//Move the 16x16 and 16x16 chunks to shared memory during the muliply
//First I will try option 2
__global__ void matrix_multiply(const float *a, const float *b, float *c, const int K, const int R, const int C) {
    //A and B will both have row major ordering
    //A is a k by 16 matrix
    //B is a 16 by k matrix
    __shared__ __nv_bfloat16 A_row[B * 16 * 16] = {0};
    __shared__ __nv_bfloat16 B_col[B * 16 * 16] = {0};

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    int A_row = blockIdx.x % C / 16;
    int B_col = blockIdx.x * 16 / C;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    for(int i = 0; i < K; i += 16){
        for(int j = 0; j < B; j++){
            A_row[B * (16 * threadIdx.y + threadIdx.x) + j] = __float2bfloat16(a[B * K * 16 * (A_row + threadIdx.y) + B * threadIdx.x + j]);
            B_col[B * (16 * threadIdx.y + threadIdx.x) + j] = __float2bfloat16(b[B * C * 16 * (A_row + threadIdx.y) + B * threadIdx.x + j]);
        }
        __syncthreads(); // Synchronize before loading the next tile

        // Load sub-tiles from shared memory into fragments
        nvcuda::wmma::load_matrix_sync(a_frag, &A_row[0][0], 16 * 16 * 16);
        nvcuda::wmma::load_matrix_sync(b_frag, &B_col[0][0], 16 * 16 * 16);

        // Perform the matrix multiply-accumulate
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads(); // Synchronize before loading the next tile
   }
   // Store the output
   float c_ptr = &(c[B * (C * A_val + B_col * 16)]);
   wmma::store_matrix_sync(c_ptr, c_frag, B*C, wmma::mem_row_major);
}



  dim3 blockmm(8, 4);
  dim3 gridmm(C/16 * R/16);
  matrix_multiply<<<gridmm, blockmm>>>(d_A, d_B, d_C2, K, R, C);
  cudaCheckErrors("kernel launch failure");



*/