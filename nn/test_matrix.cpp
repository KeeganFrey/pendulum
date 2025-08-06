#include "matrix.h"

const int DSIZE = 8192;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

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