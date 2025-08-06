#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// Performs the relu operation on the whole m x n matrix
__global__ void relu_all(float *matrix, int m, int n);
__global__ void relu_all_1D(float *matrix, int m, int n);
__global__ void relu_vector(float *matrix_in, float *matrix_out);
__global__ void mask_srd_softmax(float *sqmatrixin, const float sqrtd, const int m);

#endif // ACTIVATIONS_H