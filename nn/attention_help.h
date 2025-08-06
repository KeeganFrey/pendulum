#ifndef ATTENTION_HELP_H
#define ATTENTION_HELP_H

// Performs the relu operation on the whole m x n matrix
__global__ void create_head(float *in, float *out, const int i_height, const int o_height, const int width, const int block, const int total);
__global__ void combine(float *combine, flot *fragment, const int c_h, const int f_h, const int width, const int block, const int total);

#endif // ACTIVATIONS_H