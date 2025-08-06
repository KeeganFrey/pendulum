#ifndef TBLOCK_H
#define TBLOCK_H

__global__ void matrix_add(float *c, const float *a, const float *b, int n_elements);
__global__ void layer_norm_c(float *in, float *out, const int d);
void t_block(float *m, const int W, const int H, const float *K, const float *Q, const float *V, const float *w_ud, const float *w_ld, const int ud);

#endif