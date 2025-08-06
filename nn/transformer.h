#ifndef TRANSFORMER_H
#define TRANSFORMER_H

__global__ void h_layer(const float *input, float *output, const float *w_ud, const float *w_ld, const int ud, const int i_m, const int i_n);
__global__ void transform(const float *input, float *output, const float *K, const float *Q, const float *V, const int i_m, const int i_n);

#endif // TRANSFORMER_H