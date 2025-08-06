#ifndef ENCODERPIPELINE
#define ENCODERPIPELINE

__global__ void vec_add(float *c, const float *a, const float *b, int n_elements);
std::vector<float[1024]> encode_image(const int *image, const int height, const int width, float *encoder_weights);

#endif