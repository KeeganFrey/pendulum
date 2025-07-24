#ifndef ENCODER_H
#define ENCODER_H

#include <vector>

// Function to perform patch encoding on the GPU
void patch_encode(const std::vector<float>& image_patch, std::vector<float>& patch_embedding, const std::vector<float>& weights);
__global__ void add_positional_encoding(float *patch_embeddings, const float *positional_embeddings, int num_patches, int embedding_dim);

#endif // ENCODER_H