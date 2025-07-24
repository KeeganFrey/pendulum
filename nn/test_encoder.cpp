#include "encoder.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <random>

// Function to load weights from a binary file
bool load_weights(const std::string& filename, std::vector<float>& weights, int num_elements) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening weight file: " << filename << std::endl;
        return false;
    }
    weights.resize(num_elements);
    infile.read(reinterpret_cast<char*>(weights.data()), num_elements * sizeof(float));
    return infile.good();
}

// Function to generate and save random weights (for testing)
void generate_random_weights(const std::string& filename, int num_elements) {
    std::vector<float> weights(num_elements);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < num_elements; ++i) {
        weights[i] = dis(gen);
    }

    std::ofstream outfile(filename, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(weights.data()), num_elements * sizeof(float));
}


int main() {
    // Dimensions
    const int patch_size = 16;
    const int channels = 3;
    const int flattened_dim = patch_size * patch_size * channels; // 768
    const int embedding_dim = 1024; // Desired output dimension

    // --- Weight Handling ---
    const std::string weight_file = "patch_encoder_weights.bin";
    std::vector<float> weights;
    int num_weight_elements = embedding_dim * flattened_dim;

    // On the first run or for testing, generate random weights
    // In a real scenario, you would load pre-trained weights.
    // generate_random_weights(weight_file, num_weight_elements);

    if (!load_weights(weight_file, weights, num_weight_elements)) {
        std::cerr << "Failed to load weights. Generating new random weights for testing." << std::endl;
        generate_random_weights(weight_file, num_weight_elements);
        if(!load_weights(weight_file, weights, num_weight_elements)){
            std::cerr << "Failed to create or load weights. Exiting." << std::endl;
            return 1;
        }
    }

    // --- Input Data ---
    // Create a sample 16x16x3 image patch (flattened to 768)
    // In a real application, you would extract this from an image.
    std::vector<float> image_patch(flattened_dim);
    for (int i = 0; i < flattened_dim; ++i) {
        image_patch[i] = static_cast<float>(i % 256) / 255.0f; // Example data
    }

    // --- Perform Encoding ---
    std::vector<float> patch_embedding;
    patch_encode(image_patch, patch_embedding, weights);

    // --- Output Results ---
    std::cout << "Successfully encoded image patch." << std::endl;
    std::cout << "Output embedding dimension: " << patch_embedding.size() << std::endl;
    std::cout << "First 10 values of the embedding:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << patch_embedding[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}