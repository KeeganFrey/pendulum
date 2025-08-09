#include "tblock.h"
#include "model_init_collect_data.h"
// main.cpp

// Define the implementation macro in ONLY ONE .cpp file before including the header.
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

// Helper function to print a vector (like a tensor's shape)
template <typename T>
void print_vector(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]";
}

safetensors::SafeTensors load_and_check(std::string filename){
    safetensors::SafeTensors st;
    std::string warn, err;

    // 1. Load the .safetensors file
    bool ret = safetensors::load_from_file(filename, &st, &warn, &err);

    // Always check for warnings and errors
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load safetensors file: " << filename << std::endl;
        std::cerr << "Error: " << err << std::endl;
        return -1;
    }

    std::cout << "Successfully loaded " << filename << std::endl;
    std::cout << "Found " << st.tensors.size() << " tensors." << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    return st;
}
/*
Somewhere in main or the animation loop
    std::string filename = "model_weights.safetensors";
    safetensors::SafeTensors st = load_and_check(filename);

*/


int collect_inference_output(unsigned char* pixel_data_buffer, int buffer_length, safetensors::SafeTensors st) {

    uint8_t *pixel_data_int = new uint8_t[buffer_length];

    for(int i = 0; i < buffer_length; i++){
        pixel_data_int[i] = static_cast<uint8_t>(pixel_data_buffer[i]);
    }

    std::ofstream images_out("output_images.txt", std::ios::app);
    if (images_out.is_open()) {
        for(int i = 0; i < buffer_length; i++){
            images_out << pixel_data_int[i]
            if (i != buffer_length-1) {
                images_out << " ";
            }
            else {
                images_out << "\n";
            }
        }
        images_out.close();
        std::cout << "Successfully appended to the file.\n";
    } else {
        std::cerr << "Error opening the file!\n";
    }

    // 2. Iterate through the loaded tensors
    for (const auto& tensor_view : st.tensors) {
        std::cout << "Tensor Name: " << tensor_view.name << std::endl;

        // 3. Access Tensor Metadata (Shape and Data Type)
        std::cout << "  - DType: " << safetensors::dtype_str(tensor_view.dtype) << std::endl;
        std::cout << "  - Shape: ";
        print_vector(tensor_view.shape);
        std::cout << std::endl;

        // Calculate the total number of elements from the shape
        size_t num_elements = 1;
        if (tensor_view.shape.empty()) {
            num_elements = 1; // Scalar tensor
        } else {
            num_elements = std::accumulate(tensor_view.shape.begin(), tensor_view.shape.end(), 1, std::multiplies<size_t>());
        }

        // 4. Access the Raw Tensor Data
        // The data is stored in `st.storage` (a std::vector<uint8_t>).
        // The `tensor_view` provides the offset and size for its data within that storage.
        const uint8_t* data_ptr = st.storage.data() + tensor_view.data_offset;
        size_t data_size_bytes = tensor_view.data_size;

        std::cout << "  - Data Size (bytes): " << data_size_bytes << std::endl;

        // Example: How to use the raw data pointer.
        // We cast the raw byte pointer to the correct data type.
        if (tensor_view.dtype == safetensors::DataType::FLOAT32) {
            const float* float_data = reinterpret_cast<const float*>(data_ptr);

            // You can now use this typed pointer to copy the data into your own
            // C++ tensor library (like Eigen, xtensor, or a custom one)
            // or perform calculations directly.

            // Let's just print the first element as a proof-of-concept.
            if (num_elements > 0) {
                std::cout << "  - First element value: " << float_data[0] << std::endl;
            }
        }
        std::cout << "------------------------------------------" << std::endl;
    }
    
    // You can also access the top-level `__metadata__` if it exists
    if (!st.metadata.empty()) {
        std::cout << "Found top-level metadata:" << std::endl;
        for (const auto& meta : st.metadata) {
             std::cout << "  - " << meta.first << ": " << meta.second.as_string() << std::endl;
        }
    }

    //One of these will need to be added for the score
    std::ofstream action_choosen_out("output_images.txt", std::ios::app);
    if (action_choosen_out.is_open()) {
        action_choosen_out << rval << "\n";
        action_choosen_out.close();
        std::cout << "Successfully appended to the file.\n";
    } else {
        std::cerr << "Error opening the file!\n";
    }

    return 0;
}