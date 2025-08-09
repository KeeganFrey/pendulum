#ifndef MODEL_INIT_DATA_H
#define MODEL_INIT_DATA_H
#include "safetensors.hh"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>

template <typename T>

void print_vector(const std::vector<T>& vec);
safetensors::SafeTensors load_and_check(std::string filename);
int collect_inference_output(unsigned char* pixel_data_buffer, int buffer_length, safetensors::SafeTensors st);
#endif