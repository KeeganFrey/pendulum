#include <random>
#include "simple_actions.h"

float n_logits[3] = {0.33, 0.33, 0.34};
int num_logits = 3;

int simulate_choice(std::mt19937 generator){
    int rval = -1;
    float value = distribution(generator);
    float sum = 0.0f;
    for(int i = 0; i < num_logits; i++){
        sum += n_logits[i];
        if (sum > value){
            rval = i;
            break;
        }
    }
    return rval;
}