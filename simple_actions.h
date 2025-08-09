#ifndef SIMPLE_ACTIONS_H
#define SIMPLE_ACTIONS_H
#include <random>
extern float n_logits[3] = {0.33, 0.33, 0.34};
extern int num_logits = 3;
std::uniform_real_distribution<float> distribution(0.0, 1.0);

int simulate_choice(std::mt19937 generator);

#endif