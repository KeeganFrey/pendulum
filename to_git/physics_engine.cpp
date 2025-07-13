#include <cmath>
//global variables physics
double sled_state[6] = {0};
double pend_state[6] = {0};

/**
 * Params:
 * sled_state: vector of doubles that represent the following:
 * [x, y, x', y', x'', y'']
 * 
 * pend_state: vector of doubles that represent the following:
 * [θ, θ', x, y, x', y']
 * 
 * force_in: double that provides the force based on the current key presses
 * 
 * delta_t: double that provides the time step length in seconds
 */
void run_step(double *sled_state, double *pend_state, double *force_in, double dt){
    double b = .2; // This in the future might be a global variable. resitance to the sled
    double bt = .2; // This in the future might be a global variable. resistance to the pendulum
    double l = 10; // This may also become a global variable
    double g = 10; // This may also become a global variable

    //euler updates to the sled
    sled_state[4] = force_in[0] - b * sled_state[2];
    sled_state[5] = force_in[1] - b * sled_state[3];
    sled_state[2] += sled_state[4] * dt;
    sled_state[3] += sled_state[5] * dt;

    if(sled_state[2] < 1 && sled_state[2] > -1){
        sled_state[2] = 0;
    }

    sled_state[0] += sled_state[2] * dt;
    sled_state[1] += sled_state[3] * dt;

    //collision check

    //F vector construction
    //F1=θ', F2=θ'', F3=x', F4=y', F5=x_s'', F6=y_s''
    double F[6] = {0};
    F[0] = pend_state[1];
    F[1] = (-1/l)*(sled_state[4] * cos(pend_state[0]) + (sled_state[5] + g) * sin(pend_state[0])) - bt * pend_state[1];
    F[2] = pend_state[4];
    F[3] = pend_state[5];
    F[4] = sled_state[4];
    F[5] = sled_state[5];

    for(int i = 0; i < 6; i++){
        pend_state[i] = pend_state[i] + F[i] * dt;
    }
}