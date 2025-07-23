#include <cmath>
#include <vector>

std::vector<double> discounted_return(double *scores, int size){
    double discount = .97;
    std::vector<double> rval;
    rval.push_back(scores[size-1]);
    for(int i = 2; i <= size; i++){
        rval.push_back(scores[size-i] + discount * rval[i-2]);
    }
    return rval;
}

