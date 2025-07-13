#include <armadillo> 
#include <cmath>

arma::mat translation(double tx, double ty){
    arma::mat temp_matrix(3,3,arma::fill::eye);
    temp_matrix(0,2) = tx;
    temp_matrix(1,2) = ty;
    return temp_matrix;
}

arma::mat scale(double sx, double sy){
    arma::mat temp_matrix(3,3,arma::fill::eye);
    temp_matrix(0,0) = sx;
    temp_matrix(1,1) = sy;
    return temp_matrix;
}

arma::mat projection(double width, double height){
    arma::mat temp_matrix(3,3,arma::fill::eye);
    temp_matrix(0,0) = 2.0/width;
    temp_matrix(0,2) = -1;
    temp_matrix(1,1) = -2.0/height;
    temp_matrix(1,2) = 1;
    return temp_matrix;
}

arma::mat rotation(double angle){
    float c = cos(angle);
    float s = sin(angle);
    arma::mat temp_matrix(3,3,arma::fill::eye);
    temp_matrix(0,0) = c;
    temp_matrix(0,1) = -s;
    temp_matrix(1,0) = c;
    temp_matrix(1,1) = s;
    return temp_matrix;
}
