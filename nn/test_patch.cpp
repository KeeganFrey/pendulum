#include <iostream>
#include "patch.hpp"

int main(){
    int image[9*16*3] = {0};
    for(int i = 0; i<9; i++){
        for(int j = 0; j < 16; j++){
            image[i * 16 * 3 + j * 3 + 0] = i;
            image[i * 16 * 3 + j * 3 + 1] = j;
            image[i * 16 * 3 + j * 3 + 2] = 0;
        }
    }
    int w = 16, h = 9, pw = 4, ph = 3, pn = 2;
    int *lpatch = patcherlinear(image,h,w,ph,pw,pn);
    // Corrected printing loop
    for(int k = 0; k < ph; k++){      // k is the patch row
        for(int j = 0; j < 3; j++){   // j is the color channel (0=R, 1=G, 2=B)
            for(int i = 0; i < pw; i++){ // i is the patch column
                // Corrected index calculation
                std::cout << lpatch[(k * pw + i) * 3 + j] << "    ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    delete[] lpatch;
}