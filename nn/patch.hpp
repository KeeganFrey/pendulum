int * patcherlinear(int image[], int height, int width, int patchh, int patchw, int patchn){
    int * rval = new int[patchh * patchw * 3];
    for (int i = 0; i < patchh * patchw * 3; i++){
        rval[i] = i+11;
    }
    //first patch is patch 0
    int patch_col = patchn * patchw % width;
    int patch_height = patchn * patchw / width;
     
    for(int i = 0; i < patchh; i++){
        for(int j = 0; j < patchw; j++){
            int pixelindex = ((patch_height * patchh + i) * width + patch_col + j) * 3;
            rval[(i * patchw + j) * 3] = image[pixelindex];
            rval[(i * patchw + j) * 3 + 1] = image[pixelindex + 1];
            rval[(i * patchw + j) * 3 + 2] = image[pixelindex + 2];
        }
    }
    return rval;
}