#include <wb.h>

int main(int argc, char **argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    const char *inputImageFile;
    float *hostInputImageData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    hostInputImageData = wbImage_getData(inputImage);

    printf("Input image size: %4dx%4dx%1d\n", imageWidth, imageHeight, imageChannels);

    //@@ insert code here
    free(inputImage);
    free(hostInputImageData);

    return 0;
}