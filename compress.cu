#include <wb.h>

__global__ void kernel_float_to_char(float* input, unsigned char* output, uint num) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num) {
        output[idx] = (unsigned char) (255 * input[idx]);
    }

    return;
}

class Compressor {
private:
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    const char *inputImageFile;
    float *hostInputImageData;

    void float_to_char(unsigned char* hostCharData);
    void rgb_to_ycbcr(unsigned char* hostCharData, unsigned char* hostYCbCrData);

public:
    Compressor(wbArg_t args);
    ~Compressor();
    void sequential_compress();
    void parallel_compress();
};

Compressor::Compressor(wbArg_t args) {
    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImageFile = wbArg_getInputFile(args, 0);
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    hostInputImageData = wbImage_getData(inputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    printf("Input image size: %4dx%4dx%1d\n", imageWidth, imageHeight, imageChannels);
}

Compressor::~Compressor() {
    free(inputImage);
    free(hostInputImageData);
}

void Compressor::sequential_compress() {

    // convert from RGB floats to YCbCr unsigned char
    unsigned char* charYCbCrData = (unsigned char *)malloc(imageWidth*imageHeight*imageChannels*sizeof(unsigned char));

    for (size_t i = 0; i < imageWidth*imageHeight; i ++) {

        float r = hostInputImageData[3*i + 0] * 255;
        float g = hostInputImageData[3*i + 1] * 255;
        float b = hostInputImageData[3*i + 2] * 255;

        float y  = 0   + (0.299    * r) + (0.587    * g) + (0.114    * b);
        float cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5      * b);
        float cr = 128 + (0.5      * r) - (0.418688 * g) - (0.081312 * b);

        charYCbCrData[3*i + 0] = y;
        charYCbCrData[3*i + 1] = cb;
        charYCbCrData[3*i + 2] = cr;
    }

    // TODO: subsample chrominance using 4:2:0 subsampling
    unsigned char* subsampledData = (unsigned char *)malloc((imageWidth*imageHeight*imageChannels)/2*sizeof(unsigned char));

    for(size_t i = 0; i < imageHeight*imageWidth/4; i++) {
        subsampledData[i + 0] = 0;
    }

    // TODO: split into blocks

    // TODO: discrete cosine transform

    // TODO: quantization

    // TODO: entropy coding

    // TODO: build file, this can probably be shared with parallel process

    // clean up memory
    free(charYCbCrData);
    free(subsampledData);

}

void Compressor::parallel_compress() {
    unsigned char* hostCharData = (unsigned char *)malloc(imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
    unsigned char* hostYCbCrData = (unsigned char *)malloc(imageWidth*imageHeight*imageChannels*sizeof(unsigned char));

    float_to_char(hostCharData);
    rgb_to_ycbcr(hostCharData, hostYCbCrData);

    free(hostCharData);
    free(hostYCbCrData);
}

void Compressor::float_to_char(unsigned char* hostCharData) {
    wbTime_start(Generic, "Converting image from float to unsigned char");

    float* deviceInputImageData;
    unsigned char* deviceCharData;

    cudaMalloc((void **) &deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
    cudaMalloc((void **) &deviceCharData, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));

    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);

    dim3 DimGrid1((imageWidth*imageHeight*imageChannels-1)/1024 + 1, 1, 1);
    dim3 DimBlock1(1024, 1, 1);

    kernel_float_to_char <<<DimGrid1, DimBlock1>>> (deviceInputImageData, deviceCharData, imageWidth*imageHeight*imageChannels);
    cudaDeviceSynchronize();

    cudaMemcpy(hostCharData, deviceCharData, imageWidth*imageHeight*imageChannels*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(deviceInputImageData);
    cudaFree(deviceCharData);

    wbTime_stop(Generic, "Converting image from float to unsigned char");
}

void Compressor::rgb_to_ycbcr(unsigned char* hostCharData, unsigned char* hostYCbCrData) {
    wbTime_start(Generic, "Converting image from float to unsigned char");


    wbTime_stop(Generic, "Converting image from float to unsigned char");
}


int main(int argc, char **argv) {

    wbArg_t args = wbArg_read(argc, argv);

    Compressor compressor(args);
    compressor.sequential_compress();
    compressor.parallel_compress();

    return 0;
}