#include <wb.h>

#define pi 3.142857

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
    float dct[8][8];

    void float_to_char(unsigned char* hostCharData);
    void rgb_to_ycbcr(unsigned char* hostCharData, unsigned char* hostYCbCrData);

public:
    void sequential_dct(float* inputData, float* outputData, int width, int height);
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

    wbTime_start(Generic, "Generating Discrete Cosine Transform lookup");
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 8; j++) {
            dct[i][j] = std::cos(((2*i+1)*j*pi)/16.0);
        }
    }
    wbTime_stop(Generic, "Generating Discrete Cosine Transform lookup");
}

Compressor::~Compressor() {
    free(inputImage);
    free(hostInputImageData);
}

void Compressor::sequential_compress() {

    // convert from RGB floats to YCbCr in separate channels
    float* YData  = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CbData = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CrData = (float *)malloc(imageWidth*imageHeight*sizeof(float));

    for (size_t i = 0; i < imageWidth*imageHeight; i++) {

        float r = hostInputImageData[3*i + 0] * 255;
        float g = hostInputImageData[3*i + 1] * 255;
        float b = hostInputImageData[3*i + 2] * 255;

        float y  = 0   + (0.299    * r) + (0.587    * g) + (0.114    * b);
        float cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5      * b);
        float cr = 128 + (0.5      * r) - (0.418688 * g) - (0.081312 * b);

        YData[i]  = y;
        CbData[i] = cb;
        CrData[i] = cr;
    }

    // TODO: subsample chrominance using 4:2:0 subsampling

    // TODO: discrete cosine transform
    float* YDctData  = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CbDctData = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CrDctData = (float *)malloc(imageWidth*imageHeight*sizeof(float));

    sequential_dct(YData, YDctData, imageWidth, imageHeight);
    sequential_dct(CbData, CbDctData, imageWidth, imageHeight);
    sequential_dct(CrData, CrDctData, imageWidth, imageHeight);

    // TODO: quantization

    // TODO: entropy coding

    // TODO: build file, this can probably be shared with parallel process

    // clean up memory
    free(YData);
    free(CbData);
    free(CrData);
    free(YDctData);
    free(CbDctData);
    free(CrDctData);
}

void Compressor::sequential_dct(float* inputData, float* outputData, int width, int height) {
    // loop over all 8x8 tiles
    for (size_t i_block = 0; i_block < width/8; i_block++) {
        for (size_t j_block = 0; j_block < height/8; j_block++) {

            // loop within an 8x8 tile
            for (size_t i_tile = 0; i_tile < 8; i_tile++) {
                
                size_t i = i_block * 8 + i_tile; // overall i index in image

                for (size_t j_tile = 0; j_tile < 8; j_tile++) {

                    size_t j = j_block * 8 + j_tile; // overall j index in image

                    float c_i = (i_tile == 0) ? (std::sqrt(2.0)/2.0) : 1.0;
                    float c_j = (j_tile == 0) ? (std::sqrt(2.0)/2.0) : 1.0;

                    float sum = 0;

                    // loop within an 8x8 tile again to generate sum
                    for (size_t i_sum = 0; i_sum < 8; i_sum++) {
                        for (size_t j_sum = 0; j_sum < 8; j_sum++) {
                        
                            float cos1 = dct[i_sum][i_tile];
                            float cos2 = dct[j_sum][j_tile];

                            size_t x = i_block * 8 + i_sum;
                            size_t y = j_block * 8 + j_sum;

                            sum += (inputData[y * width + x] * cos1 * cos2);
                        }
                    }

                    outputData[j * width + i] = 0.25 * c_i * c_j * sum;
                }
            }
        }
    }
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
    // compressor.parallel_compress();

    printf("Done!");

    return 0;
}