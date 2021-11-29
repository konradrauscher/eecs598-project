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

        YData[3*i]  = y;
        CbData[3*i] = cb;
        CrData[3*i] = cr;
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
                            
                            size_t x = i_tile * 8 + i_sum;
                            size_t y = j_tile * 8 + j_sum;

                            float cos1 = std::cos(((2*x+1)*i_tile*pi)/16.0);
                            float cos2 = std::cos(((2*y+1)*j_tile*pi)/16.0);

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
    // compressor.sequential_compress();
    // compressor.parallel_compress();

    float in[64]   = { 255, 255, 255, 255, 255, 255, 255, 255, 
                       255, 255, 255, 255, 255, 255, 255, 255, 
                       255, 255, 255, 255, 255, 255, 255, 255, 
                       255, 255, 255, 255, 255, 255, 255, 255, 
                       255, 255, 255, 255, 255, 255, 255, 255, 
                       255, 255, 255, 255, 255, 255, 255, 255, 
                       255, 255, 255, 255, 255, 255, 255, 255, 
                       255, 255, 255, 255, 255, 255, 255, 255 };

    float out[64];

    compressor.sequential_dct(in, out, 8, 8);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%0.6f\t", out[j*8+i]);
        }
        printf("\n");
    }

    // Should match:

    // 2039.999878    -1.168211    1.190998    -1.230618    1.289227    -1.370580    1.480267    -1.626942    
    // -1.167731       0.000664    -0.000694    0.000698    -0.000748    0.000774    -0.000837    0.000920    
    // 1.191004       -0.000694    0.000710    -0.000710    0.000751    -0.000801    0.000864    -0.000950    
    // -1.230645       0.000687    -0.000721    0.000744    -0.000771    0.000837    -0.000891    0.000975    
    // 1.289146       -0.000751    0.000740    -0.000767    0.000824    -0.000864    0.000946    -0.001026    
    // -1.370624       0.000744    -0.000820    0.000834    -0.000858    0.000898    -0.000998    0.001093    
    // 1.480278       -0.000856    0.000870    -0.000895    0.000944    -0.001000    0.001080    -0.001177    
    // -1.626932       0.000933    -0.000940    0.000975    -0.001024    0.001089    -0.001175    0.001298

    return 0;
}