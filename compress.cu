#include <wb.h>

#define pi 3.142857f 
#define sqrt_2 0.707106781f

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
    const float dct[8][8] = {
        { 1.00000000,  0.98076987,  0.92381907,  0.83133787,  0.70688325,  0.55524170,  0.38224527,  0.19454771 },	
        { 1.00000000,  0.83133787,  0.38224527, -0.19578782, -0.70777708, -0.98101586, -0.92333424, -0.55418950 },
        { 1.00000000,  0.55524170, -0.38341337, -0.98101586, -0.70598841,  0.19702761,  0.92478424,  0.82992971 },
        { 1.00000000,  0.19454771, -0.92430240, -0.55418950,  0.70866978,  0.82992971, -0.38574794, -0.98002243 },
        { 1.00000000, -0.19578782, -0.92333424,  0.55734301,  0.70509231, -0.83344024, -0.37873754,  0.98174447 },
        { 1.00000000, -0.55629271, -0.38107678,  0.98027313, -0.70956099, -0.19082600,  0.92187083, -0.83483505 },
        { 1.00000000, -0.83204001,  0.38458106,  0.19206657, -0.70419478,  0.97977030, -0.92622089,  0.56153554 },
        { 1.00000000, -0.98101586,  0.92478424, -0.83344024,  0.71045172, -0.56048846,  0.38924527, -0.20322227 }
    };
    const uint8_t Q_l[8][8] = {
        { 16, 11, 10, 16, 24, 40, 51, 61 },
        { 12, 12, 14, 19, 26, 58, 60, 55 },
        { 14, 13, 16, 24, 40, 57, 69, 56 },
        { 14, 17, 22, 29, 51, 87, 80, 62 },
        { 18, 22, 37, 56, 68,109,103, 77 },
        { 24, 35, 55, 64, 81,104,113, 92 },
        { 49, 64, 78, 87,103,121,120,101 },
        { 72, 92, 95, 98,112,100,103, 99 }
    };
    const uint8_t Q_c[8][8] = {
        { 17, 18, 24, 47, 99, 99, 99, 99 },
        { 18, 21, 26, 66, 99, 99, 99, 99 },
        { 24, 26, 56, 99, 99, 99, 99, 99 },
        { 47, 66, 99, 99, 99, 99, 99, 99 },
        { 99, 99, 99, 99, 99, 99, 99, 99 },
        { 99, 99, 99, 99, 99, 99, 99, 99 },
        { 99, 99, 99, 99, 99, 99, 99, 99 },
        { 99, 99, 99, 99, 99, 99, 99, 99 }
    };
    const uint8_t zigzag_map[64] = {  
        0,  1,  5,  6,  14, 15, 27, 28,
        2,  4,  7,  13, 16, 26, 29, 42,
        3,  8,  12, 17, 25, 30, 41, 43,
        9,  11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63  
    };

    void float_to_char(unsigned char* hostCharData);
    void rgb_to_ycbcr(unsigned char* hostCharData, unsigned char* hostYCbCrData);
    void sequential_dct(float* inputData, float* outputData, int width, int height);

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

    // convert from RGB floats to YCbCr in separate channels
    float* YData  = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CbData = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CrData = (float *)malloc(imageWidth*imageHeight*sizeof(float));

    for (size_t i = 0; i < imageWidth*imageHeight; i++) {

        float r = hostInputImageData[3*i + 0] * 255;
        float g = hostInputImageData[3*i + 1] * 255;
        float b = hostInputImageData[3*i + 2] * 255;

        YData[i]  = 0   + (0.299   * r) + (0.587   * g) + (0.114   * b);
        CbData[i] = 128 - (0.16874 * r) - (0.33126 * g) + (0.5     * b);
        CrData[i] = 128 + (0.5     * r) - (0.41869 * g) - (0.08131 * b);
    }

    // TODO: subsample chrominance using 4:2:0 subsampling

    // discrete cosine transform
    float* YDctData  = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CbDctData = (float *)malloc(imageWidth*imageHeight*sizeof(float));
    float* CrDctData = (float *)malloc(imageWidth*imageHeight*sizeof(float));

    sequential_dct(YData, YDctData, imageWidth, imageHeight);
    sequential_dct(CbData, CbDctData, imageWidth, imageHeight);
    sequential_dct(CrData, CrDctData, imageWidth, imageHeight);

    // quantization
    unsigned char* YQData  = (unsigned char *)malloc(imageWidth*imageHeight*sizeof(unsigned char));
    unsigned char* CbQData = (unsigned char *)malloc(imageWidth*imageHeight*sizeof(unsigned char));
    unsigned char* CrQData = (unsigned char *)malloc(imageWidth*imageHeight*sizeof(unsigned char));

    for (size_t j_block = 0; j_block < imageHeight/8; j_block++) {
        for (size_t i_block = 0; i_block < imageWidth/8; i_block++) {
            for (size_t j_tile = 0; j_tile < 8; j_tile++) {
                for (size_t i_tile = 0; i_tile < 8; i_tile++) {

                    size_t i = i_block * 8 + i_tile;
                    size_t j = j_block * 8 + j_tile;

                    YQData [j * imageWidth + i] = (unsigned char) round(YDctData [j * imageWidth + i] / Q_l[j_tile][i_tile]);
                    CbQData[j * imageWidth + i] = (unsigned char) round(CbDctData[j * imageWidth + i] / Q_c[j_tile][i_tile]);
                    CrQData[j * imageWidth + i] = (unsigned char) round(CrDctData[j * imageWidth + i] / Q_c[j_tile][i_tile]);   
                }
            }
        }
    }

    //TODO: zigzag rearrange
    unsigned char* YRearrangedData  = (unsigned char *)malloc(imageWidth*imageHeight*sizeof(unsigned char));
    unsigned char* CbRearrangedData = (unsigned char *)malloc(imageWidth*imageHeight*sizeof(unsigned char));
    unsigned char* CrRearrangedData = (unsigned char *)malloc(imageWidth*imageHeight*sizeof(unsigned char));

    for (size_t j_block = 0; j_block < imageHeight/8; j_block++) {
        for (size_t i_block = 0; i_block < imageWidth/8; i_block++) {

            size_t block_num = j_block * imageWidth/8 + i_block;

            for (int j_tile = 0; j_tile < 8; j_tile++) {
                for (int i_tile = 0; i_tile < 8; i_tile++) {

                    size_t tile_num = j_tile * 8 + i_tile;

                    size_t x = i_block * 8 + i_tile;
                    size_t y = j_block * 8 + j_tile;

                    YRearrangedData [block_num * 64 + zigzag_map[tile_num]] = YQData [y * imageWidth + x];
                    CbRearrangedData[block_num * 64 + zigzag_map[tile_num]] = CbQData[y * imageWidth + x];
                    CrRearrangedData[block_num * 64 + zigzag_map[tile_num]] = CrQData[y * imageWidth + x];

                }
            }
        }
    }

    // TODO: entropy coding

    // TODO: build file, this can probably be shared with parallel process
    // I think the quantization table info is neaded in the header.
    // Not sure exactly how the data is setup yet. it appears to be in 8x8 block row order 

    // Header
    // Comment
    // Quantization Tables (both)
    // Image Info
    // Huffman Tables
    // Encoded Blocks
    // EOI

    // clean up memory
    free(YData);
    free(CbData);
    free(CrData);

    free(YDctData);
    free(CbDctData);
    free(CrDctData);

    free(YQData);
    free(CbQData);
    free(CrQData);

    free(YRearrangedData);
    free(CbRearrangedData);
    free(CrRearrangedData);
}

void Compressor::sequential_dct(float* inputData, float* outputData, int width, int height) {
    // loop over all 8x8 tiles
    for (size_t j_block = 0; j_block < height/8; j_block++) {
        for (size_t i_block = 0; i_block < width/8; i_block++) {

            // loop within an 8x8 tile
            for (size_t j_tile = 0; j_tile < 8; j_tile++) {
                
                size_t j = j_block * 8 + j_tile; // overall j index in image

                for (size_t i_tile = 0; i_tile < 8; i_tile++) {

                    size_t i = i_block * 8 + i_tile; // overall i index in image

                    float c_i = (i_tile == 0) ? sqrt_2 : 1.0;
                    float c_j = (j_tile == 0) ? sqrt_2 : 1.0;

                    float sum = 0;

                    // loop within an 8x8 tile again to generate sum
                    for (size_t j_sum = 0; j_sum < 8; j_sum++) {
                        for (size_t i_sum = 0; i_sum < 8; i_sum++) {
                        
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

    printf("Done!\n");

    return 0;
}