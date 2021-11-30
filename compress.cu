#include <wb.h>

#define pi 3.14159265f
#define sqrt1_2 0.707106781f

__global__ void kernel_float_to_char(float* input, unsigned char* output, uint num) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num) {
        output[idx] = (unsigned char) (255 * input[idx]);
    }

    return;
}

// Naive kernel with no optimization for coalescing
__global__ void kernel_rgb_to_ycbcr(const float* input, float* output_y, float* output_cr, float* output_cb, uint num_rgb_pix){
    
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx >  num_rgb_pix) return;

        float r = input[idx*3];
        float g = input[idx*3 + 1];
        float b = input[idx*3 + 2];

        output_y[idx]   =  0.f   + (0.299f   * r) + (0.587f   * g) + (0.114f   * b);
        output_cr[idx] = 128.f - (0.16874f * r) - (0.33126f * g) + (0.5f     * b);
        output_cb[idx] = 128.f + (0.5f     * r) - (0.41869f * g) - (0.08131f * b);
}

// Each thread loops over the pixels in its block to generate a single output pixel.
// Call this three times, once for each channel
__global__ void kernel_block_dct(const float* inputData, float* outputData, const float dct[8][8], uint width, uint height){
    assert(blockDim.x == 8);
    assert(blockDim.y == 8);

    uint j_block = blockIdx.y, i_block = blockIdx.x;
    uint j_tile = threadIdx.y, i_tile = threadIdx.x;

    size_t i = i_block * 8 + i_tile; // overall i index in image
    size_t j = j_block * 8 + j_tile; // overall j index in image

    float c_i = (i_tile == 0) ? sqrt1_2 : 1.0f;
    float c_j = (j_tile == 0) ? sqrt1_2 : 1.0f;

    float sum = 0;

    // loop within an 8x8 tile to generate sum
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

//Call this once for each channel
__global__ void kernel_quantize_dct_output(const float* inputData, uint8_t* outputData, const uint8_t Q[8][8], uint width, uint height)
{
    assert(blockDim.x == 8);
    assert(blockDim.y == 8);

    uint j_block = blockIdx.y, i_block = blockIdx.x;
    uint j_tile = threadIdx.y, i_tile = threadIdx.x;
    
    size_t i = i_block * 8 + i_tile;
    size_t j = j_block * 8 + j_tile;

    outputData[j * imageWidth + i] = (uint8_t) round(inputData[j * imageWidth + i] / Q[j_tile][i_tile]);
}

class Compressor {
private:
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    const char *inputImageFile;
    float *hostInputImageData;
    // Discrete cosine transform lookup: cos((2i+1)jpi/16) = dct[i][j]
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
    // Standard Quantization table for Luminance
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
    // Standard Quantization table for Chrominance
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
    // Mapping of blocks to zigzag - defines the order of selection
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
    // Huffman definitions for first DC/AC tables (luminance / Y channel)
    const uint8_t DcLuminanceCodesPerBitsize[16]   = { 0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0 };   // sum = 12
    const uint8_t DcLuminanceValues         [12]   = { 0,1,2,3,4,5,6,7,8,9,10,11 };         // => 12 codes
    const uint8_t AcLuminanceCodesPerBitsize[16]   = { 0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125 }; // sum = 162
    const uint8_t AcLuminanceValues        [162]   =                                        // => 162 codes
    { 0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08, // 16*10+2 symbols because
      0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28, // upper 4 bits can be 0..F
      0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59, // while lower 4 bits can be 1..A
      0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89, // plus two special codes 0x00 and 0xF0
      0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6, // order of these symbols was determined empirically by JPEG committee
      0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
      0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA 
    };
    // Huffman definitions for second DC/AC tables (chrominance / Cb and Cr channels)
    const uint8_t DcChrominanceCodesPerBitsize[16] = { 0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0 };   // sum = 12
    const uint8_t DcChrominanceValues         [12] = { 0,1,2,3,4,5,6,7,8,9,10,11 };         // => 12 codes (identical to DcLuminanceValues)
    const uint8_t AcChrominanceCodesPerBitsize[16] = { 0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119 }; // sum = 162
    const uint8_t AcChrominanceValues        [162] =                                        // => 162 codes
    { 0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91, // same number of symbol, just different order
      0xA1,0xB1,0xC1,0x09,0x23,0x33,0x52,0xF0,0x15,0x62,0x72,0xD1,0x0A,0x16,0x24,0x34,0xE1,0x25,0xF1,0x17,0x18,0x19,0x1A,0x26, // (which is more efficient for AC coding)
      0x27,0x28,0x29,0x2A,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,
      0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x82,0x83,0x84,0x85,0x86,0x87,
      0x88,0x89,0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,
      0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,
      0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA 
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

    // TODO: build file:
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

                    float c_i = (i_tile == 0) ? sqrt1_2 : 1.0;
                    float c_j = (j_tile == 0) ? sqrt1_2 : 1.0;

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