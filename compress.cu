#include <wb.h>
#include <memory>
#include <fstream>

#define pi 3.14159265f
#define sqrt1_2 0.707106781f

#define ERROR(MSG) {fprintf(stderr, MSG "\n"); exit(1);}

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      exit(-1);                                                           \
    }                                                                     \
  } while (0)

__global__ void kernel_float_to_char(float* input, unsigned char* output, uint num) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num) {
        output[idx] = (unsigned char) (255 * input[idx]);
    }

    return;
}

// Naive kernel with no optimization for coalescing
__global__ void kernel_rgb_to_ycbcr(const float* input, float* output_y, float* output_cb, float* output_cr, uint num_rgb_pix){
    
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx >  num_rgb_pix) return;

        float r = input[idx*3]     * 255.f;
        float g = input[idx*3 + 1] * 255.f;
        float b = input[idx*3 + 2] * 255.f;

        output_y[idx]   =  0.f   + (0.299f   * r) + (0.587f   * g) + (0.114f   * b);
        output_cr[idx] = 128.f - (0.16874f * r) - (0.33126f * g) + (0.5f     * b);
        output_cb[idx] = 128.f + (0.5f     * r) - (0.41869f * g) - (0.08131f * b);
}

// Each thread loops over the pixels in its block to generate a single output pixel.
// Call this three times, once for each channel
__global__ void kernel_block_dct(const float* inputData, float* outputData, const float* dct, uint width, uint height){
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
        
            float cos1 = dct[i_sum*8 + i_tile];
            float cos2 = dct[j_sum*8 + j_tile];

            size_t x = i_block * 8 + i_sum;
            size_t y = j_block * 8 + j_sum;

            sum += (inputData[y * width + x] * cos1 * cos2);
        }
    }

    outputData[j * width + i] = 0.25 * c_i * c_j * sum;
}


//Call this once for each channel
__global__ void kernel_quantize_dct_output(const float* inputData, uint8_t* outputData, const uint8_t* Q, uint width, uint height)
{
    assert(blockDim.x == 8);
    assert(blockDim.y == 8);

    uint j_block = blockIdx.y, i_block = blockIdx.x;
    uint j_tile = threadIdx.y, i_tile = threadIdx.x;
    
    size_t i = i_block * 8 + i_tile;
    size_t j = j_block * 8 + j_tile;

    outputData[j * width + i] = (uint8_t) round(inputData[j * width + i] / Q[j_tile*8 + i_tile]);
}

__global__ void kernel_zigzag(const uint8_t* inputData, uint8_t* outputData, const uint8_t* zigzag_map, uint width, uint height) {
    assert(blockDim.x == 8);
    assert(blockDim.y == 8);

    uint j_block = blockIdx.y, i_block = blockIdx.x;
    uint j_tile = threadIdx.y, i_tile = threadIdx.x;
    size_t block_num = j_block * width / 8 + i_block;
    size_t tile_num = j_tile * 8 + i_tile;
    size_t x = i_block * 8 + i_tile;
    size_t y = j_block * 8 + j_tile;

    outputData[block_num * 64 + zigzag_map[tile_num]] = inputData[y * width + x];
}


// Input data is an image, output data is length num_jpeg_blocks and encodes the DC offset for each block
__global__ void kernel_subtract_dc_values(const uint8_t* inputData, int8_t* diffs, uint width, uint height) {
    uint tx = threadIdx.x, ty = threadIdx.y;

    
    //TODO - MAY NEED TO CHANGE FOR PARTIAL BLOCKS
    uint numBlocksX = width / 8;
    uint numBlocksY = height / 8;

    // I think i and j might be switched here from what they are in the rest of the code
    for(uint ii = tx; ii < numBlocksY; ii += blockDim.y){
        for (uint jj = ty; jj < numBlocksX; jj += blockDim.x) {

            if (ii == 0 && jj == 0) {
                diffs[0] = 0;
                continue;
            }

            //ii and jj are indices of current JPEG block
            
            //indices of previous JPEG block
            uint ii_prev = (jj == 0) ? (ii - 1) : ii;
            uint jj_prev = (jj == 0) ? (numBlocksY - 1) : jj - 1;

            uint8_t curr_dc = inputData[ii * 8 * width + jj * 8];
            uint8_t prev_dc = inputData[ii_prev * 8 * width + jj_prev * 8];

            // I believe the wraparound should work correctly here?
            diffs[ii * numBlocksX + jj] = curr_dc - prev_dc;
        }
    }
}

template<typename T>
class DevicePtr{
private:
    void* mData;
    void cleanup(){
        if(mData){
            cudaFree(mData);
            mData = nullptr;
        }
    }
public:
    DevicePtr(): mData(nullptr){

    }
    DevicePtr(size_t size): mData(nullptr) {
        reset(size);
    }
    ~DevicePtr(){
        cleanup();
    }

    void reset(size_t size = 0){
        cleanup();
        if(size != 0){
            wbCheck(cudaMalloc(&mData, size * sizeof(T)));
        }
    }

    T* get() const {
        return (T*)mData;
    }

    T& operator[](size_t i) const {
        return get()[i];
    }
};

// represent a single Huffman code i.e. a sequence of bits with length up to 16
struct BitCode
{
  BitCode() = default; // undefined state, must be initialized at a later time
  BitCode(uint16_t code_, uint8_t numBits_)
  : code(code_), numBits(numBits_) {}
  uint16_t code;       // JPEG's Huffman codes are limited to 16 bits
  uint8_t  numBits;    // number of valid bits
};

typedef void (*WRITE_ONE_BYTE)(unsigned char);

// wrapper for bit output operations
struct BitWriter
{
  // user-supplied callback that writes/stores one byte
  WRITE_ONE_BYTE output;
  // initialize writer
  explicit BitWriter(WRITE_ONE_BYTE output_) : output(output_) {}

  // store the most recently encoded bits that are not written yet
  struct BitBuffer
  {
    int32_t data    = 0; // actually only at most 24 bits are used
    uint8_t numBits = 0; // number of valid bits (the right-most bits)
  } buffer;

  // write Huffman bits stored in BitCode, keep excess bits in BitBuffer
  BitWriter& operator<<(const BitCode& data)
  {
    // append the new bits to those bits leftover from previous call(s)
    buffer.numBits += data.numBits;
    buffer.data   <<= data.numBits;
    buffer.data    |= data.code;

    // write all "full" bytes
    while (buffer.numBits >= 8)
    {
      // extract highest 8 bits
      buffer.numBits -= 8;
      auto oneByte = uint8_t(buffer.data >> buffer.numBits);
      output(oneByte);

      if (oneByte == 0xFF) // 0xFF has a special meaning for JPEGs (it's a block marker)
        output(0);         // therefore pad a zero to indicate "nope, this one ain't a marker, it's just a coincidence"

      // note: I don't clear those written bits, therefore buffer.bits may contain garbage in the high bits
      //       if you really want to "clean up" (e.g. for debugging purposes) then uncomment the following line
      //buffer.bits &= (1 << buffer.numBits) - 1;
    }
    return *this;
  }

  // write all non-yet-written bits, fill gaps with 1s (that's a strange JPEG thing)
  void flush()
  {
    // at most seven set bits needed to "fill" the last byte: 0x7F = binary 0111 1111
    *this << BitCode(0x7F, 7); // I should set buffer.numBits = 0 but since there are no single bits written after flush() I can safely ignore it
  }

  // NOTE: all the following BitWriter functions IGNORE the BitBuffer and write straight to output !
  // write a single byte
  BitWriter& operator<<(uint8_t oneByte)
  {
    output(oneByte);
    return *this;
  }

  // write an array of bytes
  template <typename T, int Size>
  BitWriter& operator<<(T (&manyBytes)[Size])
  {
    for (auto c : manyBytes)
      output(c);
    return *this;
  }

  // start a new JFIF block
  void addMarker(uint8_t id, uint16_t length)
  {
    output(0xFF); output(id);     // ID, always preceded by 0xFF
    output(uint8_t(length >> 8)); // length of the block (big-endian, includes the 2 length bytes as well)
    output(uint8_t(length & 0xFF));
  }
};

class Compressor {
private:
    const char* inputImageFile;
    wbImage_t inputImage;
    BitWriter bitWriter;
    uint16_t imageWidth;
    uint16_t imageHeight;
    uint8_t imageChannels;
    float* hostInputImageData;
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

    // Huffman tables (from https://github.com/nothings/stb/blob/master/stb_image_write.h)
    // TODO: why were these allocated with space for 256 elements but have much fewer than that??
    // TBH we may just want to copy the toojpeg implementation
    const BitCode YDC_HT[256] = { {0,2},{2,3},{3,3},{4,3},{5,3},{6,3},{14,4},{30,5},{62,6},{126,7},{254,8},{510,9} };
    const BitCode UVDC_HT[256] = { {0,2},{1,2},{2,2},{6,3},{14,4},{30,5},{62,6},{126,7},{254,8},{510,9},{1022,10},{2046,11} };
    const BitCode YAC_HT[256] = {
       {10,4},{0,2},{1,2},{4,3},{11,4},{26,5},{120,7},{248,8},{1014,10},{65410,16},{65411,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {12,4},{27,5},{121,7},{502,9},{2038,11},{65412,16},{65413,16},{65414,16},{65415,16},{65416,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {28,5},{249,8},{1015,10},{4084,12},{65417,16},{65418,16},{65419,16},{65420,16},{65421,16},{65422,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {58,6},{503,9},{4085,12},{65423,16},{65424,16},{65425,16},{65426,16},{65427,16},{65428,16},{65429,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {59,6},{1016,10},{65430,16},{65431,16},{65432,16},{65433,16},{65434,16},{65435,16},{65436,16},{65437,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {122,7},{2039,11},{65438,16},{65439,16},{65440,16},{65441,16},{65442,16},{65443,16},{65444,16},{65445,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {123,7},{4086,12},{65446,16},{65447,16},{65448,16},{65449,16},{65450,16},{65451,16},{65452,16},{65453,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {250,8},{4087,12},{65454,16},{65455,16},{65456,16},{65457,16},{65458,16},{65459,16},{65460,16},{65461,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {504,9},{32704,15},{65462,16},{65463,16},{65464,16},{65465,16},{65466,16},{65467,16},{65468,16},{65469,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {505,9},{65470,16},{65471,16},{65472,16},{65473,16},{65474,16},{65475,16},{65476,16},{65477,16},{65478,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {506,9},{65479,16},{65480,16},{65481,16},{65482,16},{65483,16},{65484,16},{65485,16},{65486,16},{65487,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {1017,10},{65488,16},{65489,16},{65490,16},{65491,16},{65492,16},{65493,16},{65494,16},{65495,16},{65496,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {1018,10},{65497,16},{65498,16},{65499,16},{65500,16},{65501,16},{65502,16},{65503,16},{65504,16},{65505,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {2040,11},{65506,16},{65507,16},{65508,16},{65509,16},{65510,16},{65511,16},{65512,16},{65513,16},{65514,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {65515,16},{65516,16},{65517,16},{65518,16},{65519,16},{65520,16},{65521,16},{65522,16},{65523,16},{65524,16},{0,0},{0,0},{0,0},{0,0},{0,0},
       {2041,11},{65525,16},{65526,16},{65527,16},{65528,16},{65529,16},{65530,16},{65531,16},{65532,16},{65533,16},{65534,16},{0,0},{0,0},{0,0},{0,0},{0,0}
    };
    const BitCode UVAC_HT[256] = {
       {0,2},{1,2},{4,3},{10,4},{24,5},{25,5},{56,6},{120,7},{500,9},{1014,10},{4084,12},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {11,4},{57,6},{246,8},{501,9},{2038,11},{4085,12},{65416,16},{65417,16},{65418,16},{65419,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {26,5},{247,8},{1015,10},{4086,12},{32706,15},{65420,16},{65421,16},{65422,16},{65423,16},{65424,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {27,5},{248,8},{1016,10},{4087,12},{65425,16},{65426,16},{65427,16},{65428,16},{65429,16},{65430,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {58,6},{502,9},{65431,16},{65432,16},{65433,16},{65434,16},{65435,16},{65436,16},{65437,16},{65438,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {59,6},{1017,10},{65439,16},{65440,16},{65441,16},{65442,16},{65443,16},{65444,16},{65445,16},{65446,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {121,7},{2039,11},{65447,16},{65448,16},{65449,16},{65450,16},{65451,16},{65452,16},{65453,16},{65454,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {122,7},{2040,11},{65455,16},{65456,16},{65457,16},{65458,16},{65459,16},{65460,16},{65461,16},{65462,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {249,8},{65463,16},{65464,16},{65465,16},{65466,16},{65467,16},{65468,16},{65469,16},{65470,16},{65471,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {503,9},{65472,16},{65473,16},{65474,16},{65475,16},{65476,16},{65477,16},{65478,16},{65479,16},{65480,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {504,9},{65481,16},{65482,16},{65483,16},{65484,16},{65485,16},{65486,16},{65487,16},{65488,16},{65489,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {505,9},{65490,16},{65491,16},{65492,16},{65493,16},{65494,16},{65495,16},{65496,16},{65497,16},{65498,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {506,9},{65499,16},{65500,16},{65501,16},{65502,16},{65503,16},{65504,16},{65505,16},{65506,16},{65507,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {2041,11},{65508,16},{65509,16},{65510,16},{65511,16},{65512,16},{65513,16},{65514,16},{65515,16},{65516,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
       {16352,14},{65517,16},{65518,16},{65519,16},{65520,16},{65521,16},{65522,16},{65523,16},{65524,16},{65525,16},{0,0},{0,0},{0,0},{0,0},{0,0},
       {1018,10},{32707,15},{65526,16},{65527,16},{65528,16},{65529,16},{65530,16},{65531,16},{65532,16},{65533,16},{65534,16},{0,0},{0,0},{0,0},{0,0},{0,0}
    };
    const int YQT[64] = { 16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,
                              37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99 };
    const int UVQT[64] = { 17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99,
                               99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99 };
    const float aasf[8] = { 1.0f * 2.828427125f, 1.387039845f * 2.828427125f, 1.306562965f * 2.828427125f, 1.175875602f * 2.828427125f,
                                  1.0f * 2.828427125f, 0.785694958f * 2.828427125f, 0.541196100f * 2.828427125f, 0.275899379f * 2.828427125f };

    DevicePtr<float> dct_device;
    DevicePtr<uint8_t> Q_l_device;
    DevicePtr<uint8_t> Q_c_device;
    DevicePtr<uint8_t> zigzag_map_device;

    void float_to_char(unsigned char* hostCharData);
    void rgb_to_ycbcr(unsigned char* hostCharData, unsigned char* hostYCbCrData);
    void sequential_dct(float* inputData, float* outputData, int width, int height);
    void write_file();

public:
    Compressor(wbArg_t args, WRITE_ONE_BYTE _output);
    ~Compressor();
    void sequential_compress();
    void parallel_compress();
};

Compressor::Compressor(wbArg_t args, WRITE_ONE_BYTE _output)
    : bitWriter(_output)
    , dct_device(64)
    , Q_l_device(64)
    , Q_c_device(64)
    , zigzag_map_device(64)
{
    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImageFile = wbArg_getInputFile(args, 0);
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    hostInputImageData = wbImage_getData(inputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbCheck(cudaMemcpy(dct_device.get(), dct, sizeof(dct), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(Q_l_device.get(), Q_l, sizeof(Q_l), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(Q_c_device.get(), Q_c, sizeof(Q_c), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(zigzag_map_device.get(), zigzag_map, sizeof(zigzag_map), cudaMemcpyHostToDevice));
    
    printf("Input image size: %dx%dx%d\n", imageWidth, imageHeight, imageChannels);
}

Compressor::~Compressor() {
    wbImage_delete(inputImage);
}

void Compressor::write_file() {   

    const uint8_t HeaderJfif[2+2+16] = { 
        0xFF,0xD8,          // SOI marker (start of image)
        0xFF,0xE0,          // JFIF APP0 tag
        0,16,               // length: 16 bytes (14 bytes payload + 2 bytes for this length field)
        'J','F','I','F',0,  // JFIF identifier, zero-terminated
        1,1,                // JFIF version 1.1
        0,                  // no density units specified
        0,1,0,1,            // density: 1 pixel "per pixel" horizontally and vertically
        0,0                 // no thumbnail (size 0 x 0) 
    };   
    const char comment[23] = "EECS598 Project Output";

    // JFIF Header
    bitWriter << HeaderJfif;

    // Comment
    bitWriter.addMarker(0xFE, 24);
    for (auto i = 0; i < 22; i++) { 
        bitWriter << comment[i];
    }

    // Quantization Tables
    bitWriter.addMarker(0xDB, 2 + 2*(1 + 8*8));
    bitWriter   << 0x00;
    for (auto i = 0; i < 8; i++) {
        bitWriter << Q_l[i];
    }
    bitWriter   << 0x01;
    for (auto i = 0; i < 8; i++) {
        bitWriter << Q_c[i];
    }

    // Bits/Pixel, Image Size, Number of Channels, and Subsampling and Y vs C for each channel
    bitWriter.addMarker(0xC0, 2+6+3*3);
    bitWriter   << 0x08 
                << (imageHeight >> 8) << (imageHeight & 0xFF)
                << (imageWidth  >> 8) << (imageWidth  & 0xFF)
                << 0x03
                << 0x01 << 0x11 << 0x00
                << 0x02 << 0x11 << 0x01
                << 0x03 << 0x11 << 0x01;

    printf("0x%X 0x%X\n", imageHeight, imageWidth);

    // Huffman Tables
    bitWriter.addMarker(0xC4, 2+208+208);
    bitWriter   << 0x00 << DcLuminanceCodesPerBitsize   << DcLuminanceValues
                << 0x10 << AcLuminanceCodesPerBitsize   << AcLuminanceValues
                << 0x01 << DcChrominanceCodesPerBitsize << DcChrominanceValues
                << 0x11 << AcChrominanceCodesPerBitsize << AcChrominanceValues;

    // Start of Scan
    bitWriter.addMarker(0xDA, 2+1+2*3+3);

    // Number of Channels and Channel map to Huffman Tables
    bitWriter   << 0x03
                << 0x01 << 0x00
                << 0x02 << 0x11
                << 0x03 << 0x11;

    // Spectral Selection - Single Scan
    bitWriter << 0x00 << 0x3F << 0x00;

    // TODO: Image Data

    // End of Image
    bitWriter << 0xFF << 0xD9;

    bitWriter.flush();
}

void Compressor::sequential_compress() {

    size_t numPix = imageWidth*imageHeight;

    // convert from RGB floats to YCbCr in separate channels
    std::unique_ptr<float[]> 
        YData(new float[numPix]), 
        CbData(new float[numPix]),
        CrData(new float[numPix]);

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
    std::unique_ptr<float[]>
        YDctData(new float[numPix]),
        CbDctData(new float[numPix]),
        CrDctData(new float[numPix]);

    sequential_dct(YData.get(),  YDctData.get(),  imageWidth, imageHeight);
    sequential_dct(CbData.get(), CbDctData.get(), imageWidth, imageHeight);
    sequential_dct(CrData.get(), CrDctData.get(), imageWidth, imageHeight);

    // quantization
    std::unique_ptr<uint8_t[]> 
        YQData(new uint8_t[numPix]), 
        CbQData(new uint8_t[numPix]), 
        CrQData(new uint8_t[numPix]);

    for (size_t j_block = 0; j_block < imageHeight/8; j_block++) {
        for (size_t i_block = 0; i_block < imageWidth/8; i_block++) {
            for (size_t j_tile = 0; j_tile < 8; j_tile++) {
                for (size_t i_tile = 0; i_tile < 8; i_tile++) {

                    size_t i = i_block * 8 + i_tile;
                    size_t j = j_block * 8 + j_tile;

                    YQData [j * imageWidth + i] = (uint8_t) round(YDctData [j * imageWidth + i] / Q_l[j_tile][i_tile]);
                    CbQData[j * imageWidth + i] = (uint8_t) round(CbDctData[j * imageWidth + i] / Q_c[j_tile][i_tile]);
                    CrQData[j * imageWidth + i] = (uint8_t) round(CrDctData[j * imageWidth + i] / Q_c[j_tile][i_tile]);   
                }
            }
        }
    }

    // zigzag rearrange
    std::unique_ptr<uint8_t[]> 
        YRearrangedData(new uint8_t[numPix]),
        CbRearrangedData(new uint8_t[numPix]),
        CrRearrangedData(new uint8_t[numPix]);

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

    // TODO: write file
    write_file();
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

    size_t numPix = imageWidth * imageHeight;
    size_t numEl = numPix * imageChannels;
    size_t numBlocks = numPix / 64;

    DevicePtr<float> deviceRGBImageData(numEl);
    DevicePtr<float> deviceYCbCrImageData(numEl);
    DevicePtr<float> deviceDCTData(numEl);
    DevicePtr<uint8_t> deviceQuantData(numEl);
    DevicePtr<uint8_t> deviceZigzagData(numEl);
    DevicePtr<int8_t> deviceDcCoeffDiffs(numBlocks * 3);

    dim3 DimGrid1((imageWidth*imageHeight*imageChannels-1)/1024 + 1, 1, 1);
    dim3 DimBlock1(1024, 1, 1);

    float* deviceY = deviceYCbCrImageData.get();
    float* deviceCb = deviceY + numPix;
    float* deviceCr = deviceCb + numPix;

    kernel_rgb_to_ycbcr<<<DimGrid1, DimBlock1>>>(deviceRGBImageData.get(), deviceY, deviceCb, deviceCr, numPix);

    wbCheck(cudaDeviceSynchronize());
    
    float* deviceYDCT = deviceDCTData.get();
    float* deviceCbDCT = deviceYDCT + numPix;
    float* deviceCrDCT = deviceCbDCT + numPix;

    dim3 DimGrid2((imageHeight-1)/8+1, (imageWidth-1)/8 + 1, 1);
    dim3 DimBlock2(8,8,1);

    kernel_block_dct<<<DimGrid2, DimBlock2>>>(deviceY,  deviceYDCT,  dct_device.get(), imageWidth, imageHeight);
    kernel_block_dct<<<DimGrid2, DimBlock2>>>(deviceCb, deviceCbDCT, dct_device.get(), imageWidth, imageHeight);
    kernel_block_dct<<<DimGrid2, DimBlock2>>>(deviceCr, deviceCrDCT, dct_device.get(), imageWidth, imageHeight);

    wbCheck(cudaDeviceSynchronize());

    uint8_t* deviceYQuant = deviceQuantData.get();
    uint8_t* deviceCbQuant = deviceYQuant + numPix;
    uint8_t* deviceCrQuant = deviceCbQuant + numPix;

    kernel_quantize_dct_output<<<DimGrid2, DimBlock2>>>(deviceYDCT,  deviceYQuant,  Q_l_device.get(), imageWidth, imageHeight);
    kernel_quantize_dct_output<<<DimGrid2, DimBlock2>>>(deviceCbDCT, deviceCbQuant, Q_c_device.get(), imageWidth, imageHeight);
    kernel_quantize_dct_output<<<DimGrid2, DimBlock2>>>(deviceCrDCT, deviceCrQuant, Q_c_device.get(), imageWidth, imageHeight);

    wbCheck(cudaDeviceSynchronize());

    uint8_t* deviceYZigzag = deviceZigzagData.get();
    uint8_t* deviceCbZigzag = deviceYZigzag + numPix;
    uint8_t* deviceCrZigzag = deviceCbZigzag + numPix;

    kernel_zigzag << <DimGrid2, DimBlock2 >> > (deviceYQuant,  deviceYZigzag,  zigzag_map_device.get(), imageWidth, imageHeight);
    kernel_zigzag << <DimGrid2, DimBlock2 >> > (deviceCbQuant, deviceCbZigzag, zigzag_map_device.get(), imageWidth, imageHeight);
    kernel_zigzag << <DimGrid2, DimBlock2 >> > (deviceCrQuant, deviceCrZigzag, zigzag_map_device.get(), imageWidth, imageHeight);

    //Note that this and the next kernel can be done independently of each other

    dim3 DimGrid3((imageHeight - 1) / (8*16) + 1, (imageWidth - 1) / (8*16) + 1, 1);
    dim3 DimBlock3(16, 16, 1);

    int8_t* dcY = deviceDcCoeffDiffs.get();
    int8_t* dcCb = dcY + numBlocks;
    int8_t* dcCr = dcCb + numBlocks;

    kernel_subtract_dc_values << <DimGrid3, DimBlock3 >> > (deviceYQuant, dcY, imageWidth, imageHeight);
    kernel_subtract_dc_values << <DimGrid3, DimBlock3 >> > (deviceCbQuant, dcCb, imageWidth, imageHeight);
    kernel_subtract_dc_values << <DimGrid3, DimBlock3 >> > (deviceCrQuant, dcCr, imageWidth, imageHeight);

    wbCheck(cudaDeviceSynchronize());

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

std::ofstream outputFile;
void write_one_byte(unsigned char byte) { outputFile << byte; };

int main(int argc, char **argv) {

    wbArg_t args = wbArg_read(argc, argv);
    
    outputFile.open(wbArg_getOutputFile(args), std::ios_base::binary);

    if (!outputFile.is_open()) ERROR("Opening output file failed");

    Compressor compressor(args, write_one_byte);

    compressor.sequential_compress();
    // compressor.parallel_compress();

    printf("Done!\n");

    return 0;
}
