#include <wb.h>
#include <memory>
#include <fstream>
#include <functional>
#include <queue>
#include <limits>
#include <algorithm>
#include <chrono>
#include <nvjpeg.h>

#define pi 3.14159265f
#define sqrt1_2 0.707106781f

/*#define USE_COMBINED_KERNEL
#define PAGE_LOCK_HOST_BUFFERS
#define USE_STREAMS
#define SINGLE_GPU_BUFFER
#define USE_CONSTANT_MEMORY
#define INPUT_TO_CHAR
*/
//For comparison with state of the art
#define USE_NVJPEG

//using T_Quant = int;
using T_Quant = int16_t;

#ifdef INPUT_TO_CHAR
using T_Input = uint8_t;
#else
using T_Input = float;
#endif


#ifdef USE_STREAMS
constexpr unsigned int NUM_STREAMS = 6;
constexpr unsigned int LINES_PER_SLICE = 256;
#else
constexpr unsigned int NUM_STREAMS = 1;
constexpr unsigned int LINES_PER_SLICE = std::numeric_limits<unsigned int>::max();
#endif

using WRITE_ONE_BYTE = std::function<void(unsigned char)>;

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

//http://web.engr.oregonstate.edu/~mjb/cs575/Projects/helper_cuda.h
static const char* getNvJpegErrStr(nvjpegStatus_t error) {
    switch (error) {
    case NVJPEG_STATUS_SUCCESS:
        return "NVJPEG_STATUS_SUCCESS";

    case NVJPEG_STATUS_NOT_INITIALIZED:
        return "NVJPEG_STATUS_NOT_INITIALIZED";

    case NVJPEG_STATUS_INVALID_PARAMETER:
        return "NVJPEG_STATUS_INVALID_PARAMETER";

    case NVJPEG_STATUS_BAD_JPEG:
        return "NVJPEG_STATUS_BAD_JPEG";

    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

    case NVJPEG_STATUS_EXECUTION_FAILED:
        return "NVJPEG_STATUS_EXECUTION_FAILED";

    case NVJPEG_STATUS_ARCH_MISMATCH:
        return "NVJPEG_STATUS_ARCH_MISMATCH";

    case NVJPEG_STATUS_INTERNAL_ERROR:
        return "NVJPEG_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#define checkNvJpeg(stmt)                                                 \
  do {                                                                    \
    nvjpegStatus_t err = stmt;                                            \
    if (err != NVJPEG_STATUS_SUCCESS) {                                             \
      wbLog(ERROR, "NVJPEG error: ", getNvJpegErrStr(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      exit(-1);                                                           \
    }                                                                     \
  } while (0)


// Naive kernel with no optimization for coalescing
__global__ void kernel_rgb_to_ycbcr(const T_Input* input, float* output_y, float* output_cb, float* output_cr, uint num_rgb_pix){
    
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx >=  num_rgb_pix) return;

	#ifdef INPUT_TO_CHAR
        float scale = 1.f;
	#else
        float scale = 255.f;
	#endif

        float r = input[idx*3]     * scale;
        float g = input[idx*3 + 1] * scale;
        float b = input[idx*3 + 2] * scale;

        output_y[idx]  =  0.f  + (0.299f   * r) + (0.587f   * g) + (0.114f   * b) - 128.f;
        output_cb[idx] = 128.f - (0.16874f * r) - (0.33126f * g) + (0.5f     * b) - 128.f;
        output_cr[idx] = 128.f + (0.5f     * r) - (0.41869f * g) - (0.08131f * b) - 128.f;
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
    
    if (i >= width || j >= height) return;
    
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
__global__ void kernel_quantize_dct_output(const float* inputData, T_Quant* outputData, const uint8_t* Q, uint width, uint height)
{
    assert(blockDim.x == 8);
    assert(blockDim.y == 8);

    uint j_block = blockIdx.y, i_block = blockIdx.x;
    uint j_tile = threadIdx.y, i_tile = threadIdx.x;
    
    size_t i = i_block * 8 + i_tile;
    size_t j = j_block * 8 + j_tile;
    if (i >= width || j >= height) return;

    outputData[j * width + i] = (T_Quant) round(inputData[j * width + i] / Q[j_tile*8 + i_tile]);
}

__global__ void kernel_zigzag(const T_Quant* inputData, T_Quant* outputData, const uint8_t* zigzag_map, uint width, uint height) {
    assert(blockDim.x == 8);
    assert(blockDim.y == 8);

    uint j_block = blockIdx.y, i_block = blockIdx.x;
    uint j_tile = threadIdx.y, i_tile = threadIdx.x;
    size_t block_num = j_block * width / 8 + i_block;
    size_t tile_num = j_tile * 8 + i_tile;
    size_t x = i_block * 8 + i_tile;
    size_t y = j_block * 8 + j_tile;

    if (x >= width || y >= height) return;

    outputData[block_num * 64 + zigzag_map[tile_num]] = inputData[y * width + x];
}

template<typename T, int N>
struct Array {
    T data[N];
    __host__ __device__ T& operator[](size_t i) {
        return data[i];
    }
};

#ifdef USE_CONSTANT_MEMORY
__constant__ float dct_constant[64];
__constant__ uint8_t Q_l_constant[64];
__constant__ uint8_t Q_c_constant[64];
__constant__ uint8_t zigzag_map_constant[64];
#endif

__global__ void kernel_combined(const T_Input* inputData, T_Quant* outputData, 
    #ifndef USE_CONSTANT_MEMORY
    const float* dct, Array<const uint8_t*,3> Q, const uint8_t* zigzag_map, 
    #endif
    uint width, uint height) {
   
    #ifdef USE_CONSTANT_MEMORY
    const float* dct = dct_constant;
    const uint8_t* Q[3] = { Q_l_constant, Q_c_constant, Q_c_constant };
    const uint8_t* zigzag_map = zigzag_map_constant;
    #endif

    assert(blockDim.x == 8);
    assert(blockDim.y == 8);
    assert(blockDim.z == 3);

    __shared__ float blockInputData[3][8][8];

    uint i_block = blockIdx.x, j_block = blockIdx.y;
    uint i_tile = threadIdx.x, j_tile = threadIdx.y;
    uint chan_idx = threadIdx.z;
    
    size_t i = i_block * 8 + i_tile; // overall i index in image
    size_t j = j_block * 8 + j_tile; // overall j index in image
    
    if (i >= width || j >= height) return;

    // Load data and convert RGB to YCbCr
    // The input accesses can still be coalesced here
    blockInputData[chan_idx][j_tile][i_tile] = inputData[j * width * 3 + i * 3 + chan_idx] 
    #ifndef INPUT_TO_CHAR
    * 255.f
    #endif
    ;

    __syncthreads();

    float rgb[3];
    for (int ii = 0; ii < 3; ++ ii) {
        rgb[ii] = blockInputData[ii][j_tile][i_tile];
    }

    __syncthreads();

    float conversion_matrix[3][4] = { 
        {0.f,    0.299f,    0.587f,    0.114f  },
        {128.f, -0.16784f, -0.33126f,  0.5f    },
        {128.f,  0.5f,     -0.41869f, -0.08131f} 
    };
    float* conv = &conversion_matrix[chan_idx][0];

    blockInputData[chan_idx][j_tile][i_tile] =
        conv[0] + conv[1] * rgb[0] + conv[2] * rgb[1] + conv[3] * rgb[2] - 128.f;

    __syncthreads();
    
    size_t block_num = j_block * width / 8 + i_block;
    size_t tile_num = j_tile * 8 + i_tile;

    // DCT
    
    float c_i = (i_tile == 0) ? sqrt1_2 : 1.0f;
    float c_j = (j_tile == 0) ? sqrt1_2 : 1.0f;

    float sum = 0;

    // loop within an 8x8 tile to generate sum
    for (size_t j_sum = 0; j_sum < 8; j_sum++) {
        for (size_t i_sum = 0; i_sum < 8; i_sum++) {
        
            float cos1 = dct[i_sum*8 + i_tile];
            float cos2 = dct[j_sum*8 + j_tile];

            sum += (blockInputData[chan_idx][j_sum][i_sum] * cos1 * cos2);
        }
    }

    float dct_temp = 0.25 * c_i * c_j * sum;
    outputData[chan_idx * width * height + block_num * 64 + zigzag_map[tile_num]] = (T_Quant) roundf(dct_temp / Q[chan_idx][j_tile*8 + i_tile]);
}

template<typename T>
class DevicePtr{
private:
    void* mData;
    size_t mSize;
    void cleanup(){
        if(mData){
            cudaFree(mData);
            mData = nullptr;
        }
        mSize = 0;
    }
public:
    DevicePtr(): mData(nullptr), mSize(0) {

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
        mSize = size;
    }

    size_t size() const {
        return mSize;
    }

    T* get() const {
        return (T*)mData;
    }

    T& operator[](size_t i) const {
        return get()[i];
    }
};

class Stream {
    cudaStream_t mStream;
public:
    Stream() {
        wbCheck(cudaStreamCreate(&mStream));
    }
    ~Stream() {
        cudaStreamDestroy(mStream);
    }

    void synchronize() {
        wbCheck(cudaStreamSynchronize(mStream));
    }

    operator cudaStream_t() const {
        return mStream;
    }
};


class ScopedTimer {
private:
    const std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::string mMsg;
    int mDepth;
public:
    ScopedTimer(const std::string& msg, int depth = 0) :
        mStart(std::chrono::high_resolution_clock::now()),
        mMsg(msg),
        mDepth(depth)
    {
    }

    ~ScopedTimer() {
        std::chrono::duration<double,std::milli> elapsedTime = std::chrono::high_resolution_clock::now() - mStart;
        for (int i = 0; i < mDepth; ++i) {
            std::cout << "\t";
        }
        std::cout << mMsg << ": " << elapsedTime.count() << " ms" << std::endl;
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
// Discrete cosine transform lookup: cos((2i+1)jpi/16) = dct[i][j]
static const float dct[8][8] = {
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
static const uint8_t Q_l[8][8] = {
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
static const uint8_t Q_c[8][8] = {
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
static const uint8_t zigzag_map[64] = {
    0,  1,  5,  6,  14, 15, 27, 28,
    2,  4,  7,  13, 16, 26, 29, 42,
    3,  8,  12, 17, 25, 30, 41, 43,
    9,  11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63  
};
// Mapping of order of quantization tables to be streamed into file
static const uint8_t ZigZagInv[64] = {
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63 
};
// Huffman definitions for first DC/AC tables (luminance / Y channel)
static const uint8_t DcLuminanceCodesPerBitsize[16]   = { 0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0 };   // sum = 12
static const uint8_t DcLuminanceValues         [12]   = { 0,1,2,3,4,5,6,7,8,9,10,11 };         // => 12 codes
static const uint8_t AcLuminanceCodesPerBitsize[16]   = { 0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125 }; // sum = 162
static  const uint8_t AcLuminanceValues        [162]   =                                        // => 162 codes
{ 0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08, // 16*10+2 symbols because
    0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28, // upper 4 bits can be 0..F
    0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59, // while lower 4 bits can be 1..A
    0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89, // plus two special codes 0x00 and 0xF0
    0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6, // order of these symbols was determined empirically by JPEG committee
    0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
    0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA 
};
// Huffman definitions for second DC/AC tables (chrominance / Cb and Cr channels)
static const uint8_t DcChrominanceCodesPerBitsize[16] = { 0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0 };   // sum = 12
static const uint8_t DcChrominanceValues         [12] = { 0,1,2,3,4,5,6,7,8,9,10,11 };         // => 12 codes (identical to DcLuminanceValues)
static const uint8_t AcChrominanceCodesPerBitsize[16] = { 0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119 }; // sum = 162
static const uint8_t AcChrominanceValues        [162] =                                        // => 162 codes
{ 0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91, // same number of symbol, just different order
    0xA1,0xB1,0xC1,0x09,0x23,0x33,0x52,0xF0,0x15,0x62,0x72,0xD1,0x0A,0x16,0x24,0x34,0xE1,0x25,0xF1,0x17,0x18,0x19,0x1A,0x26, // (which is more efficient for AC coding)
    0x27,0x28,0x29,0x2A,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,
    0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,
    0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,
    0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA 
};

struct CudaHostFreer {
    void operator()(void* ptr) {
        cudaFreeHost(ptr);
    }
};

class Compressor {
private:
    bool combined;
    const char* inputImageFile;
    wbImage_t inputImage;
    BitWriter bitWriter;
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t imageChannels;
    float* origInputData;
    T_Input* hostInputImageData;
    size_t bytesPerLine;
    
    // Huffman tables for lookup during encoding
    BitCode  codewordsArray[4096];     // note: quantized[i] is found at codewordsArray[quantized[i] + CodeWordLimit]
    BitCode* codewords;
    BitCode huffmanLuminanceDC[256];
    BitCode huffmanLuminanceAC[256];
    BitCode huffmanChrominanceDC[256];
    BitCode huffmanChrominanceAC[256];

    float* dct_device;
    uint8_t* Q_l_device;
    uint8_t* Q_c_device;
    uint8_t* zigzag_map_device;
    char* gpuScratch;
    DevicePtr<char> gpuMem;

    std::vector<T_Quant*> rearrangedData;    
    std::unique_ptr<T_Quant[]> rearrangedBuf;
    
    void init_codewords();
    void sequential_dct(const float* inputData, float* outputData, int width, int height);
    void write_file();
    void generateHuffmanTable(const uint8_t numCodes[16], const uint8_t* values, BitCode result[256]);

public:
    Compressor(wbArg_t args, bool _combined, WRITE_ONE_BYTE _output);
    ~Compressor();
    void sequential_compress_slice(const float* inputData, T_Quant* outputData[3], size_t numLines);
    void parallel_compress_slice(void* gpuScratch, size_t startLine, size_t numLines, cudaStream_t stream);
    void compress(bool parallel);
    void compress_nvjpeg(std::vector<unsigned char>& output);
    size_t getNumPixels() const {
        return (size_t)imageWidth * (size_t)imageHeight;
    }
};

Compressor::Compressor(wbArg_t args, bool _combined, WRITE_ONE_BYTE _output)
    : combined(_combined)
    , bitWriter(_output)
{
    ScopedTimer timer("Initialization", 1);

    inputImageFile = wbArg_getInputFile(args, 0);
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    size_t numPixels = imageWidth * imageHeight * imageChannels;
    origInputData = wbImage_getData(inputImage);
    #if defined(INPUT_TO_CHAR) || defined(USE_NVJPEG)
    hostInputImageData = new T_Input[numPixels];
    std::transform(origInputData, origInputData + numPixels, hostInputImageData, [](float f){return T_Input(f*255.f);});
    #else
    hostInputImageData = wbImage_getData(inputImage);
    #endif
    wbTime_stop(Generic, "Importing data and creating memory on host");

    if (imageChannels != 3) ERROR("Image must have three channels");

    //Use cudaMemGetInfo to initialize CUDA memory system so it doesn't mess up other timing
    size_t dummy;
    wbCheck(cudaMemGetInfo(&dummy, &dummy));
    
    rearrangedBuf.reset(new T_Quant[imageWidth*imageHeight*imageChannels]);
    #ifdef PAGE_LOCK_HOST_BUFFERS
    wbCheck(cudaHostRegister(hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(*hostInputImageData), cudaHostRegisterDefault));
    wbCheck(cudaHostRegister(rearrangedBuf.get(), imageWidth * imageHeight * imageChannels * sizeof(rearrangedBuf[0]), cudaHostRegisterDefault));
    #endif
    for (uint32_t ii = 0; ii < imageChannels; ++ii) {
        rearrangedData.push_back(rearrangedBuf.get() + ii * imageWidth * imageHeight);
    }

    generateHuffmanTable(DcLuminanceCodesPerBitsize,   DcLuminanceValues,   huffmanLuminanceDC);
    generateHuffmanTable(AcLuminanceCodesPerBitsize,   AcLuminanceValues,   huffmanLuminanceAC);
    generateHuffmanTable(DcChrominanceCodesPerBitsize, DcChrominanceValues, huffmanChrominanceDC);
    generateHuffmanTable(AcChrominanceCodesPerBitsize, AcChrominanceValues, huffmanChrominanceAC);

    init_codewords();

    bytesPerLine = imageWidth * imageChannels * (sizeof(T_Input) + sizeof(T_Quant));
    unsigned int linesPerSlice = std::min(LINES_PER_SLICE,imageHeight);
    size_t scratchSize = (NUM_STREAMS * linesPerSlice) * bytesPerLine;
    #ifdef SINGLE_GPU_BUFFER
    gpuMem.reset(scratchSize
        #ifndef USE_CONSTANT_MEMORY
        + sizeof(dct) + sizeof(Q_l) + sizeof(Q_c) + sizeof(zigzag_map)
        #endif
    );
    gpuScratch = gpuMem.get();
    #ifndef USE_CONSTANT_MEMORY
    dct_device = (float*)(gpuScratch + scratchSize);
    Q_l_device = ((uint8_t*)dct_device) + sizeof(dct);
    Q_c_device = Q_l_device + sizeof(Q_l);
    zigzag_map_device = Q_c_device + sizeof(Q_c);
    #endif // USE_CONSTANT_MEMORY
    #else // SINGLE_GPU_BUFFER
    wbCheck(cudaMalloc(&gpuScratch, scratchSize));
    #ifndef USE_CONSTANT_MEMORY
    wbCheck(cudaMalloc(&dct_device, sizeof(dct)));
    wbCheck(cudaMalloc(&Q_l_device, sizeof(Q_l)));
    wbCheck(cudaMalloc(&Q_c_device, sizeof(Q_c)));
    wbCheck(cudaMalloc(&zigzag_map_device, sizeof(zigzag_map)));
    #endif // USE_CONSTANT_MEMORY
    #endif // SINGLE_GPU_BUFFER

    #ifdef USE_CONSTANT_MEMORY

    wbCheck(cudaMemcpyToSymbol(dct_constant, dct, sizeof(dct)));
    wbCheck(cudaMemcpyToSymbol(Q_l_constant, Q_l, sizeof(Q_l)));
    wbCheck(cudaMemcpyToSymbol(Q_c_constant, Q_c, sizeof(Q_c)));
    wbCheck(cudaMemcpyToSymbol(zigzag_map_constant, zigzag_map, sizeof(zigzag_map)));
    #else
    wbCheck(cudaMemcpy(dct_device, dct, sizeof(dct), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(Q_l_device, Q_l, sizeof(Q_l), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(Q_c_device, Q_c, sizeof(Q_c), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(zigzag_map_device, zigzag_map, sizeof(zigzag_map), cudaMemcpyHostToDevice));
    #endif
    
    printf("Input image size: %dx%dx%d\n", imageWidth, imageHeight, imageChannels);
}

void Compressor::init_codewords() {
    codewords = &codewordsArray[2048]; // allow negative indices, so quantized[i] is at codewords[quantized[i]]
    uint8_t numBits = 1; // each codeword has at least one bit (value == 0 is undefined)
    int32_t mask = 1; // mask is always 2^numBits - 1, initial value 2^1-1 = 2-1 = 1
    for (int16_t value = 1; value < 2048; value++) {
        // numBits = position of highest set bit (ignoring the sign)
        // mask    = (2^numBits) - 1
        if (value > mask) // one more bit ?
        {
            numBits++;
            mask = (mask << 1) | 1; // append a set bit
        }
        codewords[-value] = BitCode(mask - value, numBits); // note that I use a negative index => codewords[-value] = codewordsArray[CodeWordLimit  value]
        codewords[+value] = BitCode(value, numBits);
    }
}

Compressor::~Compressor() {
    #ifdef PAGE_LOCK_HOST_BUFFERS
    cudaHostUnregister(hostInputImageData);
    #endif
    #ifndef SINGLE_GPU_BUFFER
    cudaFree(gpuScratch);
    #ifndef USE_CONSTANT_MEMORY
    cudaFree(dct_device);
    cudaFree(Q_l_device);
    cudaFree(Q_c_device);
    cudaFree(zigzag_map_device);
    #endif //USE_CONSTANT_MEMORY
    #endif // SINGLE_GPU_BUFFER
    wbImage_delete(inputImage);
}

void Compressor::compress(bool parallel) {
    ScopedTimer timer("Compression and encoding", 1);
    if (!parallel) {
        ScopedTimer seqCompTimer("Sequential compression", 2);
        sequential_compress_slice(origInputData, rearrangedData.data(), imageHeight);
    }
    else {
        ScopedTimer parCompTimer("Parallel compression", 2);
        std::vector<void*> scratchBufs;
        std::vector<std::unique_ptr<Stream>> streams;
        for (size_t ii = 0; ii < NUM_STREAMS; ++ii) {
            scratchBufs.push_back(gpuScratch + bytesPerLine*ii*LINES_PER_SLICE);
            streams.emplace_back(new Stream());
        }

        for (size_t startLine = 0, sliceIdx = 0; startLine < imageHeight; startLine += LINES_PER_SLICE, ++sliceIdx) {
            if (sliceIdx == NUM_STREAMS) sliceIdx = 0;
            parallel_compress_slice(scratchBufs[sliceIdx], startLine, LINES_PER_SLICE, *streams[sliceIdx]);
        }

        for (auto& stream : streams) {
            stream->synchronize();
        }
    }


    write_file();
}

void Compressor::generateHuffmanTable(const uint8_t numCodes[16], const uint8_t* values, BitCode result[256])
{
  // process all bitsizes 1 thru 16, no JPEG Huffman code is allowed to exceed 16 bits
  auto huffmanCode = 0;
  for (auto numBits = 1; numBits <= 16; numBits++)
  {
    // ... and each code of these bitsizes
    for (auto i = 0; i < numCodes[numBits - 1]; i++) // note: numCodes array starts at zero, but smallest bitsize is 1
      result[*values++] = BitCode(huffmanCode++, numBits);

    // next Huffman code needs to be one bit wider
    huffmanCode <<= 1;
  }
}

void Compressor::sequential_compress_slice(const float* inputData, T_Quant* outputData[3], size_t numLines) {

    size_t numPix = imageWidth * numLines;

    // convert from RGB floats to YCbCr in separate channels
    std::unique_ptr<float[]> 
        YData(new float[numPix]), 
        CbData(new float[numPix]),
        CrData(new float[numPix]);

    for (size_t i = 0; i < numPix; i++) {

        unsigned char r = (unsigned char) (255 * inputData[3*i + 0]);
        unsigned char g = (unsigned char) (255 * inputData[3*i + 1]);
        unsigned char b = (unsigned char) (255 * inputData[3*i + 2]);

        YData[i]  = 0   + (0.299   * r) + (0.587   * g) + (0.114   * b) - 128;
        CbData[i] = 128 - (0.16874 * r) - (0.33126 * g) + (0.5     * b) - 128;
        CrData[i] = 128 + (0.5     * r) - (0.41869 * g) - (0.08131 * b) - 128;

    }

    // TODO: subsample chrominance using 4:2:0 subsampling

    // discrete cosine transform
    std::unique_ptr<float[]>
        YDctData(new float[numPix]),
        CbDctData(new float[numPix]),
        CrDctData(new float[numPix]);

    sequential_dct(YData.get(),  YDctData.get(),  imageWidth, numLines);
    sequential_dct(CbData.get(), CbDctData.get(), imageWidth, numLines);
    sequential_dct(CrData.get(), CrDctData.get(), imageWidth, numLines);

    // quantization
    std::unique_ptr<T_Quant[]> 
        YQData(new T_Quant[numPix]),
        CbQData(new T_Quant[numPix]),
        CrQData(new T_Quant[numPix]);

    for (size_t j_block = 0; j_block < numLines /8; j_block++) {
        for (size_t i_block = 0; i_block < imageWidth/8; i_block++) {
            for (size_t j_tile = 0; j_tile < 8; j_tile++) {
                for (size_t i_tile = 0; i_tile < 8; i_tile++) {

                    size_t i = i_block * 8 + i_tile;
                    size_t j = j_block * 8 + j_tile;

                    YQData [j * imageWidth + i] = (T_Quant) round(YDctData [j * imageWidth + i] / Q_l[j_tile][i_tile]);
                    CbQData[j * imageWidth + i] = (T_Quant) round(CbDctData[j * imageWidth + i] / Q_c[j_tile][i_tile]);
                    CrQData[j * imageWidth + i] = (T_Quant) round(CrDctData[j * imageWidth + i] / Q_c[j_tile][i_tile]);
                }
            }
        }
    }

    // zigzag rearrange

    for (size_t j_block = 0; j_block < numLines/8; j_block++) {
        for (size_t i_block = 0; i_block < imageWidth/8; i_block++) {

            size_t block_num = j_block * imageWidth/8 + i_block;

            for (int j_tile = 0; j_tile < 8; j_tile++) {
                for (int i_tile = 0; i_tile < 8; i_tile++) {

                    size_t tile_num = j_tile * 8 + i_tile;

                    size_t x = i_block * 8 + i_tile;
                    size_t y = j_block * 8 + j_tile;

                    outputData[0][block_num * 64 + zigzag_map[tile_num]] = YQData [y * imageWidth + x];
                    outputData[1][block_num * 64 + zigzag_map[tile_num]] = CbQData[y * imageWidth + x];
                    outputData[2][block_num * 64 + zigzag_map[tile_num]] = CrQData[y * imageWidth + x];

                }
            }
        }
    }

}

void Compressor::sequential_dct(const float* inputData, float* outputData, int width, int height) {
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

void Compressor::parallel_compress_slice(void* gpuScratch, size_t startLine, size_t numLines, cudaStream_t stream) {
    numLines = std::min(numLines, imageHeight - startLine);

    size_t numPix = imageWidth * numLines;
    size_t numEl = numPix * imageChannels;

    //DevicePtr<float> deviceRGBImageData(numEl);
    //DevicePtr<T_Quant> deviceZigzagData(numEl);
    T_Input* deviceRGBImageData = (T_Input*)gpuScratch;
    T_Quant* deviceZigzagData = (T_Quant*)(deviceRGBImageData + numEl);

    T_Quant* deviceYZigzag = deviceZigzagData;
    T_Quant* deviceCbZigzag = deviceYZigzag + numPix;
    T_Quant* deviceCrZigzag = deviceCbZigzag + numPix;

    dim3 DimGrid1((numPix-1)/1024 + 1, 1, 1);
    dim3 DimBlock1(1024, 1, 1);

    dim3 DimGrid2((imageWidth-1)/8+1, (numLines-1)/8 + 1, 1);
    dim3 DimBlock2(8,8,1);

    dim3 DimGrid3((imageWidth - 1) / 8 + 1, (numLines - 1) / 8 + 1, 1);
    dim3 DimBlock3(8, 8, 3);

    wbCheck(cudaMemcpyAsync(deviceRGBImageData, hostInputImageData + startLine*imageWidth*imageChannels, numEl*sizeof(T_Input), cudaMemcpyHostToDevice, stream));

    if (combined) {
        #ifndef USE_CONSTANT_MEMORY
        Array<const uint8_t*, 3> Q_tables{ {Q_l_device, Q_c_device, Q_c_device} };
        #endif
        kernel_combined<<<DimGrid3, DimBlock3, 0, stream>>>(deviceRGBImageData, deviceZigzagData, 
        #ifndef USE_CONSTANT_MEMORY
            dct_device, Q_tables, zigzag_map_device, 
        #endif
            imageWidth, numLines);
    }
    else {
        DevicePtr<float> deviceYCbCrImageData(numEl);
        DevicePtr<float> deviceDCTData(numEl);
        DevicePtr<T_Quant> deviceQuantData(numEl);

        float* deviceY = deviceYCbCrImageData.get();
        float* deviceCb = deviceY + numPix;
        float* deviceCr = deviceCb + numPix;

        kernel_rgb_to_ycbcr<<<DimGrid1, DimBlock1, 0, stream >>>(deviceRGBImageData, deviceY, deviceCb, deviceCr, numPix);

        float* deviceYDCT = deviceDCTData.get();
        float* deviceCbDCT = deviceYDCT + numPix;
        float* deviceCrDCT = deviceCbDCT + numPix;

        T_Quant* deviceYQuant = deviceQuantData.get();
        T_Quant* deviceCbQuant = deviceYQuant + numPix;
        T_Quant* deviceCrQuant = deviceCbQuant + numPix;

        kernel_block_dct<<<DimGrid2, DimBlock2, 0, stream >>>(deviceY,  deviceYDCT,  dct_device, imageWidth, imageHeight);
        kernel_block_dct<<<DimGrid2, DimBlock2, 0, stream >>>(deviceCb, deviceCbDCT, dct_device, imageWidth, imageHeight);
        kernel_block_dct<<<DimGrid2, DimBlock2, 0, stream >>>(deviceCr, deviceCrDCT, dct_device, imageWidth, imageHeight);
    
        kernel_quantize_dct_output<<<DimGrid2, DimBlock2, 0, stream >>>(deviceYDCT,  deviceYQuant,  Q_l_device, imageWidth, imageHeight);
        kernel_quantize_dct_output<<<DimGrid2, DimBlock2, 0, stream >>>(deviceCbDCT, deviceCbQuant, Q_c_device, imageWidth, imageHeight);
        kernel_quantize_dct_output<<<DimGrid2, DimBlock2, 0, stream >>>(deviceCrDCT, deviceCrQuant, Q_c_device, imageWidth, imageHeight);
    
        kernel_zigzag << <DimGrid2, DimBlock2, 0, stream >> > (deviceYQuant,  deviceYZigzag,  zigzag_map_device, imageWidth, imageHeight);
        kernel_zigzag << <DimGrid2, DimBlock2, 0, stream >> > (deviceCbQuant, deviceCbZigzag, zigzag_map_device, imageWidth, imageHeight);
        kernel_zigzag << <DimGrid2, DimBlock2, 0, stream >> > (deviceCrQuant, deviceCrZigzag, zigzag_map_device, imageWidth, imageHeight);
    }

    size_t memcpySize = imageWidth * numLines * sizeof(*deviceYZigzag);
    wbCheck(cudaMemcpyAsync(rearrangedData[0] + startLine * imageWidth, deviceYZigzag,  memcpySize, cudaMemcpyDeviceToHost, stream));
    wbCheck(cudaMemcpyAsync(rearrangedData[1] + startLine * imageWidth, deviceCbZigzag, memcpySize, cudaMemcpyDeviceToHost, stream));
    wbCheck(cudaMemcpyAsync(rearrangedData[2] + startLine * imageWidth, deviceCrZigzag, memcpySize, cudaMemcpyDeviceToHost, stream));

}


void Compressor::write_file() {
    ScopedTimer timer("Encoding",2);

    if ((size_t)imageWidth * (size_t)imageHeight <= 1024) {
        for (size_t ii = 0, idx = 0; ii < imageHeight; ++ii) {
            for (size_t jj = 0; jj < imageWidth; ++jj, ++idx) {
                printf("%4d %4d %4d          ",
                    (int)rearrangedData[0][idx],
                    (int)rearrangedData[1][idx],
                    (int)rearrangedData[2][idx]);
            }
            printf("\n\n");
       }
    }

    const uint8_t HeaderJfif[2 + 2 + 16] = {
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
    bitWriter.addMarker(0xDB, 2 + 2 * (1 + 8 * 8));
    bitWriter << 0x00;
    for (auto i = 0; i < 64; i++) {
        auto y = ZigZagInv[i] / 8;
        auto x = ZigZagInv[i] % 8;
        bitWriter << Q_l[y][x];
    }
    bitWriter << 0x01;
    for (auto i = 0; i < 64; i++) {
        auto y = ZigZagInv[i] / 8;
        auto x = ZigZagInv[i] % 8;
        bitWriter << Q_c[y][x];
    }

    // Bits/Pixel, Image Size, Number of Channels, and Subsampling and Y vs C for each channel
    bitWriter.addMarker(0xC0, 2 + 6 + 3 * 3);
    bitWriter << 0x08
        << (imageHeight >> 8) << (imageHeight & 0xFF)
        << (imageWidth >> 8) << (imageWidth & 0xFF)
        << 0x03
        << 0x01 << 0x11 << 0x00
        << 0x02 << 0x11 << 0x01
        << 0x03 << 0x11 << 0x01;

    // Huffman Tables
    bitWriter.addMarker(0xC4, 2 + 208 + 208);
    bitWriter << 0x00 << DcLuminanceCodesPerBitsize << DcLuminanceValues
        << 0x10 << AcLuminanceCodesPerBitsize << AcLuminanceValues
        << 0x01 << DcChrominanceCodesPerBitsize << DcChrominanceValues
        << 0x11 << AcChrominanceCodesPerBitsize << AcChrominanceValues;

    // Start of Scan
    bitWriter.addMarker(0xDA, 2 + 1 + 2 * 3 + 3);

    // Number of Channels and Channel map to Huffman Tables
    bitWriter << 0x03
        << 0x01 << 0x00
        << 0x02 << 0x11
        << 0x03 << 0x11;

    // Spectral Selection - Single Scan
    bitWriter << 0x00 << 0x3F << 0x00;

    // TODO: Image Data
    // TODO: entropy coding
    int16_t lastYDC = 0;
    int16_t lastCbDC = 0;
    int16_t lastCrDC = 0;

    for (auto j_block = 0; j_block < imageHeight / 8; j_block++) {
        for (auto i_block = 0; i_block < imageWidth / 8; i_block++) {

            auto block_num = j_block * imageWidth / 8 + i_block;

            for (auto c = 0; c < imageChannels; c++) {

                auto it = rearrangedData[c] + block_num * 64;
                std::vector<T_Quant> block64(it, it + 64);

                BitCode* huffman = (c == 0) ? huffmanLuminanceDC : huffmanChrominanceDC;

                int16_t lastDC;
                if (c == 0) { lastDC = lastYDC;  lastYDC = block64[0]; }
                else if (c == 1) { lastDC = lastCbDC; lastCbDC = block64[0]; }
                else { lastDC = lastCrDC; lastCrDC = block64[0]; }

                auto diff = block64[0] - lastDC;
                if (diff == 0) {
                    bitWriter << huffman[0x00];
                }
                else {
                    auto bits = codewords[diff];
                    bitWriter << huffman[bits.numBits] << bits;
                }

                huffman = (c == 0) ? huffmanLuminanceAC : huffmanChrominanceAC;

                // find last non-zero value in block
                auto posNonZero = 0;
                for (auto i = 1; i < 64; i++) {
                    if (block64[i] != 0) posNonZero = i;
                }

                auto offset = 0;
                for (auto i = 1; i <= posNonZero; i++) {
                    // cound the preceding zeros before a nonzero value
                    while (block64[i] == 0) {
                        offset += 0x10;
                        // write a special symbol for 16 zeros and reset count
                        if (offset > 0xF0) {
                            bitWriter << huffman[0xF0];
                            offset = 0;
                        }
                        i++;
                    }

                    auto encoded = codewords[block64[i]];

                    // combine the run with the size of the symbol
                    bitWriter << huffman[offset + encoded.numBits] << encoded;
                    offset = 0;
                }

                // Write an EOB if the remaining values are zero
                if (posNonZero < 63) bitWriter << huffman[0x00];
            }
        }
    }

    bitWriter.flush();
    // End of Image

    bitWriter << 0xFF << 0xD9;
}

void Compressor::compress_nvjpeg(std::vector<unsigned char>& output) {
    
    #ifdef USE_NVJPEG

    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream = cudaStreamDefault; 

    DevicePtr<uint8_t> deviceInputData(imageWidth*imageHeight*imageChannels);

    // initialize nvjpeg structures
    checkNvJpeg(nvjpegCreateSimple(&nv_handle));
    ScopedTimer timer("Encoding with NVJPEG", 1);

    wbCheck(cudaMemcpy(deviceInputData.get(), hostInputImageData, imageWidth*imageHeight*imageChannels, cudaMemcpyHostToDevice));
    
    checkNvJpeg(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
    checkNvJpeg(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

    checkNvJpeg(nvjpegEncoderParamsSetQuality(nv_enc_params, 50, stream));
    checkNvJpeg(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream)); 
    nvjpegImage_t nv_image;
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      nv_image.channel[c] = NULL;
      nv_image.pitch[c] = 0;
    }
    nv_image.channel[0] = deviceInputData.get();
    nv_image.pitch[0] = imageWidth * imageChannels;

    // Compress image
    checkNvJpeg(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
        &nv_image, NVJPEG_INPUT_RGBI, imageWidth, imageHeight, stream));

    // get compressed stream size
    size_t length;
    checkNvJpeg(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
    // get stream itself
    wbCheck(cudaStreamSynchronize(stream));
    output.resize(length);
    checkNvJpeg(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, output.data(), &length, stream));
    wbCheck(cudaStreamSynchronize(stream));
    wbCheck(cudaDeviceSynchronize());
    #else
    (void)output;
    ERROR("compress_nvjpeg must be enabled by #define USE_NVJPEG and T_Input=uint8_t");
    #endif
}

int main(int argc, char **argv) {

    ScopedTimer timer("Total time");

    //To run in parallel, put --parallel at the END of the command line arguments
    int num_args = argc;
    bool parallel = false;
    #ifdef USE_COMBINED_KERNEL
    bool combined = true;
    #else
    bool combined  = false;
    #endif
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--parallel")) {
            parallel = true;
            num_args -= 1;
            printf("\tRunning Parallel version\n");
        }
    }

    wbArg_t args = wbArg_read(num_args, argv);

    std::ofstream outputFile;
    outputFile.open(wbArg_getOutputFile(args), std::ios_base::binary);

    if (!outputFile.is_open()) ERROR("Opening output file failed");
    std::vector<unsigned char> outputData;
    auto write_one_byte = [&outputData](unsigned char byte) { outputData.push_back(byte); };

    //Allocate on heap because Huffman table / codeword member variables are very large
    std::unique_ptr<Compressor> compressor(new Compressor(args, combined, write_one_byte));

    // Reserve enough in output buffer for very high-quality compression (2 bits/pixel) to avoid reallocation
    outputData.reserve(compressor->getNumPixels() / 4);

    compressor->compress(parallel);

    outputFile.write((const char*)outputData.data(), outputData.size());

    outputFile.close();
    
    compressor->compress_nvjpeg(outputData);
    
    return 0;
}
