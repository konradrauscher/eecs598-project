#include "toojpeg.h"
#include <cstdlib>
#include <cstdio>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define ERROR(MSG) {fprintf(stderr, MSG "\n"); exit(1);}

FILE* outfile;

void write_one_byte(unsigned char byte) {
    fputc(byte, outfile);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        ERROR("usage: ./main input_image output_image.jpg");
    }

    #ifdef _WIN32
    fprintf(stderr, "WARNING: This code produces corrupted JPEGs on Windows for some reason, try WSL or something\n");
    #endif


    int width, height, num_components;
    unsigned char* data = stbi_load(argv[1], &width, &height, &num_components, 0);
    if (!data) ERROR("Unable to load input image");

    if (!(num_components == 1 || num_components == 3)) ERROR("Image must have 1 or 3 components");

    // 800x600 image
    //const auto width = 800;
    //const auto height = 600;
    //// RGB: one byte each for red, green, blue
    //const auto bytesPerPixel = 3;
    //auto num_components = 3;
    //// allocate memory
    //auto image = new unsigned char[width * height * bytesPerPixel];
    //auto data = image;
    //// create a nice color transition (replace with your code)
    //for (auto y = 0; y < height; y++)
    //    for (auto x = 0; x < width; x++) {
    //        // memory location of current pixel
    //        auto offset = (y * width + x) * bytesPerPixel;
    //        // red and green fade from 0 to 255, blue is always 127
    //        image[offset] = 255 * x / width;
    //        image[offset + 1] = 255 * y / height;
    //        image[offset + 2] = 127;
    //    }

    outfile = fopen(argv[2], "w");

    if (!outfile) ERROR("Unable to open output image");

    bool ok = TooJpeg::writeJpeg(&write_one_byte, data, width, height, num_components != 1, 90, false, "EECS598 Project Output");

    if (!ok) ERROR("Error writing JPEG");

    fclose(outfile);
    return 0;
}