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

    int width, height, num_components;
    unsigned char* data = stbi_load(argv[1], &width, &height, &num_components, 0);
    if (!data) ERROR("Unable to load input image");

    if (!(num_components == 1 || num_components == 3)) ERROR("Image must have 1 or 3 components");

    outfile = fopen(argv[2], "wb");

    if (!outfile) ERROR("Unable to open output image");

    bool ok = TooJpeg::writeJpeg(&write_one_byte, data, width, height, num_components != 1, 50, false, "EECS598 Project Output");

    if (!ok) ERROR("Error writing JPEG");

    fclose(outfile);
    return 0;
}